"""Shared data access for GWDG incident telemetry and Slurm scheduler state.

Start by calling load_data_gwdg to select an incident CSV and its label dictionary.
Call get_resource_ids_gwdg early to discover node and device IDs in the window.
get_metrics_gwdg then selects categories, resources, metric names, and labels,
and returns separate series of raw samples.

The most recent window is cached with all metric categories. A node query followed
by a GPU query over the same window therefore reads the telemetry file only once.
The cache holds the requested rows in memory; it does not hold the whole incident
unless the requested window covers that incident.
"""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd

from src.dwdg.metric_metadata import (
    METRIC_METADATA, NODE_METRICS, GPU_METRICS, NETWORK_METRICS,
    STORAGE_METRICS, SCHEDULER_METRICS, MONITORING_METRICS,
)

MetricCategory = Literal["all", "node", "gpu", "network", "storage", "scheduler", "monitoring", "monitor"]
METRIC_CATEGORIES = {
    "all": None, "node": NODE_METRICS, "gpu": GPU_METRICS,
    "network": NETWORK_METRICS, "storage": STORAGE_METRICS,
    "scheduler": SCHEDULER_METRICS, "monitoring": MONITORING_METRICS,
    "monitor": MONITORING_METRICS,
}

# Dataset configuration and the CSV fields that distinguish concurrent series.
# parents[2] walks from src/dwdg/telemetry.py back to the project root.
DATA_DIR = Path(__file__).resolve().parents[2] / "data/GWDG_GPU_Node_Telemetry_Dataset_v2"
SOURCE = "GWDG GPU Node Telemetry Dataset v2"
SAMPLE_INTERVAL_SECONDS = 10
SCHEDULER_SAMPLE_INTERVAL_SECONDS = 600
# The metric name alone is insufficient: one metric can have several GPUs,
# CPUs, or sensors. These columns preserve the original dataset series identity.
IDENTITY_COLUMNS = ("metric", "gpu", "device", "uuid", "job", "modelName", "driverVersion", "seriesId")

# One selected source per process; changing it also clears the cached window.
node = None  # Node pseudonym extracted from the selected file's name.
_path = None  # Path to the selected CSV or compressed CSV.xz.
_labels = {}  # (metric, seriesId) -> labels such as {"cpu": "0", "mode": "idle"}.
_chunksize = 100_000  # Number of CSV rows read at a time, not a limit on query results.
_scheduler_path = None  # Optional override for the separate multi-node Slurm table.
_scheduler_chunksize = 100_000


# Source loading and cleanup.
def load_data_gwdg(path: str, *, series_labels_path: str | None = None, chunksize: int = 100_000):
    """Select one incident file and load its label dictionary. Relative paths use DATA_DIR."""
    global node, _path, _labels, _chunksize
    # Joining an absolute path leaves it absolute; a relative path uses DATA_DIR.
    path = (DATA_DIR / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if chunksize < 1:
        raise ValueError("chunksize must be positive")

    # Resolve CPU, mode, sensor, and controller labels from the published dictionary.
    labels_path = DATA_DIR / series_labels_path if series_labels_path else path.parent / "series_labels.tsv"
    if not series_labels_path and not labels_path.is_file():
        labels_path = DATA_DIR / "series_labels.tsv"
    # create labels from series_labels.tsv with (metric, seriesId) as keys
    dictionary = pd.read_csv(labels_path, sep="\t", dtype=str, na_filter=False)
    labels = {}
    for row in dictionary.itertuples(index=False):
        # Include the metric in the lookup key so labels are resolved in its context.
        key = (row.metric, row.seriesId)
        decoded = json.loads(row.labels)
        # Repeated identical definitions are harmless; contradictory ones are not.
        if key in labels and labels[key] != decoded:
            raise ValueError(f"Conflicting series labels for {key}")
        labels[key] = decoded

    node, _path, _labels, _chunksize = path.name.split("_", 1)[0], path, labels, chunksize
    read_window.cache_clear()  # Cached rows from the previous source are now invalid.
    read_scheduler_window.cache_clear()  # The default scheduler file may be beside this source.


def load_scheduler_data_gwdg(path: str = "slurm_node_state_tidy.csv.xz", *, chunksize: int = 100_000):
    """Select a Slurm table without replacing the selected incident file."""
    global _scheduler_path, _scheduler_chunksize
    path = (DATA_DIR / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    _scheduler_path, _scheduler_chunksize = path, chunksize
    read_scheduler_window.cache_clear()


def unload_data_gwdg():
    """Release the selected sources, dictionary, and cached DataFrames."""
    global node, _path, _labels, _scheduler_path, _chunksize, _scheduler_chunksize
    node, _path, _labels = None, None, {}
    _scheduler_path = None
    _chunksize = _scheduler_chunksize = 100_000
    read_window.cache_clear()
    read_scheduler_window.cache_clear()


def utc_time(value: str) -> pd.Timestamp:
    """Normalize a timezone-aware input to UTC without changing its instant."""
    timestamp = pd.Timestamp(value)
    # Reject missing or naive times rather than guessing a local timezone.
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Times must include Z or an explicit UTC offset")
    return timestamp.tz_convert("UTC")


@lru_cache(maxsize=1)
def read_window(node_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Read all categories once; follow-up tools reuse the most recent window.
    The cache key is (node_id, start, end);
    different metrics at the same time range and node_id can therefore reuse the DataFrame without re-reading the CSV.
    Treat the returned DataFrame as read-only. Load again to refresh a changed file.
    """
    if _path is None:
        raise RuntimeError("Call load_data_gwdg before querying metrics")
    # we only need the following columns
    columns = [*IDENTITY_COLUMNS, "timeUtc", "value"]
    # The selected file belongs to one node. Return empty table if no matches.
    if node_id != node:
        return pd.DataFrame(columns=columns)

    # Keep only the requested time range [start, end). A sample at start is included; a sample exactly at end is excluded.
    parts = []
    with pd.read_csv(_path, dtype=str, na_filter=False, usecols=columns, chunksize=_chunksize) as reader:
        for chunk in reader:
            chunk["timeUtc"] = pd.to_datetime(chunk["timeUtc"], utc=True)
            rows = chunk.loc[(chunk["timeUtc"] >= start) & (chunk["timeUtc"] < end)]
            if not rows.empty:
                parts.append(rows)
    # Combine matching chunks into one table. Return an empty table if no matches.
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def _scheduler_source_path() -> Path:
    """Resolve the selected Slurm table without exposing source details to tools."""
    if _scheduler_path is not None:
        return _scheduler_path
    path = (_path.parent if _path is not None else DATA_DIR) / "slurm_node_state_tidy.csv.xz"
    return path if path.is_file() else DATA_DIR / "slurm_node_state_tidy.csv.xz"


@lru_cache(maxsize=1)
def read_scheduler_window(node_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Read one node's Slurm rows, caching them independently of incident telemetry."""
    path = _scheduler_source_path()

    columns = [*IDENTITY_COLUMNS, "timeUtc", "value"]
    parts = []
    with pd.read_csv(path, dtype=str, na_filter=False, usecols=["node", *columns], chunksize=_scheduler_chunksize) as reader:
        for chunk in reader:
            # Unlike incident files, this table contains several nodes. Filter the
            # node column first, then apply the same inclusive/exclusive UTC bounds.
            chunk = chunk.loc[chunk["node"] == node_id].copy()
            if chunk.empty:
                continue
            chunk["timeUtc"] = pd.to_datetime(chunk["timeUtc"], utc=True)
            rows = chunk.loc[(chunk["timeUtc"] >= start) & (chunk["timeUtc"] < end), columns]
            if not rows.empty:
                parts.append(rows)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def series_labels(identity: dict) -> dict:
    """Combine dictionary labels with the original CSV resource labels."""
    labels = {}
    # CPU/mode, sensor, and controller labels often live only in the TSV dictionary.
    # Blank seriesId values are valid and do not require a dictionary entry.
    if identity["seriesId"]:
        key = (identity["metric"], identity["seriesId"])
        if key not in _labels:
            raise ValueError(f"No series_labels.tsv entry for {key}")
        labels.update(_labels[key])  # Copy labels so we never modify the dictionary.
    # Exclude metric and seriesId: they have their own fields in the response.
    for name in IDENTITY_COLUMNS[1:-1]:
        value = identity[name]
        if value:  # Empty CSV fields are absent labels, not meaningful identifiers.
            if name in labels and labels[name] != value:
                raise ValueError(f"CSV and dictionary labels disagree for {name}")
            labels[name] = value
    return dict(sorted(labels.items()))  # Keep the output order consistent.


def get_resources(rows: pd.DataFrame) -> dict:
    """Return observed IDs that follow-up tools can match through uuid/device labels.
    Thoses ids are gpu_ids, network_interface_ids, infiniband_adapter_ids, storage_ids,
    and cpus (with logical_cpu_count and logical_cpu_ids).
    """
    resource_sets = {
        "gpu_ids": set(), "network_interface_ids": set(),
        "infiniband_adapter_ids": set(), "storage_ids": set(),
    }
    families = (
        ("DCGM_", "uuid", "gpu_ids"),
        ("node_network_", "device", "network_interface_ids"),
        ("node_infiniband_", "device", "infiniband_adapter_ids"),
        (("node_disk_", "node_filesystem_", "node_nvme_"), "device", "storage_ids"),
    )
    cpu_ids = set()
    # Resolve each identity once, including IDs present only in the dictionary.
    for identity in rows[list(IDENTITY_COLUMNS)].drop_duplicates().to_dict("records"):
        labels = series_labels(identity)
        cpu = labels.get("cpu")
        if cpu is not None:
            cpu_ids.add(int(cpu))
        for prefix, label, key in families:
            if identity["metric"].startswith(prefix) and labels.get(label):
                resource_sets[key].add(labels[label])
    resources = {key: sorted(values) for key, values in resource_sets.items()}
    # Observed logical indices do not describe physical CPU topology.
    if cpu_ids:
        resources["cpus"] = {"logical_cpu_count": len(cpu_ids), "logical_cpu_ids": sorted(cpu_ids)}
    return resources


def evidence_id(kind: str, identity: dict) -> str:
    """Combine a readable metric mnemonic with a stable evidence-scope hash."""
    # Canonical JSON ignores dictionary insertion order. The caller supplies the
    # dataset, node, metric, window, and (for series) full identity and labels.
    # Values are not hashed: this ID identifies the evidence scope, not a checksum.
    # The hash cannot be decoded. Read the metric from the short name, and use
    # the returned window and labels to understand the scope behind the hash.
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    short_name = METRIC_METADATA.get(identity["metric"], {}).get("short_name", "UNKNOWN")
    return kind + "-" + short_name + "-" + hashlib.sha256(text.encode()).hexdigest()[:32]


def sample_value(text: str) -> int | float:
    """return an integer or float from a CSV string, rejecting NaN and infinity.
    """
    try:
        return int(text)
    except ValueError:
        # Decimal and scientific-notation samples require a float.
        value = float(text)
        # Standard JSON has no numeric representation for NaN or infinity.
        # Reject them explicitly instead of replacing them with zero or null.
        if not math.isfinite(value):
            raise ValueError(f"Non-finite sample cannot be represented in JSON: {text}")
        return value


def _query_window(start_time: str, end_time: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = utc_time(start_time), utc_time(end_time)
    if start >= end:
        raise ValueError("start_time must be earlier than end_time")
    return start, end


def get_resource_ids_gwdg(start_time: str, end_time: str) -> dict:
    """Discover node/device IDs before metric queries, without returning samples.

    With an incident selected, inventory covers that node's full window and warms
    its shared query cache. Without an incident, discover all nodes in the selected
    scheduler table (or the default scheduler source).
    """
    start, end = _query_window(start_time, end_time)
    inventories = {}
    if _path is not None:
        rows = read_window(node, start, end)
        if not rows.empty:
            inventories[node] = get_resources(rows)
    else:
        # Discover all scheduler nodes in one pass, retaining only identities.
        parts = []
        with pd.read_csv(
            _scheduler_source_path(), dtype=str, na_filter=False,
            usecols=["node", "timeUtc", *IDENTITY_COLUMNS], chunksize=_scheduler_chunksize,
        ) as reader:
            for chunk in reader:
                times = pd.to_datetime(chunk["timeUtc"], utc=True)
                rows = chunk.loc[(times >= start) & (times < end), ["node", *IDENTITY_COLUMNS]]
                if not rows.empty:
                    parts.append(rows.drop_duplicates())
        if parts:
            rows = pd.concat(parts, ignore_index=True).drop_duplicates()
            for node_id, group in rows.groupby("node", sort=True):
                if node_id:
                    inventories[node_id] = get_resources(group)
    resource_ids = set(inventories)
    for resources in inventories.values():
        for key in ("gpu_ids", "network_interface_ids", "infiniband_adapter_ids", "storage_ids"):
            resource_ids.update(resources[key])
    return {
        "source": SOURCE,
        "node_list": sorted(inventories),
        "resource_ids": sorted(resource_ids),
        "nodes": {
            node_id: {"resources": {"node_id": node_id, **inventories[node_id]}}
            for node_id in sorted(inventories)
        },
    }


def _category_metrics(metrics, category):
    """Select a category centrally; all also permits unregistered metric names."""
    if not isinstance(category, str) or category not in METRIC_CATEGORIES:
        raise ValueError(f"Unsupported metric category: {category}")
    allowed = METRIC_CATEGORIES[category]
    if metrics is None:
        return list(allowed) if allowed is not None else None
    if not isinstance(metrics, (list, tuple)) or not all(isinstance(name, str) for name in metrics):
        raise ValueError("metrics must be a list of strings")
    if allowed is not None:
        unsupported = set(metrics) - set(allowed)
        if unsupported:
            raise ValueError(f"Unsupported {category} metrics: {sorted(unsupported)}")
    return metrics


def _filter_device_metrics(result, node_id, resource_id, category):
    """Filter resource series while retaining node-wide metrics and inventory."""
    if resource_id is None or node_id not in result["nodes"]:
        return result
    if category == "all" and resource_id == node_id:
        return result
    metrics = result["nodes"][node_id]["metrics"]
    for name in list(metrics):
        label = None
        if name.startswith("DCGM_"):
            label = "uuid"
        elif name.startswith(("node_network_", "node_infiniband_", "node_disk_", "node_filesystem_", "node_nvme_")):
            label = "device"
        if label:
            metrics[name]["series"] = [
                series for series in metrics[name]["series"]
                if series["labels"].get(label) == resource_id
            ]
            if not metrics[name]["series"]:
                del metrics[name]
    return result


# Public metric query and result formatting.
def get_metrics_gwdg(
    node_id: str,
    start_time: str,
    end_time: str,
    metrics: list[str] | None = None,
    label_filters: dict[str, str] | None = None,
    category: MetricCategory = "all",
    resource_id: str | None = None,
) -> dict:
    """Return raw category metrics with optional names, resource and label filters.

    all reads the selected incident, including unregistered metrics. scheduler
    reads the separate Slurm source at its native cadence. None selects the full
    category and [] selects no metrics. Device filtering retains node-wide series.
    """
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("node_id must be a non-empty string")
    if resource_id is not None and (not isinstance(resource_id, str) or not resource_id):
        raise ValueError("resource_id must be a non-empty string")
    metrics = _category_metrics(metrics, category)
    if category in ("node", "scheduler", "monitoring", "monitor") and resource_id not in (None, node_id):
        raise ValueError(f"resource_id must be the node ID for {category} metrics")
    if label_filters is not None and (
        not isinstance(label_filters, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in label_filters.items())
    ):
        raise ValueError("label_filters must map labels to strings")
    start, end = _query_window(start_time, end_time)
    if category == "scheduler":
        rows = read_scheduler_window(node_id, start, end)
        interval = SCHEDULER_SAMPLE_INTERVAL_SECONDS
    else:
        rows = read_window(node_id, start, end)
        interval = SAMPLE_INTERVAL_SECONDS
    if metrics is None:
        metrics = rows["metric"].unique().tolist()
    result = format_metrics(rows, node_id, start, end, metrics, label_filters, interval)
    return _filter_device_metrics(result, node_id, resource_id, category)


def expected_sample_count(times, start, end, interval_seconds):
    """Count native slots using the modal observed phase (earliest phase breaks ties).

    This is an estimate of a fixed schedule, including leading/trailing gaps.
    Jitter does not add expected samples; observations remain untouched.
    """
    interval = pd.Timedelta(seconds=interval_seconds)
    phases = pd.Series([time.value % interval.value for time in set(times)])
    counts = phases.value_counts()
    phase = int(min(counts[counts == counts.max()].index))
    offset = pd.Timedelta((phase - start.value) % interval.value, unit="ns")
    first = start + offset
    return max(0, -((first.value - end.value) // interval.value))


def format_metrics(
    rows: pd.DataFrame,
    node_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    metrics: list[str],
    label_filters: dict[str, str] | None = None,
    sample_interval_seconds: int = SAMPLE_INTERVAL_SECONDS,
) -> dict:
    """Format either source as raw series, keeping its native sampling cadence."""
    # Start at the first native slot inside the window: 10 seconds for incident
    # telemetry, 600 seconds for Slurm. Missing scrapes still have axis slots.
    frequency = f"{sample_interval_seconds}s"
    timestamps = pd.date_range(start.ceil(frequency), end, freq=frequency, inclusive="left")
    if not rows.empty:
        # union adds any observations outside the native grid without duplicating
        # timestamps. Sorting gives every series the same chronological axis.
        timestamps = timestamps.union(pd.DatetimeIndex(rows["timeUtc"].unique())).sort_values()
    # ISO timestamps use Z for UTC. These normalized bounds also enter evidence IDs.
    window = {
        "start_time": start.isoformat().replace("+00:00", "Z"),
        "end_time": end.isoformat().replace("+00:00", "Z"),
        "end_time_exclusive": True,
        "sample_interval_seconds": sample_interval_seconds,
    }
    # Start with an empty response so unknown nodes and empty windows have a
    # consistent shape. Metric selection alone does not determine node presence.
    result = {
        "source": SOURCE, "node_list": [], "window": window,
        "sample_fields": ["timestamp_index", "value"],
        "sample_encoding": (
            "timestamp_index is a zero-based index into timestamps. Absent indices mean "
            "no observation for that series, not zero. The shared axis may include other "
            "series' timestamps. Coverage uses native cadence with phase inferred per series; "
            "returned_samples counts raw rows, including duplicates."
        ),
        "timestamps": [t.isoformat().replace("+00:00", "Z") for t in timestamps], "nodes": {},
    }
    if rows.empty:
        return result

    # 3. Discover resources before filtering, so IDs from other categories remain
    # available for follow-up queries even when only node metrics were requested.
    resources = get_resources(rows)
    selected = rows.loc[rows["metric"].isin(metrics)]
    # Example: {09:00:00: 0, 09:00:10: 1}. A sample [1, 42] means value 42
    # at timestamps[1]; this avoids repeating full timestamps in every series.
    timestamp_indices = {time: index for index, time in enumerate(timestamps)}
    returned_metrics = {}
    # 4. Group observations into separate series, rather than averaging them.
    for values, group in selected.groupby(list(IDENTITY_COLUMNS), sort=True):
        # groupby returns one tuple of identity values plus that series' rows.
        # zip turns the tuple back into a dictionary with the original field names.
        identity = dict(zip(IDENTITY_COLUMNS, values))
        labels = series_labels(identity)
        # All filters must match. A missing label fails the match; no filters means
        # every series passes. For example, cpu=0 AND mode=idle selects one CPU mode.
        if any(labels.get(name) != value for name, value in (label_filters or {}).items()):
            continue
        # 5. Create the metric entry once, then append its individual series.
        name = identity["metric"]
        metric_identity = {"dataset": SOURCE, "node": node_id, "metric": name, "window": window}
        if name not in returned_metrics:
            # Metadata is static. Preserve readings from other categories even
            # when their type and unit have not been verified in the registry.
            metadata = METRIC_METADATA.get(name, {
                "short_name": "UNKNOWN", "type": "unknown", "unit": None,
                "note": "Metadata unavailable for this metric.",
            })
            returned_metrics[name] = {
                "evidence_id": evidence_id("metric", metric_identity), **metadata, "series": [],
            }
        # 6. Convert and sort the raw samples. Sorting preserves duplicates and
        # does not fill gaps, average readings, or calculate counter rates.
        samples = sorted(zip(group["timeUtc"], map(sample_value, group["value"])))
        returned_metrics[name]["series"].append({
            # Series evidence adds the complete identity and resolved labels to
            # the metric scope, distinguishing CPUs, sensors, and other resources.
            "evidence_id": evidence_id("series", {**metric_identity, "identity": identity, "labels": labels}),
            "series_id": identity["seriesId"] or None,  # Blank dataset IDs become JSON null.
            "labels": labels,
            # Infer phase from this series, never from the union of other series.
            "coverage": {
                "expected_samples": expected_sample_count(
                    [time for time, _ in samples], start, end, sample_interval_seconds),
                "returned_samples": len(samples),
            },
            "samples": [[timestamp_indices[time], value] for time, value in samples],
        })
    # 7. Attach the node and its metrics. Keep node_list consistent with nodes' keys.
    result["node_list"] = [node_id]
    result["nodes"][node_id] = {"resources": resources, "metrics": returned_metrics}
    return result
