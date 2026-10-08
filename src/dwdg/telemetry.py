"""Shared data access for GWDG node, GPU, network, and storage tools.

Start by calling load_data_gwdg to select an incident CSV and its label dictionary.
Each get_metrics_gwdg call then reads a node's time window, discovers its resources,
selects the requested metrics and labels, and returns separate series of raw samples.

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

import pandas as pd

from src.dwdg.node_metadata import NODE_METRIC_METADATA

# Dataset configuration and the CSV fields that distinguish concurrent series.
# parents[2] walks from src/dwdg/telemetry.py back to the project root.
DATA_DIR = Path(__file__).resolve().parents[2] / "data/GWDG_GPU_Node_Telemetry_Dataset_v2"
SOURCE = "GWDG GPU Node Telemetry Dataset v2"
SAMPLE_INTERVAL_SECONDS = 10
# The metric name alone is insufficient: one metric can have several GPUs,
# CPUs, or sensors. These columns preserve the original dataset series identity.
IDENTITY_COLUMNS = ("metric", "gpu", "device", "uuid", "job", "modelName", "driverVersion", "seriesId")

# One selected source per process; changing it also clears the cached window.
node = None  # Node pseudonym extracted from the selected file's name.
_path = None  # Path to the selected CSV or compressed CSV.xz.
_labels = {}  # (metric, seriesId) -> labels such as {"cpu": "0", "mode": "idle"}.
_chunksize = 100_000  # Number of CSV rows read at a time, not a limit on query results.


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


def unload_data_gwdg():
    """Release the selected source, dictionary, and cached DataFrame."""
    global node, _path, _labels
    node, _path, _labels = None, None, {}
    read_window.cache_clear()


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
    def ids(prefix, column):
        # Restrict by metric family before collecting IDs: a device label can mean
        # a GPU, network interface, adapter, or disk depending on its metric.
        values = rows.loc[rows["metric"].str.startswith(prefix), column]
        return sorted(value for value in values.unique() if value)

    # Count observed logical CPU indices; these do not describe physical topology.
    cpu_ids = set()
    # Each series repeats its identity at every timestamp. Resolve it once here.
    identities = rows[list(IDENTITY_COLUMNS)].drop_duplicates().to_dict("records")
    for identity in identities:
        cpu = series_labels(identity).get("cpu")
        if cpu is not None:
            cpu_ids.add(int(cpu))

    resources = {
        "gpu_ids": ids("DCGM_", "uuid"),
        "network_interface_ids": ids("node_network_", "device"),
        "infiniband_adapter_ids": ids("node_infiniband_", "device"),
        "storage_ids": ids("node_disk_", "device"),
    }
    # Do not invent a zero CPU count when no CPU labels were observed.
    if cpu_ids:
        resources["cpus"] = {"logical_cpu_count": len(cpu_ids), "logical_cpu_ids": sorted(cpu_ids)}
    return resources


def evidence_id(kind: str, identity: dict) -> str:
    """Hash an evidence scope so equivalent queries produce the same identifier."""
    # Canonical JSON ignores dictionary insertion order. The caller supplies the
    # dataset, node, metric, window, and (for series) full identity and labels.
    # Values are not hashed: this ID identifies the evidence scope, not a checksum.
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return kind + "-" + hashlib.sha256(text.encode()).hexdigest()[:32]


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


# Public metric query and result formatting.
def get_metrics_gwdg(
    node_id: str,
    start_time: str,
    end_time: str,
    metrics: list[str],
    label_filters: dict[str, str] | None = None,
) -> dict:
    """Return named metrics from any category in [start_time, end_time).

    Optional label filters match all supplied labels exactly. Metrics without
    registered metadata retain unknown type and null unit. Unobserved metric
    names return no readings. The most recent window is shared with node queries.
    """
    # 1. Validate the selectors and normalize the inclusive start / exclusive end.
    # An empty metric list is valid and selects no readings.
    if not isinstance(metrics, (list, tuple)) or not all(isinstance(v, str) for v in metrics):
        raise ValueError("metrics must be a list of strings")
    if label_filters is not None and (
        not isinstance(label_filters, dict) or not all(isinstance(v, str) for v in label_filters.values())
    ):
        raise ValueError("label_filters must map labels to strings")
    start, end = utc_time(start_time), utc_time(end_time)
    if start >= end:
        raise ValueError("start_time must be earlier than end_time")

    # 2. Read the window and build the shared time axis.
    rows = read_window(node_id, start, end)
    # ceil("10s") starts at the first native slot inside the window. For a start
    # of 09:00:05, the first slot is 09:00:10. Missing scrapes still have axis slots.
    timestamps = pd.date_range(start.ceil("10s"), end, freq="10s", inclusive="left")
    if not rows.empty:
        # union adds any observations outside the native grid without duplicating
        # timestamps. Sorting gives every series the same chronological axis.
        timestamps = timestamps.union(pd.DatetimeIndex(rows["timeUtc"].unique())).sort_values()
    # ISO timestamps use Z for UTC. These normalized bounds also enter evidence IDs.
    window = {
        "start_time": start.isoformat().replace("+00:00", "Z"),
        "end_time": end.isoformat().replace("+00:00", "Z"),
        "end_time_exclusive": True,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
    }
    # Start with an empty response so unknown nodes and empty windows have a
    # consistent shape. Metric selection alone does not determine node presence.
    result = {
        "source": SOURCE, "node_list": [], "window": window,
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
            metadata = NODE_METRIC_METADATA.get(name, {
                "type": "unknown", "unit": None, "note": "Metadata unavailable for this metric.",
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
            # Expected samples count axis slots; returned samples count actual rows.
            # A gap lowers the returned count. Duplicate rows are counted separately.
            "coverage": {"expected_samples": len(timestamps), "returned_samples": len(samples)},
            "samples": [[timestamp_indices[time], value] for time, value in samples],
        })
    # 7. Attach the node and its metrics. Keep node_list consistent with nodes' keys.
    result["node_list"] = [node_id]
    result["nodes"][node_id] = {"resources": resources, "metrics": returned_metrics}
    return result
