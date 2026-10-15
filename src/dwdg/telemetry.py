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

from src.dwdg.execution import DataValidationError, ToolArgumentError
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

class GwdgDataSource:
    """One investigation's files, labels and two single-window caches.

    Files are assumed unchanged for this object's lifetime. Construct a new source
    after editing a fixture. Revision fingerprints use file metadata, not content
    hashes, which is sufficient for local test datasets.
    """

    def __init__(self, path: str | Path | None = None, *, series_labels_path: str | Path | None = None,
                 scheduler_path: str | Path | None = None, chunksize: int = 100_000,
                 scheduler_chunksize: int = 100_000):
        if chunksize < 1 or scheduler_chunksize < 1:
            raise ValueError("chunksize must be positive")
        self.path = (DATA_DIR / path).resolve() if path is not None else None
        if self.path is not None and not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.node = self.path.name.split("_", 1)[0] if self.path else None
        self.chunksize = chunksize
        self.scheduler_chunksize = scheduler_chunksize
        self.labels_path = None
        self.labels = {}
        if self.path is not None:
            labels_path = DATA_DIR / series_labels_path if series_labels_path else self.path.parent / "series_labels.tsv"
            if not series_labels_path and not labels_path.is_file():
                labels_path = DATA_DIR / "series_labels.tsv"
            self.labels_path = labels_path.resolve()
            dictionary = pd.read_csv(self.labels_path, sep="\t", dtype=str, na_filter=False)
            for row in dictionary.itertuples(index=False):
                key = (row.metric, row.seriesId)
                decoded = json.loads(row.labels)
                if key in self.labels and self.labels[key] != decoded:
                    raise DataValidationError(f"Conflicting series labels for {key}")
                self.labels[key] = decoded

        # Resolve scheduler fallback once, so later default-source changes cannot
        # redirect an existing investigation.
        self.scheduler_override = scheduler_path
        if scheduler_path is not None:
            self.scheduler_path = (DATA_DIR / scheduler_path).resolve()
            if not self.scheduler_path.is_file():
                raise FileNotFoundError(self.scheduler_path)
        else:
            candidate = (self.path.parent if self.path else DATA_DIR) / "slurm_node_state_tidy.csv.xz"
            self.scheduler_path = (candidate if candidate.is_file() else DATA_DIR / candidate.name).resolve()

        self.dataset_id = SOURCE
        files = []
        for file in (self.path, self.labels_path, self.scheduler_path):
            if file is None:
                files.append(None)
                continue
            try:
                stat = file.stat()
                files.append((str(file), stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:
                files.append((str(file), None, None))  # Optional scheduler is absent.
        self.revision = hashlib.sha256(json.dumps(files).encode()).hexdigest()
        # Instance-owned wrappers avoid a class-level cache retaining other runs.
        self._incident_cache = lru_cache(maxsize=1)(self._read_window)
        self._scheduler_cache = lru_cache(maxsize=1)(self._read_window)

    def read_window(self, node_id, start, end, *, scheduler=False):
        kind = "scheduler" if scheduler else "incident"
        key = (self.dataset_id, self.revision, kind, node_id, utc_time(start), utc_time(end))
        cache = self._scheduler_cache if scheduler else self._incident_cache
        return cache(key)

    def _read_window(self, key):
        _, _, kind, node_id, start, end = key
        scheduler = kind == "scheduler"
        columns = [*IDENTITY_COLUMNS, "timeUtc", "value"]
        if not scheduler:
            if self.path is None:
                raise RuntimeError("Call load_data_gwdg before querying metrics")
            if node_id != self.node:
                return pd.DataFrame(columns=columns)
        path = self.scheduler_path if scheduler else self.path
        chunksize = self.scheduler_chunksize if scheduler else self.chunksize
        parts = []
        with pd.read_csv(path, dtype=str, na_filter=False,
                         usecols=(["node"] if scheduler else []) + columns,
                         chunksize=chunksize) as reader:
            for chunk in reader:
                if scheduler:
                    chunk = chunk.loc[chunk["node"] == node_id].copy()
                    if chunk.empty:
                        continue
                chunk["timeUtc"] = pd.to_datetime(chunk["timeUtc"], utc=True)
                rows = chunk.loc[(chunk["timeUtc"] >= start) & (chunk["timeUtc"] < end), columns]
                if not rows.empty:
                    parts.append(rows)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)

    def clear_cache(self):
        """Release cached windows without affecting another source."""
        self._incident_cache.cache_clear()
        self._scheduler_cache.cache_clear()


# Compatibility for sequential callers. Investigations pass their own source.
_default_source = None


def _get_source(source=None):
    global _default_source
    if source is not None:
        return source
    if _default_source is None:
        _default_source = GwdgDataSource()
    return _default_source


def load_data_gwdg(path: str, *, series_labels_path: str | None = None, chunksize: int = 100_000):
    """Select and return a default source for sequential callers; old objects stay valid."""
    global _default_source
    previous = _default_source
    source = GwdgDataSource(
        path, series_labels_path=series_labels_path, chunksize=chunksize,
        scheduler_path=previous.scheduler_override if previous else None,
        scheduler_chunksize=previous.scheduler_chunksize if previous else 100_000,
    )
    _default_source = source
    return source


def load_scheduler_data_gwdg(path: str = "slurm_node_state_tidy.csv.xz", *, chunksize: int = 100_000):
    """Replace the default source's scheduler selection, preserving its incident path."""
    global _default_source
    previous = _default_source
    source = GwdgDataSource(
        previous.path if previous else None,
        series_labels_path=previous.labels_path if previous else None,
        chunksize=previous.chunksize if previous else 100_000,
        scheduler_path=path, scheduler_chunksize=chunksize,
    )
    _default_source = source
    return source


def unload_data_gwdg():
    """Drop the default source. Explicit investigation sources remain usable."""
    global _default_source
    _default_source = None


def utc_time(value: str) -> pd.Timestamp:
    """Normalize a timezone-aware input to UTC without changing its instant."""
    try:
        timestamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ToolArgumentError("Invalid timestamp; use ISO 8601 with Z or an explicit UTC offset.") from error
    # Reject missing or naive times rather than guessing a local timezone.
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ToolArgumentError("Times must include Z or an explicit UTC offset")
    return timestamp.tz_convert("UTC")


def read_window(node_id: str, start: pd.Timestamp, end: pd.Timestamp, *, source=None) -> pd.DataFrame:
    """Read all categories using the selected source's incident cache."""
    return _get_source(source).read_window(node_id, start, end)


def read_scheduler_window(node_id: str, start: pd.Timestamp, end: pd.Timestamp, *, source=None) -> pd.DataFrame:
    """Read Slurm rows using the selected source's separate scheduler cache."""
    return _get_source(source).read_window(node_id, start, end, scheduler=True)


def series_labels(identity: dict, *, source=None) -> dict:
    """Combine dictionary labels with the original CSV resource labels."""
    dictionary = _get_source(source).labels
    labels = {}
    # CPU/mode, sensor, and controller labels often live only in the TSV dictionary.
    # Blank seriesId values are valid and do not require a dictionary entry.
    if identity["seriesId"]:
        key = (identity["metric"], identity["seriesId"])
        if key not in dictionary:
            raise DataValidationError(f"No series_labels.tsv entry for {key}")
        labels.update(dictionary[key])  # Copy labels so we never modify the dictionary.
    # Exclude metric and seriesId: they have their own fields in the response.
    for name in IDENTITY_COLUMNS[1:-1]:
        value = identity[name]
        if value:  # Empty CSV fields are absent labels, not meaningful identifiers.
            if name in labels and labels[name] != value:
                raise DataValidationError(f"CSV and dictionary labels disagree for {name}")
            labels[name] = value
    return dict(sorted(labels.items()))  # Keep the output order consistent.


def get_resources(rows: pd.DataFrame, *, source=None) -> dict:
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
        labels = series_labels(identity, source=source)
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
        try:
            value = float(text)
        except (TypeError, ValueError) as error:
            raise DataValidationError("Sample value is not numeric") from error
        # Standard JSON has no numeric representation for NaN or infinity.
        # Reject them explicitly instead of replacing them with zero or null.
        if not math.isfinite(value):
            raise DataValidationError(f"Non-finite sample cannot be represented in JSON: {text}")
        return value


def _query_window(start_time: str, end_time: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = utc_time(start_time), utc_time(end_time)
    if start >= end:
        raise ToolArgumentError("start_time must be earlier than end_time")
    return start, end


def get_resource_ids_gwdg(start_time: str, end_time: str, *, source=None) -> dict:
    """Discover node/device IDs before metric queries, without returning samples.

    With an incident selected, inventory covers that node's full window and warms
    its shared query cache. Without an incident, discover all nodes in the selected
    scheduler table (or the default scheduler source).
    """
    start, end = _query_window(start_time, end_time)
    source = _get_source(source)
    inventories = {}
    if source.path is not None:
        rows = read_window(source.node, start, end, source=source)
        if not rows.empty:
            inventories[source.node] = get_resources(rows, source=source)
    else:
        # Discover all scheduler nodes in one pass, retaining only identities.
        parts = []
        with pd.read_csv(
            source.scheduler_path, dtype=str, na_filter=False,
            usecols=["node", "timeUtc", *IDENTITY_COLUMNS], chunksize=source.scheduler_chunksize,
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
                    inventories[node_id] = get_resources(group, source=source)
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
        raise ToolArgumentError(f"Unsupported metric category: {category}",
                                details={"available_categories": list(METRIC_CATEGORIES)})
    allowed = METRIC_CATEGORIES[category]
    if metrics is None:
        return list(allowed) if allowed is not None else None
    if not isinstance(metrics, (list, tuple)) or not all(isinstance(name, str) for name in metrics):
        raise ToolArgumentError("metrics must be a list of strings")
    if allowed is not None:
        unsupported = set(metrics) - set(allowed)
        if unsupported:
            supported = sorted(allowed)
            raise ToolArgumentError(f"Unsupported {category} metrics: {sorted(unsupported)}",
                                    code="unsupported_metrics", details={
                                        "available_metrics": supported[:64],
                                        "omitted_metrics": max(0, len(supported) - 64),
                                    })
    return metrics


def _filter_device_metrics(result, node_id, resource_id, category):
    """Filter resource series while retaining node-wide metrics."""
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
    *, source: GwdgDataSource | None = None,
) -> dict:
    """Return raw category metrics with optional names, resource and label filters.

    all reads the selected incident, including unregistered metrics. scheduler
    reads the separate Slurm source at its native cadence. None selects the full
    category and [] selects no metrics. Device filtering retains node-wide series.
    """
    if not isinstance(node_id, str) or not node_id:
        raise ToolArgumentError("node_id must be a non-empty string")
    if resource_id is not None and (not isinstance(resource_id, str) or not resource_id):
        raise ToolArgumentError("resource_id must be a non-empty string")
    metrics = _category_metrics(metrics, category)
    if category in ("node", "scheduler", "monitoring", "monitor") and resource_id not in (None, node_id):
        raise ToolArgumentError(f"resource_id must be the node ID for {category} metrics")
    if label_filters is not None and (
        not isinstance(label_filters, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in label_filters.items())
    ):
        raise ToolArgumentError("label_filters must map labels to strings")
    source = _get_source(source)
    start, end = _query_window(start_time, end_time)
    if category == "scheduler":
        rows = read_scheduler_window(node_id, start, end, source=source)
        interval = SCHEDULER_SAMPLE_INTERVAL_SECONDS
    else:
        rows = read_window(node_id, start, end, source=source)
        interval = SAMPLE_INTERVAL_SECONDS
    if metrics is None:
        metrics = rows["metric"].unique().tolist()
    result = format_metrics(rows, node_id, start, end, metrics, label_filters, interval, source=source)
    return _filter_device_metrics(result, node_id, resource_id, category)


def _sample_schedule(times, start, end, interval_seconds):
    """Find native slots using the modal observed phase (earliest phase breaks ties).

    This is an estimate of a fixed schedule, including leading/trailing gaps.
    Jitter does not add expected samples; observations remain untouched.
    """
    interval = pd.Timedelta(seconds=interval_seconds)
    phases = pd.Series([time.value % interval.value for time in set(times)])
    counts = phases.value_counts()
    phase = int(min(counts[counts == counts.max()].index))
    offset = pd.Timedelta((phase - start.value) % interval.value, unit="ns")
    first = start + offset
    count = max(0, -((first.value - end.value) // interval.value))
    return first, count


def expected_sample_count(times, start, end, interval_seconds):
    """Count expected observations on the series' inferred native schedule."""
    return _sample_schedule(times, start, end, interval_seconds)[1]


def sample_time(time_axis: dict, step_index: int | float) -> pd.Timestamp:
    """Decode start + step_seconds * step_index, including off-grid observations."""
    interval_ns = pd.Timedelta(seconds=time_axis["step_seconds"]).value
    return utc_time(time_axis["start"]) + pd.Timedelta(round(step_index * interval_ns), unit="ns")


def _sample_runs(samples, start, end, interval_seconds):
    """Encode consecutive equal samples as [first_index, repeat_count, value]."""
    interval_ns = pd.Timedelta(seconds=interval_seconds).value
    times = [time for time, _ in samples]
    first, count = _sample_schedule(times, start, end, interval_seconds)
    observed = {time.value for time in times}
    missing = [
        (first.value + index * interval_ns, "miss")
        for index in range(count) if first.value + index * interval_ns not in observed
    ]
    runs = []
    previous_time_ns = None
    # Sort by time alone: numeric observations and missing markers have different types.
    observed_pairs = [(time.value, value) for time, value in samples]
    for time_ns, value in sorted([*observed_pairs, *missing], key=lambda sample: sample[0]):
        offset = time_ns - start.value
        step_index = offset // interval_ns if offset % interval_ns == 0 else offset / interval_ns
        if (runs and time_ns == previous_time_ns + interval_ns
                and value == runs[-1][2] and type(value) is type(runs[-1][2])):
            runs[-1][1] += 1
        else:
            runs.append([step_index, 1, value])
        previous_time_ns = time_ns
    return runs


def format_metrics(
    rows: pd.DataFrame,
    node_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    metrics: list[str],
    label_filters: dict[str, str] | None = None,
    sample_interval_seconds: int = SAMPLE_INTERVAL_SECONDS,
    *, source: GwdgDataSource | None = None,
) -> dict:
    """Format raw series with a compact time axis and explicit missing samples."""
    interval_ns = pd.Timedelta(seconds=sample_interval_seconds).value
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
        "time_axis": {
            "start": window["start_time"], "step_seconds": sample_interval_seconds,
            "count": -((start.value - end.value) // interval_ns),
        },
        "sample_fields": ["first_index", "repeat_count", "value"],
        "nodes": {},
    }
    if rows.empty:
        return result

    # Resource inventories are returned only by get_resource_ids_gwdg.
    selected = rows.loc[rows["metric"].isin(metrics)]
    returned_metrics = {}
    # 4. Group observations into separate series, rather than averaging them.
    for values, group in selected.groupby(list(IDENTITY_COLUMNS), sort=True):
        # groupby returns one tuple of identity values plus that series' rows.
        # zip turns the tuple back into a dictionary with the original field names.
        identity = dict(zip(IDENTITY_COLUMNS, values))
        labels = series_labels(identity, source=source)
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
        # 6. Preserve raw readings and duplicates. Encoding adds explicit 'miss'
        # markers without inventing numeric readings or rates.
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
            "sample_runs": _sample_runs(samples, start, end, sample_interval_seconds),
        })
    # 7. Attach the node and its metrics. Keep node_list consistent with nodes' keys.
    result["node_list"] = [node_id]
    result["nodes"][node_id] = {"metrics": returned_metrics}
    return result
