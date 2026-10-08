# GWDG metrics

```python
from pathlib import Path
from src.dwdg.tools_dwdg import (
    load_data_gwdg, get_resource_ids_gwdg, get_node_metrics_gwdg,
    get_metrics_gwdg, get_gpu_metrics_gwdg, get_network_metrics_gwdg,
    get_storage_metrics_gwdg, get_scheduler_metrics_gwdg,
    get_monitoring_metrics_gwdg, get_overview_gwdg,
)

load_data_gwdg(str(Path(
    "data/extracts/node6a5ebc855306_2025-03-18_090000Z_092000Z_first20min_tidy.csv"
).resolve()))
window = {
    "start_time": "2025-03-18T09:00:00Z",
    "end_time": "2025-03-18T09:00:20Z",
}

# Discover IDs before retrieving metrics. Discovery needs no known node ID.
inventory = get_resource_ids_gwdg(**window)
node_id = inventory["node_list"][0]
resources = inventory["nodes"][node_id]["resources"]

overview = get_overview_gwdg(node_id=node_id, **window)
node = get_node_metrics_gwdg(resource_id=node_id, metrics=["node_load1"], **window)
gpu = get_gpu_metrics_gwdg(
    node_id=node_id, resource_id=resources["gpu_ids"][0], **window,
)
network = get_network_metrics_gwdg(
    node_id=node_id, resource_id=resources["network_interface_ids"][0], **window,
)
storage = get_storage_metrics_gwdg(
    node_id=node_id, resource_id=resources["storage_ids"][0], **window,
)
monitoring = get_monitoring_metrics_gwdg(resource_id=node_id, **window)
scheduler = get_scheduler_metrics_gwdg(resource_id=node_id, **window)

# The generic query additionally supports exact label filters.
cpu = get_metrics_gwdg(
    node_id=node_id, category="node", metrics=["node_cpu_seconds_total"],
    label_filters={"cpu": "0", "mode": "idle"}, **window,
)
gpu_temperature = get_metrics_gwdg(
    node_id=node_id, category="gpu", resource_id=resources["gpu_ids"][0],
    metrics=["DCGM_FI_DEV_GPU_TEMP"], **window,
)
```

Load one CSV or CSV.xz before querying. Relative paths use `DATA_DIR`.
`series_labels.tsv` is read beside the file, falling back to the dataset directory
for extracts. Optional `series_labels_path` and `chunksize` configure loading.
Loading another file or calling `unload_data_gwdg()` clears the cached window.
Scheduler queries automatically use `slurm_node_state_tidy.csv.xz` beside the
selected incident file, falling back to `DATA_DIR`. They also work without loading
an incident. Use `load_scheduler_data_gwdg(path, chunksize=...)` from
`tools_dwdg.py` to select a different scheduler CSV or CSV.xz; this leaves the
incident source selected. `unload_data_gwdg()` clears both sources and their caches.

Queries include the start and exclude the end; input times must include a timezone.
Call `get_resource_ids_gwdg(start_time, end_time)` first. It returns a flat
`resource_ids` list including node IDs, a `node_list`, and per-node `resources`.
With an incident selected, it discovers that node's resources in the window and
warms the shared window cache. Without an incident, it discovers every node
observed in the selected or default scheduler table. Empty windows return empty
lists and nodes. Logical CPU IDs are listed in `resources.cpus.logical_cpu_ids`.

All six category functions accept `metrics: list[str] | None = None`.
`None` (JSON `null`) selects the complete category, `[]` selects no metrics,
and a list selects those metric names. Unknown or out-of-category names raise
`ValueError`. The string `"NULL"` is not a null value.
Category tool schemas retain `strict=True`: model calls must supply `metrics`
explicitly, using `null` for all metrics. Python callers can omit it.
Node, scheduler, and monitoring tools use the node ID as `resource_id`. GPU,
network, and storage tools also require the owning `node_id`.
GPU IDs match `uuid`; network, InfiniBand, disk, filesystem, and NVMe IDs match
exact `device` labels. Network queries retain protocol, connection-tracking,
socket, and per-CPU softnet series. Storage queries retain node-wide NFS counters;
a disk name such as `nvme0n1` does not implicitly match `/dev/nvme0n1p6`.
Monitoring preserves job and collector series. Scheduler queries share the generic
query implementation and use their separate source's native 600-second cadence.

`get_metrics_gwdg` is the only model tool with optional arguments (`strict=False`).
Its `category` enum is `all | node | gpu | network | storage | scheduler | monitoring | monitor`,
where `monitor` aliases `monitoring`. The default `all` reads every metric in the
incident source, including unregistered names. Select `scheduler` for the separate
Slurm source. Optional `metrics` selects names within the chosen category:
`None` or omission selects the full category, `[]` selects none, and names outside
an explicit category raise `ValueError`. Optional `resource_id` matches the same
IDs as the category wrappers; omission selects all resources. For `all`, a node
resource ID selects the entire node, while a device ID filters device series and
retains node-wide metrics. Optional `label_filters` requires every supplied label
to match exactly. Original `series_id` values remain in the output.
Samples remain raw `[timestamp_index, value]` pairs. The timestamp axis
includes missing 10-second slots and preserves any off-grid observations.
Every raw response includes `sample_fields: ["timestamp_index", "value"]` and
`sample_encoding`: indices are zero-based offsets into `timestamps`, and absent
indices mean no observation for that series, never zero. Other series can add
timestamps to the shared axis without changing this series' expected coverage.
`expected_samples` counts native-cadence slots inside the requested window using
the series' modal observed phase (earliest phase breaks ties). This estimates a
fixed sampling schedule, including leading and trailing gaps; it does not infer
resource lifetime or a changing schedule. `returned_samples` counts raw rows,
including duplicates, and is therefore not by itself a completeness percentage.
Evidence IDs look
like `metric-CPUTIME-<hash>` and `series-GPUTEMP-<hash>`: the uppercase mnemonic
identifies the metric, and the hash identifies its evidence scope. Hash inputs
include the dataset, node, metric, and normalized window; series evidence also
includes the complete original identity and labels. Selection and row order do
not affect IDs. The hash cannot be decoded; look up an ID in its returned result
to recover the node, window, and labels. Existing IDs now have the mnemonic added
to their prefix; the scope hash is unchanged.

`metric_metadata.py` holds all 330 agreed metrics: node (127), GPU (19), network
(138), storage (31), scheduler (3), and monitoring (12). Each has a unique
`short_name` of one to seven uppercase letters, plus `type`, `unit`, and a compact
`note`. Counter notes
refer to changes or rates. Unverified units are null; health direction is included
only for clear cases. NVLink bandwidth remains unverified because exporter versions
differ in its definition. Metrics outside the registry return `short_name: "UNKNOWN"`,
`type: "unknown"`, and `unit: null`; their values and labels are preserved.
Use the scheduler wrapper or `category="scheduler"` for the separate Slurm table.

Resource inventory covers all observations in the window, before selection:
`gpu_ids` match `uuid`, while network, InfiniBand, and storage IDs match `device`.
CPU IDs are observed logical CPU indices. Empty inventory lists mean unobserved
resources, and windows without observations return empty `nodes` and `node_list`.

`telemetry.py` contains the shared loader, `read_window`, label lookup, resource
discovery, shared `format_metrics` formatter, and `get_metrics_gwdg` query.
The most recent incident window stays in memory
with **all metric categories**, so follow-up queries over that window avoid
rereading the CSV. GPU, network, storage, and monitoring wrappers reuse that data.
The most recent scheduler window is cached independently so switching to Slurm
does not evict the incident window. Each source uses its own native cadence.
The cached DataFrame is read-only by convention; load the source again if the file
changes. Source selection is process-global.

For model calls, use `GWDG_TOOLS` from `tools_dwdg.py` and dispatch
the named function with `**arguments`. Discovery is registered first, and tool
descriptions instruct the agent to call it before metric retrieval. After discovery,
use `get_overview_gwdg` for a compact first view, then category tools with a metric
list for details. Use the generic query for specific label filters.

## Compact overview

`get_overview_gwdg(node_id, start_time, end_time)` requires a loaded incident and
queries each incident category once through `telemetry.get_metrics_gwdg`, using the curated
`OVERVIEW_METRICS` lists in `overview.py`. Each query includes all devices, so no
per-device merging or deduplication is needed. Samples are decoded once into
timestamp/value maps for summaries and the separate memory-percentage and disk
read-latency calculations. The scheduler source is queried at its own cadence;
a missing scheduler file is reported as `source_unavailable`. Invalid data and
other retrieval errors are not silently converted into healthy results.
Dependencies flow from `tools_dwdg` to `overview` to `telemetry`; the overview
does not import the tool entry points.

The overview returns node, GPU, network, storage, monitoring, and scheduler
categories without raw samples or a timestamp axis. Each metric has units, type,
series count, at most three series summaries, and an explicit omitted count.
Non-nominal state series are prioritized, then counter increases and gauge peaks.
State transitions and observed code lists are also capped at three with omitted
counts. Missing metrics are listed explicitly. These are observations, not an
automatic diagnosis; for example, an unused interface can legitimately be down.

Gauges include latest, minimum, maximum, sample mean, and extrema timestamps.
State/code metrics retain transitions rather than averages. Counter summaries
report observed increases and rates only over consecutive native-cadence intervals.
Resets, gaps and ambiguous duplicate timestamps are excluded from calculations;
valid duration and exclusion information accompany the result. No interpolation
or extrapolation is performed. Raw retrieval continues to preserve every row.
Evidence IDs and complete labels on returned summaries support follow-up queries.

`derived.memory_available_percent` calculates `100 * MemAvailable / MemTotal`
only at matching timestamps with positive totals and valid available values.
`derived.average_disk_read_latency_ms` calculates
`1000 * sum(delta(read_time_seconds)) / sum(delta(reads_completed))` on shared,
consecutive intervals without resets. A disk with no completed reads or no valid
paired intervals returns `null`, not zero latency. Both calculations require an
unambiguous match of complete label sets and report their source coverage.

`categories.node.edac_controller` and `categories.node.edac_csrow` contain
separate controller totals and chip-select-row breakdowns; they are never added
together. Unverified NVLink bandwidth and chip-temperature definitions are not
used in overview calculations. Detailed identity/configuration metrics remain
available through the raw tools.
