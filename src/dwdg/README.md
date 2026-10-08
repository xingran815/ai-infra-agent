# GWDG metrics

```python
from pathlib import Path
from src.dwdg.tools_dwdg import (
    load_data_gwdg, get_resource_ids_gwdg, get_node_metrics_gwdg,
    get_metrics_gwdg, get_gpu_metrics_gwdg, get_network_metrics_gwdg,
    get_storage_metrics_gwdg, get_scheduler_metrics_gwdg,
    get_monitoring_metrics_gwdg,
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

node = get_node_metrics_gwdg(resource_id=node_id, **window)
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

# Only the generic query accepts optional selectors.
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

All category tools have `strict=True`, require `resource_id`, and return their
complete category. None accepts a `metrics` argument or any optional arguments.
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
Coverage counts expected axis slots and returned observations. Evidence IDs look
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
use strict category tools for complete categories or the generic query for a few
names and specific label filters.
