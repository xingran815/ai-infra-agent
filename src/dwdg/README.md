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
Loading another file replaces the default source and starts with empty caches.
Scheduler queries automatically use `slurm_node_state_tidy.csv.xz` beside the
selected incident file, falling back to `DATA_DIR`. They also work without loading
an incident. Use `load_scheduler_data_gwdg(path, chunksize=...)` from
`tools_dwdg.py` to select a different scheduler CSV or CSV.xz; this leaves the
incident path selected, creating a new default source. `unload_data_gwdg()` drops
the default source. Explicitly retained source objects remain usable.

For an investigation, create a `GwdgDataSource` and pass it to each tool:

```python
from src.dwdg.tools_dwdg import GwdgDataSource, call_function_dwdg

source = GwdgDataSource(
    "node6a5ebc855306_2025-03-19_gpu-problem_tidy.csv.xz",
    series_labels_path="series_labels.tsv",
    # scheduler_path="slurm_node_state_tidy.csv.xz",  # Optional override.
)
try:
    inventory = call_function_dwdg(
        "get_resource_ids_gwdg", window, "discovery-1", source=source,
    )
    node_id = inventory["node_list"][0]
    overview = get_overview_gwdg(node_id=node_id, **window, source=source)
finally:
    source.clear_cache()
```

Each object owns its paths, resolved labels, and one cached window per source kind
(incident and scheduler). The agent runner constructs one object per GWDG run and
clears its caches on success or failure. `source` is an application-only keyword
argument; model tool schemas do not expose it. All queries within the run use the
same object, regardless of subsequent calls to the default loaders.

Cache keys include `dataset_id`, `revision`, source kind, node, and normalized UTC
window bounds. `revision` fingerprints the resolved paths, file sizes and
nanosecond modification times of the incident, label dictionary, and scheduler
files at construction. This is a lightweight test-fixture identifier, not a
content checksum or a snapshot of file contents. Keep files and source configuration
unchanged during a run; construct a new object after editing a fixture. No file
watcher, shared cache, or full-file hashing is required. Cache revisions do not
change the existing evidence IDs, which still identify observation scopes.

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

Model calls through `call_function_dwdg` have output budgets: 24 KiB of compact
UTF-8 JSON for raw queries and discovery, 48 KiB for the overview. Raw queries
also allow at most 64 series and 1,000 encoded sample runs. These are independent
limits; there is no fixed metric-count cap. The runner uses `serialize_tool_result`
from `response_limits.py` so the measured encoding matches the model input.

An oversized result is replaced by `status: "query_too_broad"`, with
`data_returned: false`, actual size/counts, limits exceeded, and an `available`
catalog of matched metric names and exact label values. Monitoring includes
collectors and jobs; other queries can include UUIDs, devices, CPUs, modes, and
sensors. Each metric can include an `example_retry` using `get_metrics_gwdg`,
which supports `label_filters`. Examples preserve the window and original
resource/filter constraints and select an observed label combination. They are
not ranked findings and may require further narrowing. No raw samples or
evidence IDs from the withheld result reach the agent's evidence trace.

Hint responses fit within 16 KiB, with at most 32 metric entries, 12 labels per
metric, and 64 values per label; byte limits can reduce these further. All
omissions have counts, and strings are omitted whole rather than cut into invalid
selectors. Catalogs reflect the matched query, not the entire dataset. Overview
fallback catalogs cover only its returned detail series. The agent chooses
metrics/labels to retry, or explicitly requests a smaller window when one series
is still too large. The tool does not silently shorten windows or return the
first few series as if they were complete.

This is a response-size guard after query execution. Direct Python query calls
remain complete so overview calculations can inspect every input series. CSV
scans, intermediate memory, total investigation cost, and maximum window duration
are not bounded by this guard.

The agent runner uses `ToolExecutor` from `execution.py` around each dispatched
call, including JSON parsing and output serialization. A failed call returns a
JSON `function_call_output` with the original `call_id`:

```json
{
  "status": "error",
  "data_returned": false,
  "error": {
    "code": "unsupported_metrics",
    "message": "Unsupported gpu metrics: ['invented_metric']",
    "retryable": true,
    "action": "correct_arguments",
    "details": {"available_metrics": ["DCGM_FI_DEV_GPU_UTIL"], "omitted_metrics": 0}
  }
}
```

The example list is illustrative; actual unsupported-metric errors list up to
64 supported names in that category, with an omitted count. Error codes guide
the model's next request:

| Code | Recovery |
| --- | --- |
| `invalid_json`, `invalid_arguments`, `unknown_tool`, `unsupported_metrics` | Correct the request using the schema or returned choices. |
| `temporarily_unavailable` | Retry a timeout or connection failure within the remaining budget. |
| `data_source_unavailable`, `invalid_source_data` | Use other evidence and disclose the unavailable data; operator attention is needed. |
| `internal_tool_error` | Use another tool or report the gap; inspect the server logs. |
| `retry_limit_exceeded` | Change the request or use other evidence. |
| `tool_budget_exhausted` | Return the retained investigation results without executing more tools. |

Retry decisions belong to the model; the executor does not automatically rerun
failed tools. Two consecutive failures of the same tool and JSON arguments block
that request for this investigation. JSON whitespace and object-key ordering do
not create a new request. Errors include `failed_attempts` and
`remaining_identical_attempts`; exhausting retries changes `retryable` to false
and directs the model to change its request. Non-retryable failures block an identical request
immediately. A corrected request has its own budget, while every attempt,
including malformed or blocked requests, consumes the investigation's tool
budget. Successful requests reset their failure count. Failures do not prevent
other calls in the same model response from running.

`diagnose_cluster_with_fixture` defaults to five model requests and 80 tool
attempts; Python callers can set `max_model_runs` and `max_tool_calls`. It returns
`status: "completed"` for a validated report, or `status: "incomplete"` plus
`stopping_reason` when budgets, fixture/client failures, or other ordinary
failures prevent completion. Both include `tool_trace` (successful outputs and
structured errors), `collected_evidence_ids`, `model_runs`, `tool_attempts`, and
`investigation_errors`. A fallback report uses `suspected_component: "unknown"`
and does not convert retained observations into a causal claim. An invalid final
report is sent back for correction within the remaining model budget.

Error and oversized responses never add IDs to the citation ledger. Unexpected
exceptions are logged locally without exposing exception text or filesystem
paths to the model. Direct Python tool calls still raise typed exceptions:
`ToolArgumentError` for request validation and `DataValidationError` for explicitly
validated source corruption (both subclass `ValueError`). Signature binding is
checked before tool execution so a tool's own `TypeError` is an internal failure,
not incorrectly presented as a request error.

The runner no longer exits the calling process for investigation failures.
Model-client construction is lazy; missing credentials become an incomplete
result. Source caches and runner-owned clients are cleaned up on completion or
failure. User cancellation and process-control exceptions are allowed to
propagate; this is not a guarantee against process termination or resource
exhaustion. The standalone CLI still exits for missing command-line arguments.

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
Every raw response includes a compact `time_axis` with `start`, `step_seconds`,
and `count`, plus `sample_fields: ["first_index", "repeat_count", "value"]`.
Each series contains `sample_runs`: `[first_index, repeat_count, value]` triples.
Each run expands to the same value at consecutive indices from `first_index`
through `first_index + repeat_count - 1`. Only consecutive equal observations
are grouped; a missing interval or duplicate index starts a separate run.
Encoding explanations are included in the raw tools' descriptions instead of
being repeated in each response.

```json
{
  "time_axis": {
    "start": "2025-03-19T07:30:00Z",
    "step_seconds": 10,
    "count": 900
  },
  "sample_runs": [[297, 185, 79], [482, 260, "miss"], [742, 158, 79]]
}
```

For each run offset `i` from 0 through `repeat_count - 1`, calculate its timestamp
as `time_axis.start + step_seconds * (first_index + i)`.
In this example, 79 was observed at all 185 steps from index 297 through 481,
the next 260 steps have no observation, and 79 resumes at index 742 for 158 steps.
`"miss"` never means zero and is excluded from summary
calculations. The axis starts at the requested UTC start; `count` is the number of
nominal cadence steps in the exclusive-end window, rounded up for partial steps.
Incident telemetry uses a 10-second step and scheduler telemetry a 600-second step.
Fractional first indices preserve off-grid observations and unaligned query bounds:
first index 0.5 on a 10-second axis is five seconds after its start. Duplicate readings
retain overlapping runs. Missing runs follow each series' inferred native
schedule; other series do not change its expected slots.

`expected_samples` counts native-cadence slots inside the requested window using
the series' modal observed phase (earliest phase breaks ties). This estimates a
fixed sampling schedule, including leading and trailing gaps; it does not infer
resource lifetime or a changing schedule. `returned_samples` counts observed rows,
including duplicates, excludes `"miss"` entries, and is therefore not by itself
a completeness percentage. Entirely unobserved series are not invented.
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

Only `get_resource_ids_gwdg` returns the resource inventory. Metric responses contain
`nodes[node_id].metrics` without `resources`; their complete series labels retain
the GPU UUID, device, CPU, job, and other distinguishing identities.
Discovery inventory covers all observations in the window:
`gpu_ids` match `uuid`, while network, InfiniBand, and storage IDs match `device`.
CPU IDs are observed logical CPU indices. Empty inventory lists mean unobserved
resources, and windows without observations return empty `nodes` and `node_list`.

`telemetry.py` contains the shared loader, `read_window`, label lookup, resource
discovery, shared `format_metrics` formatter, and `get_metrics_gwdg` query.
For each data-source object, the most recent incident window stays in memory
with **all metric categories**, so follow-up queries over that window avoid
rereading the CSV. GPU, network, storage, and monitoring wrappers reuse that data.
The most recent scheduler window is cached independently so switching to Slurm
does not evict the incident window. Each source uses its own native cadence.
The cached DataFrame is read-only by convention; construct a new source if a file
changes. Calls without `source=` use the process-global default for compatibility
with sequential scripts. Independent investigations should pass explicit objects.

For model calls, use `GWDG_TOOLS` from `tools_dwdg.py` and dispatch
the named function with `**arguments`. Discovery is registered first, and tool
descriptions instruct the agent to call it before metric retrieval. After discovery,
use `get_overview_gwdg` for a compact first view, then category tools with a metric
list for details. Use the generic query for specific label filters.

## Compact overview

`get_overview_gwdg(node_id, start_time, end_time)` requires a loaded incident and
queries each incident category once through `telemetry.get_metrics_gwdg`, using the curated
`OVERVIEW_METRICS` lists in `overview.py`, plus the four inputs in `DERIVED_METRICS`
needed for memory percentage and disk latency. Each query includes all devices, so no
per-device merging or deduplication is needed. Samples are decoded once into
timestamp/value maps for summaries and the separate memory-percentage and disk
read-latency calculations. The scheduler source is queried at its own cadence;
a missing scheduler file is reported as `source_unavailable`. Invalid data and
other retrieval errors are not silently converted into healthy results.
Dependencies flow from `tools_dwdg` to `overview` to `telemetry`; the overview
does not import the tool entry points.

The overview checks 28 critical metrics across six categories:

| Category | Metrics | Health signals |
| --- | ---: | --- |
| Node | 5 | Load, blocked tasks, OOM kills, uncorrectable memory errors, system state |
| GPU | 5 | XID errors, remap failure, PCIe replays, temperature, utilization |
| Network | 8 | Interface/InfiniBand state, receive/transmit errors and drops, link-down and symbol errors |
| Storage | 5 | Filesystem available/total capacity, readonly/device errors, NFS retries |
| Monitoring | 3 | Target availability, collector success, textfile errors |
| Scheduler | 2 | Node health and GPU allocations |

Two additional derived signals report memory available percentage and disk read
latency. Their four input metrics are fetched internally without duplicating raw
summaries or listing those inputs as missing overview metrics; each derived result
reports its own availability and coverage. CPU-mode details, secondary temperatures,
power draw, traffic volumes, inode counts, and other metrics remain available through
the raw category tools. Exclusion from the overview means not checked, not healthy.

The overview returns these categories without raw samples or a timestamp axis.
Each metric has units, type,
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
or extrapolation is performed. Raw retrieval preserves every observed row and
marks missing scheduled samples with `"miss"`.
Evidence IDs and complete labels on returned summaries support follow-up queries.

`derived.memory_available_percent` calculates `100 * MemAvailable / MemTotal`
only at matching timestamps with positive totals and valid available values.
`derived.average_disk_read_latency_ms` calculates
`1000 * sum(delta(read_time_seconds)) / sum(delta(reads_completed))` on shared,
consecutive intervals without resets. A disk with no completed reads or no valid
paired intervals returns `null`, not zero latency. Both calculations require an
unambiguous match of complete label sets and report their source coverage.

The overview includes controller-level uncorrectable EDAC errors in node metrics.
Correctable errors and chip-select-row breakdowns require a raw follow-up query;
controller totals and row breakdowns should not be added together. Unverified NVLink
bandwidth and chip-temperature definitions are not used in overview calculations.
