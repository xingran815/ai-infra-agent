# Node telemetry JSON export

This describes the current GWDG tool response format. Regenerate local JSON
extracts with the example below to use this format.

The example selects node `node6a5ebc855306` from 2025-03-18T09:00:00Z inclusive
to 2025-03-18T09:20:00Z exclusive. Times are UTC and incident cadence is 10 seconds.

## Structure

- `source`: dataset name.
- `window`: normalized UTC bounds, exclusive end, and native sample interval.
- `time_axis`: `start`, `step_seconds`, and `count`; no timestamp array is needed.
- `sample_fields`: `["first_index", "repeat_count", "value"]`.
- `nodes[node].metrics[metric_name]`: metric evidence ID, metadata, and distinct series.
- Each series: evidence ID, original `series_id`, complete labels, coverage, and `sample_runs`.

Resource inventories are returned only by `get_resource_ids_gwdg`. Metric responses
retain identifying series labels without repeating a `resources` object.

Each `sample_runs` entry is `[first_index, repeat_count, value]`. For example,
`[[0, 2, 79], [2, 2, "miss"], [4, 2, 79]]` records 79 at indices 0 and 1,
missing observations at 2 and 3, and 79 at 4 and 5. For offsets `i` from 0 through
`repeat_count - 1`, calculate timestamps as `start + step_seconds * (first_index + i)`.
The string `"miss"` is excluded from summaries and never treated as zero. Only equal
values at consecutive indices are grouped; duplicates retain overlapping runs.
The timestamp formula and missing-observation rules are provided in tool descriptions.

The axis starts at the requested UTC start. `count` counts nominal cadence steps
inside the exclusive-end window, rounding up a partial step. Fractional indices
preserve off-grid observations and unaligned query bounds. Duplicate observations
retain repeated indices. Missing markers use each series' modal observed sampling
phase; this estimates a fixed schedule and does not infer resource lifetime.
`coverage.returned_samples` counts observed rows, including duplicates, and excludes
`"miss"` entries. Entirely unobserved series are absent.

Labels combine the original CSV identity with `series_labels.tsv`. Device IDs are
node-local except GPU UUIDs; retain the node and complete series identity when
joining observations. Evidence IDs identify the dataset, node, metric, and normalized
window; series IDs also include the complete identity and resolved labels. Consult
`src/dwdg/README.md` for metadata and evidence-scope details.

## Exporting and reading

```python
import json
from datetime import datetime, timedelta
from pathlib import Path
from src.dwdg.tools_dwdg import load_data_gwdg, get_node_metrics_gwdg

root = Path("data/extracts")
stem = "node6a5ebc855306_2025-03-18_090000Z_092000Z_first20min_tidy"
load_data_gwdg(str((root / f"{stem}.csv").resolve()))
data = get_node_metrics_gwdg(
    resource_id="node6a5ebc855306",
    start_time="2025-03-18T09:00:00Z",
    end_time="2025-03-18T09:20:00Z",
)
(root / f"{stem}_node.json").write_text(json.dumps(data, separators=(",", ":")))

axis = data["time_axis"]
start = datetime.fromisoformat(axis["start"].replace("Z", "+00:00"))
series = data["nodes"]["node6a5ebc855306"]["metrics"]["node_load1"]["series"][0]
readings = [
    (start + timedelta(seconds=axis["step_seconds"] * (first_index + offset)), value)
    for first_index, repeat_count, value in series["sample_runs"]
    if value != "miss"
    for offset in range(repeat_count)
]
```

Store complete data locally and retrieve selected metrics and windows for the agent.
These responses contain observations and coverage, not automatic diagnoses.
