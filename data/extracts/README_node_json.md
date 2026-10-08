# Node telemetry JSON export

Full export: `node6a5ebc855306_2025-03-18_090000Z_092000Z_first20min_tidy_node.json` (5.91 MB).
Readable preview: `node6a5ebc855306_2025-03-18_090000Z_092000Z_first20min_tidy_node_preview.json` (12 representative metrics, one series per metric, first three available samples; explicitly marked `is_preview`).

Source: the first 20 minutes of `node6a5ebc855306_2025-03-18_090000Z_092000Z_first20min_tidy.csv`. The source CSV contains one node, `node6a5ebc855306`. This export contains only the 127 metrics assigned to `get_node_metrics_gwdg`, with 503,346 observations in 4,195 distinct series. Other categories' readings remain in the CSV.

Window: 2025-03-18T09:00:00Z inclusive to 2025-03-18T09:20:00Z exclusive, sampled every 10 seconds. All timestamps use UTC. `window.utc_offset_hours` is the observed constant Europe/Berlin offset (+1) for this window.

## Structure

- `source`: dataset name, extracted CSV path, and label-dictionary path.
- `window`: explicit time bounds, exclusive end, native cadence, and local UTC offset.
- `timestamps`: the shared axis containing all 120 UTC timestamps.
- `sampling.sample_fields`: `["timestamp_index", "value"]`; each series' `samples` uses this layout.
- `nodes`: mapping keyed by actual node pseudonym; no separate duplicate node list is needed.
- `nodes[node].resources`: observed GPU UUID/index/model information, OS network interfaces, InfiniBand adapters, disk devices, and filesystem device labels.
- `nodes[node].metrics[metric_name]`: unit, producer, optional baseline (null), and distinct `series`.
- Each series: dataset `series_id`, a node-specific `series_key`, distinguishing `labels`, sample count, missing samples in the source, and raw samples.

For example, `samples: [[0, 68001.91], [1, 68001.91]]` means values at `timestamps[0]` and `timestamps[1]`. Missing indices are absent observations, not zero values. No filling or downsampling was applied. Counters remain counters; no rates or counter averages were invented.

There are 54 missing observations in one node series against the 120-slot time grid. `coverage` describes the full export. In the preview, `source_coverage` describes the full source export and `preview_counts` describes the included subset; truncated series also include `available_sample_count` and `samples_truncated`.

Units are inferred only from explicit metric-name suffixes; these have `unit_source: "metric_name"`. Unspecified units are null. Baselines are null because the CSV provides none; estimating a baseline requires a separately defined reference period and method. Original metric names and numeric values are retained, although numeric text formatting may differ from the CSV.

GPU IDs are the original pseudonymized UUIDs. Interface/disk/adapter names are node-local; combine them with the node pseudonym for cross-node identity. Multiple filesystem series can share a device label because mountpoints are sanitized; retain the series identifiers. Published labels from `series_labels.tsv` are combined with nonempty labels from the CSV. A dataset `series_id` can be null; `series_key` always distinguishes the node, metric, and full promoted-label identity.

Deterministic evidence IDs for tool responses can be reconstructed as:

```python
import hashlib

evidence_id = node_id + "-obs-" + hashlib.sha256(
    (series_key + "|" + timestamp).encode()
).hexdigest()[:20]
```

`series_key` is the first 20 hex characters of SHA-256 over sorted compact JSON containing `node_id` and the original CSV identity fields (`metric`, `gpu`, `device`, `uuid`, `job`, `modelName`, `driverVersion`, `seriesId`).

## Reading the export

```python
import json
from pathlib import Path

data = json.loads(Path("data/extracts/node6a5ebc855306_2025-03-18_090000Z_092000Z_first20min_tidy_node.json").read_text())
node = data["nodes"]["node6a5ebc855306"]
series = node["metrics"]["node_load1"]["series"][0]
readings = [(data["timestamps"][i], value) for i, value in series["samples"]]
```

Store the complete data locally. Query tools should return selected series/time ranges or explicit summaries with evidence IDs and coverage, rather than passing all 503,346 readings to the model. This export describes observed telemetry; it does not supply diagnoses or incident labels.
