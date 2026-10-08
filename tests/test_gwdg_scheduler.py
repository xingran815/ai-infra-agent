import csv
import lzma
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dwdg import telemetry
from src.dwdg import tools_dwdg as gwdg


NODE = "node6a5ebc855306"
OTHER_NODE = "node000000000000"
START = "2025-03-18T09:00:00Z"
END = "2025-03-18T09:30:00Z"
FIELDS = ["node", "timeUtc", "utcOffsetHours", "value", *telemetry.IDENTITY_COLUMNS]


class SchedulerToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(gwdg.unload_data_gwdg)
        self.root = Path(self.directory.name)
        self.path = self.root / "slurm_node_state_tidy.csv.xz"
        self.rows = []
        for node_id in (NODE, OTHER_NODE):
            for metric, value in (("slurm_total_gpus", "4"), ("slurm_alloc_gpus", "2"), ("slurm_status_ok", "1")):
                for time in ("08:50:00", "09:00:00", "09:10:00", "09:20:00", "09:30:00"):
                    if node_id == NODE and metric == "slurm_alloc_gpus" and time == "09:10:00":
                        continue
                    row = dict.fromkeys(FIELDS, "")
                    row.update(node=node_id, timeUtc=f"2025-03-18 {time}", utcOffsetHours="+1",
                               metric=metric, value=value if node_id == NODE else "99", job="slurm")
                    self.rows.append(row)
        self.write_scheduler()

    def write_scheduler(self, path=None):
        path = path or self.path
        with lzma.open(path, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def query(self, **kwargs):
        if kwargs:
            return gwdg.get_metrics_gwdg(NODE, START, END, category="scheduler", **kwargs)
        return gwdg.get_scheduler_metrics_gwdg(NODE, START, END)

    def test_scheduler_only_discovery_lists_all_nodes_in_one_pass(self):
        gwdg.load_scheduler_data_gwdg(str(self.path), chunksize=7)
        with patch.object(telemetry.pd, "read_csv", wraps=telemetry.pd.read_csv) as read_csv:
            inventory = gwdg.get_resource_ids_gwdg(START, END)
            self.assertEqual(read_csv.call_count, 1)
        self.assertEqual(inventory["node_list"], sorted([NODE, OTHER_NODE]))
        self.assertEqual(inventory["resource_ids"], sorted([NODE, OTHER_NODE]))
        for node_id in inventory["node_list"]:
            self.assertEqual(inventory["nodes"][node_id]["resources"]["node_id"], node_id)
            self.assertEqual(inventory["nodes"][node_id]["resources"]["gpu_ids"], [])
            result = gwdg.get_scheduler_metrics_gwdg(node_id, START, END)
            self.assertEqual(result, gwdg.get_metrics_gwdg(node_id, START, END, category="scheduler"))
        empty = gwdg.get_resource_ids_gwdg("2025-03-18T12:00:00Z", "2025-03-18T12:20:00Z")
        self.assertEqual(empty["node_list"], [])
        self.assertEqual(empty["resource_ids"], [])

    def test_scheduler_filters_node_and_utc_window_with_native_coverage(self):
        gwdg.load_scheduler_data_gwdg(str(self.path), chunksize=7)
        result = self.query()
        self.assertEqual(result["node_list"], [NODE])
        self.assertEqual(result["window"]["sample_interval_seconds"], 600)
        self.assertEqual(result["timestamps"], [START, "2025-03-18T09:10:00Z", "2025-03-18T09:20:00Z"])
        metric = result["nodes"][NODE]["metrics"]["slurm_alloc_gpus"]
        series = metric["series"][0]
        self.assertEqual(series["samples"], [[0, 2], [2, 2]])
        self.assertEqual(series["coverage"], {"expected_samples": 3, "returned_samples": 2})
        self.assertIsNone(series["series_id"])
        self.assertEqual(series["labels"], {"job": "slurm"})
        self.assertTrue(series["evidence_id"].startswith("series-SLALLOC-"))
        self.assertEqual(result["nodes"][NODE]["resources"]["gpu_ids"], [])
        self.assertNotIn("cpus", result["nodes"][NODE]["resources"])

    def test_default_dataset_path_works_without_loading_an_incident(self):
        with patch.object(telemetry, "DATA_DIR", self.root):
            self.assertEqual(self.query()["window"]["sample_interval_seconds"], 600)

    def test_default_scheduler_path_is_beside_selected_incident(self):
        incident = self.root / f"{NODE}_test_tidy.csv"
        with incident.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS[1:])
            writer.writeheader()
            row = dict.fromkeys(FIELDS[1:], "")
            row.update(timeUtc="2025-03-18 09:00:00", utcOffsetHours="+1", metric="node_load1", value="1")
            writer.writerow(row)
        (self.root / "series_labels.tsv").write_text("seriesId\tmetric\tlabels\n")
        gwdg.load_data_gwdg(str(incident))
        node_result = gwdg.get_metrics_gwdg(NODE, START, END, category="node", metrics=["node_load1"])
        scheduler_result = self.query()
        with patch.object(telemetry.pd, "read_csv", side_effect=AssertionError("Source was read again")):
            self.assertEqual(node_result, gwdg.get_metrics_gwdg(NODE, START, END, category="node", metrics=["node_load1"]))
            self.assertEqual(scheduler_result, self.query())

    def test_selections_empty_windows_nodes_and_invalid_requests(self):
        gwdg.load_scheduler_data_gwdg(str(self.path))
        selected = self.query(metrics=["slurm_status_ok"])
        self.assertEqual(set(selected["nodes"][NODE]["metrics"]), {"slurm_status_ok"})
        self.assertEqual(self.query(metrics=[])["nodes"][NODE]["metrics"], {})
        self.assertEqual(gwdg.get_scheduler_metrics_gwdg("nodeffffffffffff", START, END)["nodes"], {})
        self.assertEqual(gwdg.get_scheduler_metrics_gwdg(NODE, "2025-03-18T12:00:00Z", "2025-03-18T12:20:00Z")["nodes"], {})
        other = gwdg.get_scheduler_metrics_gwdg(OTHER_NODE, START, END)
        self.assertEqual(other["nodes"][OTHER_NODE]["metrics"]["slurm_total_gpus"]["series"][0]["samples"], [[0, 99], [1, 99], [2, 99]])
        with self.assertRaises(ValueError):
            self.query(metrics=["node_load1"])
        for start, end in ((START, START), (END, START), ("2025-03-18T09:00:00", END)):
            with self.assertRaises(ValueError):
                gwdg.get_scheduler_metrics_gwdg(NODE, start, end)

    def test_scheduler_reuses_normalized_window_and_handles_unaligned_bounds(self):
        gwdg.load_scheduler_data_gwdg(str(self.path))
        original = self.query()
        with patch.object(telemetry.pd, "read_csv", side_effect=AssertionError("Source was read again")):
            equivalent = gwdg.get_scheduler_metrics_gwdg(NODE, "2025-03-18T10:00:00+01:00", "2025-03-18T10:30:00+01:00")
        self.assertEqual(original, equivalent)
        unaligned = gwdg.get_scheduler_metrics_gwdg(NODE, "2025-03-18T09:05:00Z", "2025-03-18T09:21:00Z")
        self.assertEqual(unaligned["timestamps"], ["2025-03-18T09:10:00Z", "2025-03-18T09:20:00Z"])
        series = unaligned["nodes"][NODE]["metrics"]["slurm_alloc_gpus"]["series"][0]
        self.assertEqual(series["samples"], [[1, 2]])
        self.assertEqual(series["coverage"], {"expected_samples": 2, "returned_samples": 1})

    def test_loading_another_scheduler_table_refreshes_cache(self):
        gwdg.load_scheduler_data_gwdg(str(self.path))
        original = self.query()
        for row in self.rows:
            if row["node"] == NODE and row["metric"] == "slurm_status_ok":
                row["value"] = "0"
        other_path = self.root / "alternate.csv.xz"
        self.write_scheduler(other_path)
        gwdg.load_scheduler_data_gwdg(str(other_path))
        changed = self.query()
        before = original["nodes"][NODE]["metrics"]["slurm_status_ok"]
        after = changed["nodes"][NODE]["metrics"]["slurm_status_ok"]
        self.assertEqual(before["evidence_id"], after["evidence_id"])
        self.assertEqual(after["series"][0]["samples"], [[0, 0], [1, 0], [2, 0]])
        gwdg.unload_data_gwdg()
        self.assertEqual(telemetry.read_window.cache_info().currsize, 0)
        self.assertEqual(telemetry.read_scheduler_window.cache_info().currsize, 0)


if __name__ == "__main__":
    unittest.main()
