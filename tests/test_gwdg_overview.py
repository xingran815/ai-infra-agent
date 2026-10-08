import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dwdg import telemetry, tools_dwdg as api
from src.dwdg.overview import (
    OVERVIEW_METRICS, _counter_changes, _gauge_statistics,
    _memory_available_percent, _disk_read_latency,
)


NODE = "nodetest"
START = "2025-03-18T09:00:00Z"
END = "2025-03-18T09:00:30Z"


class CalculationTests(unittest.TestCase):
    def test_memory_percentage_and_disk_latency_without_loading_data(self):
        times = [START, "2025-03-18T09:00:10Z", "2025-03-18T09:00:20Z"]
        readings = {
            "node_memory_MemAvailable_bytes": [50, 25, 10],
            "node_memory_MemTotal_bytes": [100, 100, 100],
            "node_disk_read_time_seconds_total": [1, 1.2, 1.6],
            "node_disk_reads_completed_total": [100, 110, 130],
        }
        metrics = {}
        for name, values in readings.items():
            metrics[name] = {"series": [{
                "labels": {}, "points": dict(zip(times, values)),
                "evidence_id": name, "coverage": {"expected_samples": 3, "returned_samples": 3},
            }]}

        memory = _memory_available_percent(metrics)["series"][0]
        latency = _disk_read_latency(metrics, 10)["series"][0]

        self.assertEqual((memory["min"], memory["max"], memory["latest"]), (10, 50, 10))
        self.assertAlmostEqual(memory["mean"], 85 / 3)
        self.assertAlmostEqual(latency["value"], 20)
        self.assertEqual(latency["completed_reads"], 30)

    def test_gauge_statistics(self):
        points = {START: 50, "2025-03-18T09:00:10Z": 10, "2025-03-18T09:00:20Z": 30}

        summary = _gauge_statistics(points)

        self.assertEqual(summary, {
            "latest": 30, "latest_time": "2025-03-18T09:00:20Z",
            "min": 10, "min_time": "2025-03-18T09:00:10Z",
            "max": 50, "max_time": START, "mean": 30,
        })

    def test_counter_changes(self):
        middle = "2025-03-18T09:00:10Z"
        last = "2025-03-18T09:00:20Z"

        changes = _counter_changes({START: 100, middle: 110, last: 130}, 10)

        self.assertEqual(changes, {
            "increases": {(START, middle): 10, (middle, last): 20}, "resets": 0, "gaps": 0,
        })


class OverviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(api.unload_data_gwdg)
        self.root = Path(self.temp.name)
        self.path = self.root / f"{NODE}_incident.csv"
        self.rows, self.labels = [], []
        scheduler = self.root / "scheduler.csv"
        fields = ["node", "timeUtc", "value", *telemetry.IDENTITY_COLUMNS]
        with scheduler.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(node=NODE, timeUtc=START, value=1, metric="slurm_status_ok"))
        api.load_scheduler_data_gwdg(str(scheduler))

    def add(self, metric, values, *, times=(0, 10, 20), **labels):
        sid = f"s{len(self.labels)}" if labels else ""
        if labels:
            self.labels.append((sid, metric, json.dumps(labels)))
        for second, value in zip(times, values):
            self.rows.append(dict(metric=metric, value=value, seriesId=sid,
                                  timeUtc=f"2025-03-18T09:00:{second:02d}Z"))

    def load(self):
        fields = ["timeUtc", "value", *telemetry.IDENTITY_COLUMNS]
        with self.path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        with (self.root / "series_labels.tsv").open("w") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["seriesId", "metric", "labels"])
            writer.writerows(self.labels)
        api.load_data_gwdg(str(self.path))

    def overview(self):
        return api.get_overview_gwdg(NODE, START, END)

    def test_category_selectors_null_empty_subset_and_invalid(self):
        cases = [
            ("node", "node_load1", {}, (NODE,)),
            ("gpu", "DCGM_FI_DEV_GPU_TEMP", {"uuid": "gpu1"}, (NODE, "gpu1")),
            ("network", "node_network_up", {"device": "eth0"}, (NODE, "eth0")),
            ("storage", "node_disk_io_now", {"device": "disk0"}, (NODE, "disk0")),
            ("monitoring", "up", {}, (NODE,)),
            ("scheduler", "slurm_status_ok", {}, (NODE,)),
        ]
        for category, metric, labels, args in cases:
            if category != "scheduler":
                self.add(metric, [1, 2, 3], **labels)
        self.load()
        for category, metric, _, args in cases:
            with self.subTest(category=category):
                query = getattr(api, f"get_{category}_metrics_gwdg")
                default = query(*args, START, END)
                self.assertEqual(default, query(*args, START, END, metrics=None))
                selected = query(*args, START, END, metrics=[metric])
                self.assertEqual(set(selected["nodes"][NODE]["metrics"]), {metric})
                self.assertEqual(query(*args, START, END, metrics=[])["nodes"][NODE]["metrics"], {})
                for metrics in (["not_a_metric"], "NULL", [1]):
                    with self.assertRaises(ValueError):
                        query(*args, START, END, metrics=metrics)

    def test_coverage_uses_each_series_phase_and_retains_raw_axis(self):
        self.add("node_load1", [1, 2, 3], times=(5, 15, 25))
        self.add("node_load5", [1, 3], times=(0, 20))
        self.load()
        result = api.get_node_metrics_gwdg(NODE, START, END, metrics=["node_load1", "node_load5"])
        self.assertEqual(len(result["timestamps"]), 6)
        self.assertEqual(result["sample_fields"], ["timestamp_index", "value"])
        self.assertIn("not zero", result["sample_encoding"])
        metrics = result["nodes"][NODE]["metrics"]
        self.assertEqual(metrics["node_load1"]["series"][0]["coverage"], {"expected_samples": 3, "returned_samples": 3})
        self.assertEqual(metrics["node_load5"]["series"][0]["coverage"], {"expected_samples": 3, "returned_samples": 2})
        unaligned = api.get_node_metrics_gwdg(NODE, "2025-03-18T09:00:06Z", "2025-03-18T09:00:25Z", metrics=["node_load1"])
        self.assertEqual(unaligned["nodes"][NODE]["metrics"]["node_load1"]["series"][0]["coverage"],
                         {"expected_samples": 1, "returned_samples": 1})

    def test_overview_derived_values_and_separate_edac_levels(self):
        self.add("node_memory_MemAvailable_bytes", [50, 25, 10])
        self.add("node_memory_MemTotal_bytes", [100, 100, 100])
        self.add("node_disk_read_time_seconds_total", [1, 1.2, 1.6], device="disk0")
        self.add("node_disk_reads_completed_total", [100, 110, 130], device="disk0")
        self.add("node_edac_correctable_errors_total", [0, 2, 3], controller="0")
        self.add("node_edac_csrow_correctable_errors_total", [0, 2, 3], controller="0", csrow="0")
        self.add("DCGM_FI_DEV_XID_ERRORS", [0, 79, 79], uuid="gpu1")
        self.add("node_network_up", [1, 0, 1], device="eth0")
        self.add("up", [1, 1, 1])
        self.load()
        with patch.object(telemetry, "get_metrics_gwdg", wraps=telemetry.get_metrics_gwdg) as query:
            result = self.overview()
        self.assertEqual(query.call_count, len(OVERVIEW_METRICS))
        self.assertEqual(
            {call.kwargs["category"]: call.kwargs["metrics"] for call in query.call_args_list},
            OVERVIEW_METRICS,
        )
        memory = result["derived"]["memory_available_percent"]["series"][0]
        self.assertEqual((memory["min"], memory["max"], memory["latest"]), (10, 50, 10))
        latency = result["derived"]["average_disk_read_latency_ms"]["series"][0]
        self.assertAlmostEqual(latency["value"], 20)
        self.assertEqual(latency["completed_reads"], 30)
        node = result["categories"]["node"]
        self.assertEqual(node["edac_controller"]["node_edac_correctable_errors_total"]["series"][0]["observed_increase"], 3)
        self.assertEqual(node["edac_csrow"]["node_edac_csrow_correctable_errors_total"]["series"][0]["observed_increase"], 3)
        xid = result["categories"]["gpu"]["metrics"]["DCGM_FI_DEV_XID_ERRORS"]["series"][0]
        self.assertEqual(xid["transition_count"], 1)
        self.assertNotIn("mean", xid)
        self.assertNotIn('"samples":', json.dumps(result, allow_nan=False))

    def test_latency_excludes_resets_and_returns_null_for_idle_disk(self):
        self.add("node_disk_read_time_seconds_total", [100, 1, 1.4], device="reset")
        self.add("node_disk_reads_completed_total", [1000, 10, 30], device="reset")
        self.add("node_disk_read_time_seconds_total", [1, 1, 1], device="idle")
        self.add("node_disk_reads_completed_total", [10, 10, 10], device="idle")
        self.load()
        entries = self.overview()["derived"]["average_disk_read_latency_ms"]["series"]
        by_device = {e["labels"]["device"]: e for e in entries}
        self.assertAlmostEqual(by_device["reset"]["value"], 20)
        self.assertEqual(by_device["reset"]["valid_seconds"], 10)
        self.assertIsNone(by_device["idle"]["value"])

    def test_derived_values_require_matching_labels_times_and_valid_denominators(self):
        self.add("node_memory_MemAvailable_bytes", [50, 25, 10], job="a")
        self.add("node_memory_MemTotal_bytes", [0, 100], times=(0, 20), job="a")
        self.add("node_disk_read_time_seconds_total", [1, 2], times=(0, 20), device="a")
        self.add("node_disk_reads_completed_total", [1, 2, 3], device="a")
        self.add("node_disk_reads_completed_total", [1, 2, 3], device="b")
        self.load()
        result = self.overview()["derived"]
        memory = result["memory_available_percent"]["series"][0]
        self.assertEqual(memory["matched_samples"], 1)
        self.assertEqual(memory["latest"], 10)
        for entry in result["average_disk_read_latency_ms"]["series"]:
            self.assertIsNone(entry["value"])

    def test_bounded_summaries_prioritize_failed_series_and_deduplicate_node_metrics(self):
        for i in range(8):
            self.add("node_network_up", [1, 1, int(i != 7)], device=f"eth{i}")
        self.add("node_netstat_Tcp_RetransSegs", [0, 1, 2])
        self.load()
        result = self.overview()["categories"]["network"]["metrics"]
        links = result["node_network_up"]
        self.assertEqual(links["series_count"], 8)
        self.assertEqual(links["series_omitted"], 5)
        self.assertEqual(links["non_nominal_series"], 1)
        self.assertEqual(links["series"][0]["labels"]["device"], "eth7")
        self.assertEqual(result["node_netstat_Tcp_RetransSegs"]["series_count"], 1)

    def test_missing_scheduler_is_reported_without_hiding_incident_data(self):
        self.add("node_load1", [1, 2, 3])
        self.load()
        with patch.object(telemetry, "read_scheduler_window", side_effect=FileNotFoundError):
            result = self.overview()
        self.assertEqual(result["categories"]["scheduler"]["status"], "source_unavailable")
        self.assertEqual(result["derived"]["memory_available_percent"]["status"], "missing_metrics")
        self.assertIn("node_load1", result["categories"]["node"]["metrics"])

    def test_ambiguous_samples_do_not_produce_derived_measurements(self):
        self.add("node_memory_MemAvailable_bytes", [50, 90, 25], times=(0, 0, 10))
        self.add("node_memory_MemTotal_bytes", [100, 100, 100])
        self.add("node_disk_read_time_seconds_total", [1, 2, 4], device="disk0")
        self.add("node_disk_reads_completed_total", [10, 20, 21, 30], times=(0, 10, 10, 20), device="disk0")
        self.load()
        result = self.overview()
        memory = result["derived"]["memory_available_percent"]["series"][0]
        self.assertEqual(memory["matched_samples"], 1)
        self.assertEqual(memory["latest"], 25)
        latency = result["derived"]["average_disk_read_latency_ms"]["series"][0]
        self.assertIsNone(latency["value"])

    def test_scheduler_coverage_uses_600_second_phase(self):
        import pandas as pd
        rows = []
        for minute in (5, 15, 25):
            row = dict.fromkeys(telemetry.IDENTITY_COLUMNS, "")
            row.update(metric="slurm_status_ok", value="1",
                       timeUtc=pd.Timestamp(f"2025-03-18T09:{minute:02d}:00Z"))
            rows.append(row)
        result = telemetry.format_metrics(pd.DataFrame(rows), NODE, telemetry.utc_time(START),
                                          telemetry.utc_time("2025-03-18T09:30:00Z"),
                                          ["slurm_status_ok"], sample_interval_seconds=600)
        self.assertEqual(len(result["timestamps"]), 6)
        coverage = result["nodes"][NODE]["metrics"]["slurm_status_ok"]["series"][0]["coverage"]
        self.assertEqual(coverage, {"expected_samples": 3, "returned_samples": 3})


if __name__ == "__main__":
    unittest.main()
