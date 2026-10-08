import csv
import json
import lzma
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dwdg import tools_dwdg as gwdg
from src.dwdg import telemetry
from src.dwdg.metric_metadata import NODE_METRICS, NODE_METRIC_METADATA


NODE = "node6a5ebc855306"
START = "2025-03-18T09:00:00Z"
END = "2025-03-18T09:00:30Z"
FIELDS = ["timeUtc", "utcOffsetHours", "metric", "value", *gwdg.IDENTITY_COLUMNS[1:]]


class GwdgToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(gwdg.unload_data_gwdg)
        self.root = Path(self.directory.name)
        self.path = self.root / f"{NODE}_test_tidy.csv"
        self.dictionary = self.root / "series_labels.tsv"
        self.rows = []
        self.labels = []
        for cpu in range(128):
            sid = f"cpu-{cpu}"
            self.labels.append((sid, "node_cpu_seconds_total", {"cpu": str(cpu), "mode": "idle"}))
            self.add_row("node_cpu_seconds_total", "68001.91", "09:00:00", sid)
            self.add_row("node_cpu_seconds_total", "68011.91", "09:00:20", sid)
        self.labels.extend([
            ("cpu-user", "node_cpu_seconds_total", {"cpu": "0", "mode": "user"}),
            ("controller", "node_edac_correctable_errors_total", {"controller": "0"}),
            ("sensor-1", "node_hwmon_temp_celsius", {"chip": "cpu0", "sensor": "temp1"}),
            ("sensor-2", "node_hwmon_temp_celsius", {"chip": "cpu0", "sensor": "temp2"}),
        ])
        self.add_row("node_cpu_seconds_total", "123.4", "09:00:00", "cpu-user")
        self.add_row("node_edac_correctable_errors_total", "0", "09:00:00", "controller")
        self.add_row("node_hwmon_temp_celsius", "43", "09:00:00", "sensor-1")
        self.add_row("node_hwmon_temp_celsius", "44", "09:00:00", "sensor-2")
        self.add_row("node_load1", "2.5", "08:59:50")
        self.add_row("node_load1", "3.5", "09:00:00")
        self.add_row("node_load1", "4.5", "09:00:20")
        self.add_row("node_load1", "99", "09:00:30")
        self.add_row("node_context_switches_total", "9007199254740993", "09:00:00")
        self.add_row("DCGM_FI_DEV_GPU_TEMP", "30", "09:00:00", uuid="gpu1", device="nvidia0")
        self.add_row("node_network_up", "1", "09:00:00", device="eth1")
        self.add_row("node_infiniband_state_id", "4", "09:00:00", device="mlx5_0")
        self.add_row("node_disk_reads_completed_total", "10", "09:00:00", device="nvme0n1")
        self.add_row("node_network_up", "1", "08:59:50", device="outside-window")
        self.write_fixture()
        gwdg.load_data_gwdg(str(self.path), chunksize=17)

    def add_row(self, metric, value, time, sid="", **labels):
        row = dict.fromkeys(FIELDS, "")
        row.update(timeUtc=f"2025-03-18 {time}", utcOffsetHours="+1", metric=metric,
                   value=value, seriesId=sid, job="prometheus.scrape.default", **labels)
        self.rows.append(row)

    def write_fixture(self, path=None):
        path = path or self.path
        opener = lzma.open if path.suffix == ".xz" else open
        with opener(path, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        with self.dictionary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["seriesId", "metric", "labels"])
            writer.writerows((sid, metric, json.dumps(labels)) for sid, metric, labels in self.labels)

    def query(self, **kwargs):
        return gwdg.get_node_metrics_gwdg(NODE, START, END, **kwargs)

    def query_direct(self, metrics, **kwargs):
        return gwdg.get_metrics_gwdg(NODE, START, END, metrics, **kwargs)

    def metric(self, result, metric):
        return result["nodes"][NODE]["metrics"][metric]

    def test_schema_exclusive_end_and_missing_observations(self):
        result = self.query(metrics=["node_load1"])
        self.assertEqual(result["node_list"], list(result["nodes"]))
        self.assertEqual(result["timestamps"], [START, "2025-03-18T09:00:10Z", "2025-03-18T09:00:20Z"])
        series = self.metric(result, "node_load1")["series"][0]
        self.assertIsNone(series["series_id"])
        self.assertEqual(series["samples"], [[0, 3.5], [2, 4.5]])
        self.assertEqual(series["coverage"], {"expected_samples": 3, "returned_samples": 2})
        self.assertTrue(result["window"]["end_time_exclusive"])
        self.assertEqual(result["source"], gwdg.SOURCE)
        self.assertFalse({"baseline", "producer", "is_preview"} & set(self.metric(result, "node_load1")))
        json.dumps(result, allow_nan=False)

    def test_separate_cpu_controller_and_sensor_series(self):
        result = self.query()
        cpu = self.metric(result, "node_cpu_seconds_total")
        self.assertEqual(len(cpu["series"]), 129)
        self.assertEqual(len({series["evidence_id"] for series in cpu["series"]}), 129)
        self.assertEqual(self.metric(result, "node_edac_correctable_errors_total")["series"][0]["labels"]["controller"], "0")
        sensors = self.metric(result, "node_hwmon_temp_celsius")["series"]
        self.assertEqual({s["labels"]["sensor"] for s in sensors}, {"temp1", "temp2"})

    def test_inventory_uses_window_before_selectors(self):
        result = self.query(metrics=["node_load1"])
        resources = result["nodes"][NODE]["resources"]
        self.assertEqual(resources["cpus"], {"logical_cpu_count": 128, "logical_cpu_ids": list(range(128))})
        self.assertEqual(resources["gpu_ids"], ["gpu1"])
        self.assertEqual(resources["network_interface_ids"], ["eth1"])
        self.assertEqual(resources["infiniband_adapter_ids"], ["mlx5_0"])
        self.assertEqual(resources["storage_ids"], ["nvme0n1"])
        self.assertNotIn("socket_count", resources["cpus"])
        self.assertEqual(set(result["nodes"][NODE]["metrics"]), {"node_load1"})

    def test_direct_query_matches_all_label_filters(self):
        result = self.query_direct(["node_cpu_seconds_total"], label_filters={"cpu": "0", "mode": "idle"})
        series = self.metric(result, "node_cpu_seconds_total")["series"]
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["series_id"], "cpu-0")
        self.assertEqual(series[0]["samples"], [[0, 68001.91], [2, 68011.91]])
        blank = self.query_direct(["node_load1"])
        self.assertIsNone(self.metric(blank, "node_load1")["series"][0]["series_id"])

    def test_ids_are_stable_across_order_chunks_selectors_and_utc_offsets(self):
        original = self.query(metrics=["node_cpu_seconds_total"])
        self.rows.reverse()
        self.write_fixture()
        gwdg.load_data_gwdg(str(self.path), chunksize=500)
        reordered = self.query(metrics=["node_cpu_seconds_total"])
        self.assertEqual(original, reordered)
        equivalent = gwdg.get_metrics_gwdg(NODE, "2025-03-18T10:00:00+01:00",
                                         "2025-03-18T10:00:30+01:00",
                                         metrics=["node_cpu_seconds_total"],
                                         label_filters={"cpu": "0", "mode": "idle"})
        all_metric = self.metric(original, "node_cpu_seconds_total")
        selected_metric = self.metric(equivalent, "node_cpu_seconds_total")
        self.assertEqual(all_metric["evidence_id"], selected_metric["evidence_id"])
        self.assertEqual(all_metric["series"][0], selected_metric["series"][0])
        changed_window = gwdg.get_node_metrics_gwdg(NODE, START, "2025-03-18T09:00:20Z",
                                                  metrics=["node_cpu_seconds_total"])
        self.assertNotEqual(all_metric["evidence_id"], self.metric(changed_window, "node_cpu_seconds_total")["evidence_id"])

    def test_full_promoted_identity_keeps_blank_id_series_separate(self):
        self.add_row("node_load1", "9", "09:00:00", device="another-device")
        self.write_fixture()
        result = self.query(metrics=["node_load1"])
        series = self.metric(result, "node_load1")["series"]
        self.assertEqual(len(series), 2)
        self.assertEqual(len({s["evidence_id"] for s in series}), 2)
        self.assertTrue(all(s["series_id"] is None for s in series))

    def test_ids_include_complete_labels(self):
        original = self.query_direct(["node_cpu_seconds_total"], label_filters={"cpu": "0", "mode": "idle"})
        self.labels[0] = ("cpu-0", "node_cpu_seconds_total", {"cpu": "0", "mode": "steal"})
        self.write_fixture()
        gwdg.load_data_gwdg(str(self.path))
        changed = self.query_direct(["node_cpu_seconds_total"], label_filters={"cpu": "0", "mode": "steal"})
        self.assertNotEqual(self.metric(original, "node_cpu_seconds_total")["series"][0]["evidence_id"],
                            self.metric(changed, "node_cpu_seconds_total")["series"][0]["evidence_id"])

    def test_large_integer_is_not_rounded(self):
        result = self.query(metrics=["node_context_switches_total"])
        self.assertEqual(self.metric(result, "node_context_switches_total")["series"][0]["samples"],
                         [[0, 9007199254740993]])

    def test_duplicate_observations_are_preserved_without_averaging(self):
        self.add_row("node_load1", "8.5", "09:00:00")
        self.write_fixture()
        series = self.metric(self.query(metrics=["node_load1"]), "node_load1")["series"][0]
        self.assertEqual(series["samples"], [[0, 3.5], [0, 8.5], [2, 4.5]])
        self.assertEqual(series["coverage"]["returned_samples"], 3)

    def test_non_finite_sample_is_rejected_without_replacing_it(self):
        self.add_row("node_load1", "NaN", "09:00:10")
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Non-finite sample"):
            self.query(metrics=["node_load1"])

    def test_conflicting_labels_and_invalid_load_preserve_previous_source(self):
        self.labels.append(("cpu-0", "node_cpu_seconds_total", {"cpu": "999", "mode": "idle"}))
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Conflicting series labels"):
            gwdg.load_data_gwdg(str(self.path))
        result = self.query_direct(["node_cpu_seconds_total"], label_filters={"cpu": "0", "mode": "idle"})
        self.assertEqual(self.metric(result, "node_cpu_seconds_total")["series"][0]["labels"]["cpu"], "0")

    def test_xz_and_source_replacement(self):
        compressed = self.root / f"{NODE}_test_tidy.csv.xz"
        self.write_fixture(compressed)
        original = self.query(metrics=["node_load1"])
        gwdg.load_data_gwdg(str(compressed), chunksize=7)
        self.assertEqual(original, self.query(metrics=["node_load1"]))

    def test_empty_selection_unknown_node_and_empty_window(self):
        self.assertEqual(self.query(metrics=[])["nodes"][NODE]["metrics"], {})
        empty_selection = self.query_direct(["node_cpu_seconds_total"], label_filters={"cpu": "999"})
        self.assertEqual(empty_selection["nodes"][NODE]["metrics"], {})
        unknown = gwdg.get_node_metrics_gwdg("node000000000000", START, END)
        self.assertEqual(unknown["node_list"], [])
        self.assertEqual(unknown["nodes"], {})
        empty = gwdg.get_node_metrics_gwdg(NODE, "2025-03-18T11:00:00Z", "2025-03-18T11:00:20Z")
        self.assertEqual(empty["nodes"], {})
        self.assertEqual(empty["node_list"], [])

    def test_unaligned_window_and_off_grid_samples_remain_raw(self):
        self.add_row("node_load1", "7", "09:00:15")
        self.write_fixture()
        result = gwdg.get_node_metrics_gwdg(NODE, "2025-03-18T09:00:05Z",
                                          "2025-03-18T09:00:21Z", metrics=["node_load1"])
        self.assertEqual(result["timestamps"], ["2025-03-18T09:00:10Z", "2025-03-18T09:00:15Z", "2025-03-18T09:00:20Z"])
        self.assertEqual(self.metric(result, "node_load1")["series"][0]["samples"], [[1, 7], [2, 4.5]])

    def test_invalid_inputs_and_unloaded_source(self):
        for selectors in ({"metrics": ["DCGM_FI_DEV_GPU_TEMP"]}, {"metrics": "node_load1"}):
            with self.subTest(selectors=selectors), self.assertRaises(ValueError):
                self.query(**selectors)
        with self.assertRaises(ValueError):
            self.query_direct(["node_cpu_seconds_total"], label_filters={"cpu": 0})
        for start, end in [("2025-03-18T09:00:00", END), (END, START), (START, START), ("NaT", END)]:
            with self.subTest(start=start), self.assertRaises(ValueError):
                gwdg.get_node_metrics_gwdg(NODE, start, end)
        gwdg.unload_data_gwdg()
        with self.assertRaises(RuntimeError):
            self.query()

    def test_missing_dictionary_entry_is_not_silently_merged(self):
        self.labels = self.labels[1:]
        self.write_fixture()
        gwdg.load_data_gwdg(str(self.path))
        with self.assertRaisesRegex(ValueError, "No series_labels.tsv entry"):
            self.query()

    def test_follow_up_queries_reuse_all_categories_in_the_window(self):
        self.query(metrics=["node_load1"])
        with patch.object(telemetry.pd, "read_csv", side_effect=AssertionError("Window was read again")):
            self.query_direct(["node_cpu_seconds_total"], label_filters={"cpu": "0"})
            rows = telemetry.read_window(NODE, telemetry.utc_time(START), telemetry.utc_time(END))
            self.assertIn("DCGM_FI_DEV_GPU_TEMP", set(rows["metric"]))
            self.assertIn("node_network_up", set(rows["metric"]))
            self.assertIn("node_disk_reads_completed_total", set(rows["metric"]))

    def test_direct_queries_support_other_categories_and_registered_metadata(self):
        node_result = self.query(metrics=["node_load1"])
        gpu_id = node_result["nodes"][NODE]["resources"]["gpu_ids"][0]
        with patch.object(telemetry.pd, "read_csv", side_effect=AssertionError("Window was read again")):
            gpu = self.query_direct(["DCGM_FI_DEV_GPU_TEMP"], label_filters={"uuid": gpu_id})
            network = self.query_direct(["node_network_up"], label_filters={"device": "eth1"})
            storage = self.query_direct(["node_disk_reads_completed_total"], label_filters={"device": "nvme0n1"})
        for result, name, value, kind, unit, short_name in (
            (gpu, "DCGM_FI_DEV_GPU_TEMP", 30, "gauge", "celsius", "GPUTEMP"),
            (network, "node_network_up", 1, "gauge", None, "NETUP"),
            (storage, "node_disk_reads_completed_total", 10, "counter", "operations", "DISKR"),
        ):
            metric = self.metric(result, name)
            self.assertEqual(metric["type"], kind)
            self.assertEqual(metric["unit"], unit)
            self.assertEqual(metric["short_name"], short_name)
            self.assertTrue(metric["evidence_id"].startswith(f"metric-{short_name}-"))
            self.assertTrue(metric["series"][0]["evidence_id"].startswith(f"series-{short_name}-"))
            self.assertEqual(metric["series"][0]["samples"], [[0, value]])
        self.assertEqual(self.query_direct(["unobserved_metric"])["nodes"][NODE]["metrics"], {})

    def test_unregistered_metric_keeps_raw_samples_and_unknown_metadata(self):
        self.add_row("custom_metric", "42", "09:00:00")
        self.write_fixture()
        metric = self.metric(self.query_direct(["custom_metric"]), "custom_metric")
        self.assertEqual(metric["short_name"], "UNKNOWN")
        self.assertEqual(metric["type"], "unknown")
        self.assertIsNone(metric["unit"])
        self.assertTrue(metric["evidence_id"].startswith("metric-UNKNOWN-"))
        self.assertEqual(metric["series"][0]["samples"], [[0, 42]])

    def test_node_query_and_direct_query_have_the_same_evidence(self):
        node_result = self.query(metrics=["node_cpu_seconds_total"])
        direct_result = self.query_direct(["node_cpu_seconds_total"])
        self.assertEqual(node_result, direct_result)
        for tool in gwdg.GWDG_TOOLS:
            properties = tool["parameters"]["properties"]
            if tool["name"] == "get_node_metrics_gwdg":
                self.assertEqual(set(properties), {"node_id", "start_time", "end_time", "metrics"})

    def test_static_registry_has_compact_notes_and_counter_guidance(self):
        self.assertEqual(len(NODE_METRICS), 127)
        for name, metadata in NODE_METRIC_METADATA.items():
            with self.subTest(metric=name):
                self.assertIn(metadata["type"], {"counter", "gauge", "info", "unknown"})
                self.assertTrue(metadata["note"])
                self.assertLess(len(metadata["note"]), 160)
                self.assertNotIn("generally preferable", metadata["note"])
                self.assertNotIn("Context-dependent", metadata["note"])
                if metadata["type"] == "counter":
                    self.assertRegex(metadata["note"], "changes|increases")
                    self.assertIn("rate", metadata["note"])
        self.assertIsNone(NODE_METRIC_METADATA["node_hwmon_max_chip_temp"]["unit"])


if __name__ == "__main__":
    unittest.main()
