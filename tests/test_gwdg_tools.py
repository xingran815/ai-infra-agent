import csv
import json
import lzma
import tempfile
import unittest
import inspect
from pathlib import Path
from unittest.mock import patch

from src.dwdg import tools_dwdg as gwdg
from src.dwdg import telemetry
from src.dwdg.metric_metadata import (
    GPU_METRICS, NODE_METRICS, NODE_METRIC_METADATA, NETWORK_METRICS,
    STORAGE_METRICS, SCHEDULER_METRICS, MONITORING_METRICS,
)


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
        if kwargs:
            return gwdg.get_metrics_gwdg(NODE, START, END, category="node", **kwargs)
        return gwdg.get_node_metrics_gwdg(NODE, START, END)

    def query_direct(self, metrics, **kwargs):
        return gwdg.get_metrics_gwdg(NODE, START, END, metrics, **kwargs)

    def metric(self, result, metric):
        return result["nodes"][NODE]["metrics"][metric]

    def test_discovery_returns_node_and_device_ids_before_metrics_and_warms_cache(self):
        inventory = gwdg.get_resource_ids_gwdg(START, END)
        self.assertEqual(inventory["node_list"], [NODE])
        self.assertEqual(inventory["resource_ids"], sorted([NODE, "gpu1", "eth1", "mlx5_0", "nvme0n1"]))
        resources = inventory["nodes"][NODE]["resources"]
        self.assertEqual(resources["node_id"], NODE)
        self.assertEqual(resources["cpus"]["logical_cpu_ids"], list(range(128)))
        self.assertNotIn("metrics", inventory["nodes"][NODE])
        self.assertNotIn("outside-window", inventory["resource_ids"])
        with patch.object(telemetry.pd, "read_csv", side_effect=AssertionError("Discovery window was read again")):
            self.assertEqual(inventory, gwdg.get_resource_ids_gwdg(
                "2025-03-18T10:00:00+01:00", "2025-03-18T10:00:30+01:00"))
            gwdg.get_node_metrics_gwdg(resource_id=NODE, start_time=START, end_time=END)
            gwdg.get_gpu_metrics_gwdg(NODE, "gpu1", START, END)
            gwdg.get_network_metrics_gwdg(NODE, "eth1", START, END)
            gwdg.get_storage_metrics_gwdg(NODE, "nvme0n1", START, END)
            gwdg.get_monitoring_metrics_gwdg(NODE, START, END)

    def test_discovery_includes_filesystem_nvme_and_dictionary_only_resource_ids(self):
        self.add_row("node_filesystem_avail_bytes", "100", "09:00:00", device="/dev/nvme0n1p6")
        self.add_row("node_nvme_info", "1", "09:00:00", device="nvme0")
        self.labels.append(("dictionary-interface", "node_network_up", {"device": "eth-dictionary"}))
        self.add_row("node_network_up", "1", "09:00:00", "dictionary-interface")
        self.write_fixture()
        gwdg.load_data_gwdg(str(self.path))
        inventory = gwdg.get_resource_ids_gwdg(START, END)
        resources = inventory["nodes"][NODE]["resources"]
        self.assertEqual(resources["storage_ids"], ["/dev/nvme0n1p6", "nvme0", "nvme0n1"])
        self.assertIn("eth-dictionary", resources["network_interface_ids"])
        result = gwdg.get_network_metrics_gwdg(NODE, "eth-dictionary", START, END)
        self.assertEqual(self.metric(result, "node_network_up")["series"][0]["labels"]["device"], "eth-dictionary")

    def test_discovery_empty_window_and_invalid_times(self):
        empty = gwdg.get_resource_ids_gwdg("2025-03-18T11:00:00Z", "2025-03-18T11:00:20Z")
        self.assertEqual(empty["nodes"], {})
        self.assertEqual(empty["node_list"], [])
        self.assertEqual(empty["resource_ids"], [])
        for start, end in ((START, START), (END, START), ("2025-03-18T09:00:00", END)):
            with self.assertRaises(ValueError):
                gwdg.get_resource_ids_gwdg(start, end)

    def test_generic_category_defaults_all_resource_filter_and_monitor_alias(self):
        self.add_row("custom_metric", "42", "09:00:00")
        self.add_row("up", "1", "09:00:00")
        self.write_fixture()
        result = gwdg.get_metrics_gwdg(NODE, START, END)
        expected = {row["metric"] for row in self.rows if START[11:19] <= row["timeUtc"][11:19] < END[11:19]}
        self.assertEqual(set(result["nodes"][NODE]["metrics"]), expected)
        self.assertEqual(result, gwdg.get_metrics_gwdg(NODE, START, END, resource_id=NODE))
        filtered = gwdg.get_metrics_gwdg(NODE, START, END, resource_id="gpu1")
        metrics = filtered["nodes"][NODE]["metrics"]
        self.assertIn("DCGM_FI_DEV_GPU_TEMP", metrics)
        self.assertIn("node_load1", metrics)
        self.assertNotIn("node_network_up", metrics)
        self.assertNotIn("node_disk_reads_completed_total", metrics)
        self.assertEqual(gwdg.get_metrics_gwdg(NODE, START, END, category="monitor"),
                         gwdg.get_monitoring_metrics_gwdg(NODE, START, END))
        self.assertEqual(gwdg.get_metrics_gwdg(NODE, START, END, category="gpu", resource_id="gpu1"),
                         gwdg.get_gpu_metrics_gwdg(NODE, "gpu1", START, END))
        for category in ("unknown", None, []):
            with self.subTest(category=category), self.assertRaisesRegex(ValueError, "Unsupported metric category"):
                gwdg.get_metrics_gwdg(NODE, START, END, category=category)
        for category in ("node", "scheduler", "monitoring"):
            with self.assertRaisesRegex(ValueError, "resource_id must be the node ID"):
                gwdg.get_metrics_gwdg(NODE, START, END, category=category, resource_id="gpu1")

    def test_category_wrappers_reject_null_and_empty_resource_ids(self):
        for tool in gwdg.GWDG_TOOLS:
            if "resource_id" not in tool["parameters"]["required"]:
                continue
            function = getattr(gwdg, tool["name"])
            args = {"start_time": START, "end_time": END}
            if "node_id" in tool["parameters"]["properties"]:
                args["node_id"] = NODE
            for resource_id in (None, "", 1):
                with self.subTest(tool=tool["name"], resource_id=resource_id), self.assertRaises(ValueError):
                    function(resource_id=resource_id, **args)

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
        changed_window = gwdg.get_metrics_gwdg(NODE, START, "2025-03-18T09:00:20Z",
                                                  category="node", metrics=["node_cpu_seconds_total"])
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
        result = gwdg.get_metrics_gwdg(NODE, "2025-03-18T09:00:05Z",
                                          "2025-03-18T09:00:21Z", category="node", metrics=["node_load1"])
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

    def test_gpu_query_filters_uuid_preserves_gaps_and_reuses_node_window(self):
        self.add_row("DCGM_FI_DEV_GPU_TEMP", "40", "09:00:20", uuid="gpu1", device="nvidia0")
        self.add_row("DCGM_FI_DEV_GPU_TEMP", "99", "09:00:30", uuid="gpu1", device="nvidia0")
        self.add_row("DCGM_FI_DEV_GPU_TEMP", "80", "09:00:00", uuid="gpu2", device="nvidia1")
        self.write_fixture()
        node_result = self.query(metrics=["node_load1"])
        gpu_id = node_result["nodes"][NODE]["resources"]["gpu_ids"][0]
        with patch.object(telemetry.pd, "read_csv", side_effect=AssertionError("Window was read again")):
            result = gwdg.get_gpu_metrics_gwdg(NODE, gpu_id, START, END)
            direct = self.query_direct(["DCGM_FI_DEV_GPU_TEMP"], label_filters={"uuid": gpu_id})
        self.assertEqual(result, direct)
        metric = self.metric(result, "DCGM_FI_DEV_GPU_TEMP")
        self.assertEqual(len(metric["series"]), 1)
        self.assertEqual(metric["series"][0]["labels"]["uuid"], "gpu1")
        self.assertEqual(metric["series"][0]["samples"], [[0, 30], [2, 40]])
        self.assertEqual(metric["series"][0]["coverage"], {"expected_samples": 3, "returned_samples": 2})
        self.assertTrue(metric["series"][0]["evidence_id"].startswith("series-GPUTEMP-"))
        self.assertEqual(result["nodes"][NODE]["resources"]["gpu_ids"], ["gpu1", "gpu2"])

    def test_gpu_query_defaults_selection_and_category_validation(self):
        self.add_row("DCGM_FI_DEV_POWER_USAGE", "110", "09:00:00", uuid="gpu1", device="nvidia0")
        self.write_fixture()
        result = gwdg.get_gpu_metrics_gwdg(NODE, "gpu1", START, END)
        self.assertEqual(set(result["nodes"][NODE]["metrics"]), {"DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_DEV_POWER_USAGE"})
        selected = gwdg.get_metrics_gwdg(NODE, START, END, category="gpu", resource_id="gpu1", metrics=["DCGM_FI_DEV_POWER_USAGE"])
        self.assertEqual(set(selected["nodes"][NODE]["metrics"]), {"DCGM_FI_DEV_POWER_USAGE"})
        empty = gwdg.get_metrics_gwdg(NODE, START, END, category="gpu", resource_id="gpu1", metrics=[])
        self.assertEqual(empty["nodes"][NODE]["metrics"], {})
        with self.assertRaisesRegex(ValueError, "Unsupported gpu metrics"):
            gwdg.get_metrics_gwdg(NODE, START, END, category="gpu", resource_id="gpu1", metrics=["node_load1"])

    def test_gpu_query_returns_no_readings_for_unknown_gpu_node_or_absent_dcgm(self):
        unknown_gpu = gwdg.get_gpu_metrics_gwdg(NODE, "unknown-gpu", START, END)
        self.assertEqual(unknown_gpu["nodes"][NODE]["metrics"], {})
        unknown_node = gwdg.get_gpu_metrics_gwdg("node000000000000", "gpu1", START, END)
        self.assertEqual(unknown_node["nodes"], {})
        self.rows = [row for row in self.rows if not row["metric"].startswith("DCGM_")]
        self.write_fixture()
        gwdg.load_data_gwdg(str(self.path))
        missing = gwdg.get_gpu_metrics_gwdg(NODE, "gpu1", START, END)
        self.assertEqual(missing["nodes"][NODE]["metrics"], {})
        self.assertEqual(missing["nodes"][NODE]["resources"]["gpu_ids"], [])

    def test_network_interface_filter_keeps_node_wide_and_per_cpu_metrics(self):
        self.add_row("node_network_up", "1", "09:00:00", device="eth2")
        self.add_row("node_netstat_Tcp_CurrEstab", "5", "09:00:00")
        self.labels.append(("softnet", "node_softnet_processed_total", {"cpu": "0"}))
        self.add_row("node_softnet_processed_total", "42", "09:00:00", "softnet")
        self.write_fixture()
        gwdg.load_data_gwdg(str(self.path))
        self.query(metrics=["node_load1"])
        with patch.object(telemetry.pd, "read_csv", side_effect=AssertionError("Window was read again")):
            result = gwdg.get_network_metrics_gwdg(NODE, "eth1", START, END)
        metrics = result["nodes"][NODE]["metrics"]
        self.assertEqual(set(metrics), {"node_network_up", "node_netstat_Tcp_CurrEstab", "node_softnet_processed_total"})
        self.assertEqual(metrics["node_network_up"]["series"][0]["labels"]["device"], "eth1")
        self.assertEqual(result["nodes"][NODE]["resources"]["network_interface_ids"], ["eth1", "eth2"])
        all_interfaces = gwdg.get_metrics_gwdg(NODE, START, END, category="network", metrics=["node_network_up"])
        self.assertEqual(len(self.metric(all_interfaces, "node_network_up")["series"]), 2)
        adapter = gwdg.get_network_metrics_gwdg(NODE, "mlx5_0", START, END)
        self.assertIn("node_infiniband_state_id", adapter["nodes"][NODE]["metrics"])
        self.assertNotIn("node_network_up", adapter["nodes"][NODE]["metrics"])
        missing = gwdg.get_network_metrics_gwdg(NODE, "unknown-interface", START, END)
        self.assertEqual(set(missing["nodes"][NODE]["metrics"]), {"node_netstat_Tcp_CurrEstab", "node_softnet_processed_total"})

    def test_storage_device_filter_keeps_nfs_and_matches_filesystems_exactly(self):
        self.add_row("node_disk_reads_completed_total", "20", "09:00:00", device="nvme1n1")
        self.add_row("node_nfs_rpcs_total", "9", "09:00:00")
        for sid in ("fs-a", "fs-b"):
            self.labels.append((sid, "node_filesystem_avail_bytes", {"mountpoint": "sanitized", "fstype": "ext4"}))
            self.add_row("node_filesystem_avail_bytes", "100", "09:00:00", sid, device="/dev/nvme0n1p6")
        self.add_row("node_filesystem_avail_bytes", "200", "09:00:00", device="/dev/nvme1n1p6")
        self.add_row("node_nvme_info", "1", "09:00:00", device="nvme0")
        self.write_fixture()
        gwdg.load_data_gwdg(str(self.path))
        self.query(metrics=["node_load1"])
        with patch.object(telemetry.pd, "read_csv", side_effect=AssertionError("Window was read again")):
            result = gwdg.get_storage_metrics_gwdg(NODE, "nvme0n1", START, END)
        metrics = result["nodes"][NODE]["metrics"]
        self.assertEqual(set(metrics), {"node_disk_reads_completed_total", "node_nfs_rpcs_total"})
        self.assertEqual(metrics["node_disk_reads_completed_total"]["series"][0]["labels"]["device"], "nvme0n1")
        self.assertEqual(result["nodes"][NODE]["resources"]["storage_ids"], ["/dev/nvme0n1p6", "/dev/nvme1n1p6", "nvme0", "nvme0n1", "nvme1n1"])
        filesystem = gwdg.get_storage_metrics_gwdg(NODE, "/dev/nvme0n1p6", START, END)
        series = self.metric(filesystem, "node_filesystem_avail_bytes")["series"]
        self.assertEqual({s["series_id"] for s in series}, {"fs-a", "fs-b"})
        self.assertEqual(len({s["evidence_id"] for s in series}), 2)
        controller = gwdg.get_storage_metrics_gwdg(NODE, "nvme0", START, END)
        self.assertIn("node_nvme_info", controller["nodes"][NODE]["metrics"])
        missing = gwdg.get_storage_metrics_gwdg(NODE, "unknown-device", START, END)
        self.assertEqual(set(missing["nodes"][NODE]["metrics"]), {"node_nfs_rpcs_total"})

    def test_monitoring_preserves_collector_job_series_and_missing_samples(self):
        for sid, collector in (("collector-cpu", "cpu"), ("collector-memory", "meminfo")):
            self.labels.append((sid, "node_scrape_collector_success", {"collector": collector}))
            self.add_row("node_scrape_collector_success", "1", "09:00:00", sid)
            self.add_row("node_scrape_collector_success", "0", "09:00:20", sid)
        for job in ("exporter-a", "exporter-b"):
            row = dict.fromkeys(FIELDS, "")
            row.update(timeUtc="2025-03-18 09:00:00", utcOffsetHours="+1", metric="up", value="1", job=job)
            self.rows.append(row)
        self.add_row("up", "0", "09:00:30")
        self.write_fixture()
        gwdg.load_data_gwdg(str(self.path))
        self.query(metrics=["node_load1"])
        with patch.object(telemetry.pd, "read_csv", side_effect=AssertionError("Window was read again")):
            result = gwdg.get_monitoring_metrics_gwdg(NODE, START, END)
        self.assertEqual(result, self.query_direct(list(MONITORING_METRICS)))
        collectors = self.metric(result, "node_scrape_collector_success")["series"]
        self.assertEqual({s["labels"]["collector"] for s in collectors}, {"cpu", "meminfo"})
        self.assertTrue(all(s["samples"] == [[0, 1], [2, 0]] for s in collectors))
        self.assertTrue(all(s["coverage"] == {"expected_samples": 3, "returned_samples": 2} for s in collectors))
        jobs = self.metric(result, "up")["series"]
        self.assertEqual({s["labels"]["job"] for s in jobs}, {"exporter-a", "exporter-b"})
        selected = gwdg.get_metrics_gwdg(NODE, START, END, category="monitoring", metrics=["up"])
        self.assertEqual(set(selected["nodes"][NODE]["metrics"]), {"up"})

    def test_incident_category_empty_selections_validation_and_unknown_nodes(self):
        for category in ("network", "storage", "monitoring"):
            with self.subTest(category=category):
                self.assertEqual(gwdg.get_metrics_gwdg(NODE, START, END, category=category, metrics=[])["nodes"][NODE]["metrics"], {})
                self.assertEqual(gwdg.get_metrics_gwdg("node000000000000", START, END, category=category)["nodes"], {})
                with self.assertRaises(ValueError):
                    gwdg.get_metrics_gwdg(NODE, START, END, category=category, metrics=["node_load1"])

    def test_tool_schemas_and_required_arguments_match_functions(self):
        self.assertEqual(gwdg.GWDG_TOOLS[0]["name"], "get_resource_ids_gwdg")
        for tool in gwdg.GWDG_TOOLS:
            function = getattr(gwdg, tool["name"])
            signature = inspect.signature(function)
            parameters = tool["parameters"]
            self.assertEqual(set(parameters["properties"]), set(signature.parameters))
            python_required = {
                name for name, parameter in signature.parameters.items()
                if parameter.default is inspect.Parameter.empty
            }
            self.assertTrue(python_required <= set(parameters["required"]))
            if not tool["strict"]:
                self.assertEqual(set(parameters["required"]), python_required)
            self.assertFalse(parameters["additionalProperties"])
            if tool["name"] == "get_metrics_gwdg":
                self.assertFalse(tool["strict"])
                self.assertEqual(parameters["properties"]["category"]["enum"], list(telemetry.METRIC_CATEGORIES))
            else:
                self.assertTrue(tool["strict"])
                self.assertEqual(set(parameters["required"]), set(parameters["properties"]))
                if "resource_id" in signature.parameters:
                    self.assertIn("metrics", parameters["properties"])
                    self.assertEqual(parameters["properties"]["metrics"]["type"], ["array", "null"])
                    with self.assertRaises(TypeError):
                        function(**{name: None if name == "metrics" else NODE if name == "node_id" else START
                                    for name in signature.parameters if name != "resource_id"})

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
                self.assertEqual(set(properties), {"resource_id", "start_time", "end_time", "metrics"})

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
