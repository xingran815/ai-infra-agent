import unittest

from src.dwdg import metric_metadata as metadata
from src.dwdg import telemetry


class MetricMetadataTests(unittest.TestCase):
    def test_registries_cover_the_agreed_category_lists(self):
        counts = {"NODE": 127, "GPU": 19, "NETWORK": 138, "STORAGE": 31, "SCHEDULER": 3, "MONITORING": 12}
        combined = set()
        for category, count in counts.items():
            registry = getattr(metadata, f"{category}_METRIC_METADATA")
            self.assertEqual(len(registry), count)
            self.assertFalse(combined & set(registry))
            combined.update(registry)
            self.assertEqual(set(registry), set(telemetry.METRIC_CATEGORIES[category.lower()]))
        self.assertEqual(combined, set(metadata.METRIC_METADATA))
        self.assertEqual(len(combined), 330)

    def test_short_names_are_unique_uppercase_letters_and_notes_stay_compact(self):
        short_names = []
        for name, entry in metadata.METRIC_METADATA.items():
            with self.subTest(metric=name):
                self.assertRegex(entry["short_name"], r"^[A-Z]{1,7}$")
                short_names.append(entry["short_name"])
                self.assertIn(entry["type"], {"counter", "gauge", "info", "unknown"})
                self.assertTrue(entry["note"])
                self.assertLess(len(entry["note"]), 160)
                if entry["type"] == "counter":
                    self.assertIn("changes or rates", entry["note"])
        self.assertEqual(len(short_names), len(set(short_names)))

    def test_gpu_codes_counters_units_and_version_ambiguity(self):
        gpu = metadata.GPU_METRIC_METADATA
        self.assertEqual(gpu["DCGM_FI_DEV_XID_ERRORS"]["type"], "gauge")
        self.assertIn("not an error count", gpu["DCGM_FI_DEV_XID_ERRORS"]["note"])
        self.assertEqual(gpu["DCGM_FI_DEV_FB_FREE"]["unit"], "MiB")
        self.assertEqual(gpu["DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"]["unit"], "millijoules")
        self.assertEqual(gpu["DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL"]["type"], "unknown")
        self.assertIsNone(gpu["DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL"]["unit"])

    def test_link_capacity_queue_gauges_and_scrape_counts(self):
        registry = metadata.METRIC_METADATA
        self.assertEqual(registry["node_network_speed_bytes"]["unit"], "bytes_per_second")
        self.assertIn("not measured traffic", registry["node_infiniband_rate_bytes_per_second"]["note"])
        self.assertEqual(registry["node_sockstat_TCP_mem"]["unit"], "pages")
        self.assertEqual(registry["node_disk_io_now"]["type"], "gauge")
        self.assertEqual(registry["scrape_series_added"]["type"], "gauge")
        self.assertEqual(registry["slurm_alloc_gpus"]["unit"], "GPUs")

    def test_evidence_ids_keep_hashes_stable_and_distinguish_scopes(self):
        scope = {"dataset": telemetry.SOURCE, "node": "node6a5ebc855306", "metric": "node_cpu_seconds_total",
                 "window": {"start_time": "2025-03-18T09:00:00Z", "end_time": "2025-03-18T09:00:20Z",
                            "end_time_exclusive": True, "sample_interval_seconds": 10}}
        evidence = telemetry.evidence_id("metric", scope)
        # This digest was returned before the human-readable prefix was added.
        self.assertEqual(evidence, "metric-CPUTIME-83dd5189658b296b0897473da23b6527")
        self.assertEqual(evidence, telemetry.evidence_id("metric", dict(reversed(list(scope.items())))))
        self.assertNotEqual(evidence, telemetry.evidence_id("metric", {**scope, "node": "node000000000000"}))
        series = {**scope, "identity": {"seriesId": "cpu-0"}, "labels": {"cpu": "0", "mode": "idle"}}
        first = telemetry.evidence_id("series", series)
        second = telemetry.evidence_id("series", {**series, "labels": {"cpu": "1", "mode": "idle"}})
        self.assertRegex(first, r"^series-CPUTIME-[0-9a-f]{32}$")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
