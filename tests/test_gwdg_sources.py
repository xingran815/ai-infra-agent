import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dwdg import telemetry, tools_dwdg as api


NODE = "nodetest"
START = "2025-03-18T09:00:00Z"
END = "2025-03-18T09:00:20Z"


class DataSourceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(api.unload_data_gwdg)
        self.root = Path(directory.name)
        self.paths = {}
        for name, value in (("a", 1), ("b", 9)):
            folder = self.root / name
            folder.mkdir()
            path = folder / f"{NODE}_incident.csv"
            rows = []
            labels = []
            for metric, extra in (
                ("node_load1", {}),
                ("DCGM_FI_DEV_GPU_TEMP", {"uuid": f"gpu-{name}"}),
                ("node_network_up", {"device": f"eth-{name}"}),
                ("node_disk_io_now", {"device": f"disk-{name}"}),
                ("up", {}),
            ):
                # Identical series IDs resolve differently in each source.
                labels.append(("same-id", metric, json.dumps({"job": name, **extra})))
                for time in (START, "2025-03-18T09:00:10Z"):
                    rows.append(dict(timeUtc=time, metric=metric, value=value, seriesId="same-id"))
            with path.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["timeUtc", "value", *telemetry.IDENTITY_COLUMNS])
                writer.writeheader()
                writer.writerows(rows)
            with (folder / "series_labels.tsv").open("w") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["seriesId", "metric", "labels"])
                writer.writerows(labels)
            scheduler = folder / "scheduler.csv"
            with scheduler.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["node", "timeUtc", "value", *telemetry.IDENTITY_COLUMNS])
                writer.writeheader()
                writer.writerow(dict(node=NODE, timeUtc=START, value=value, metric="slurm_alloc_gpus", job=name))
            self.paths[name] = path
        self.a, self.b = self.source("a"), self.source("b")
        self.addCleanup(self.a.clear_cache)
        self.addCleanup(self.b.clear_cache)

    def source(self, name):
        path = self.paths[name]
        return api.GwdgDataSource(str(path), scheduler_path=str(path.parent / "scheduler.csv"))

    def query(self, source, **kwargs):
        return api.get_metrics_gwdg(NODE, START, END, source=source, **kwargs)

    def test_interleaved_sources_keep_labels_inventory_and_all_tool_categories_isolated(self):
        calls = [
            ("get_node_metrics_gwdg", "node_load1", NODE),
            ("get_gpu_metrics_gwdg", "DCGM_FI_DEV_GPU_TEMP", "gpu-{name}"),
            ("get_network_metrics_gwdg", "node_network_up", "eth-{name}"),
            ("get_storage_metrics_gwdg", "node_disk_io_now", "disk-{name}"),
            ("get_monitoring_metrics_gwdg", "up", NODE),
            ("get_scheduler_metrics_gwdg", "slurm_alloc_gpus", NODE),
        ]
        for source, name, value in ((self.a, "a", 1), (self.b, "b", 9), (self.a, "a", 1)):
            inventory = api.call_function_dwdg("get_resource_ids_gwdg", dict(start_time=START, end_time=END), "id", source=source)
            self.assertEqual(inventory["nodes"][NODE]["resources"]["gpu_ids"], [f"gpu-{name}"])
            for tool, metric, resource in calls:
                with self.subTest(source=name, tool=tool):
                    args = dict(resource_id=resource.format(name=name), start_time=START, end_time=END, metrics=[metric])
                    if resource != NODE:
                        args["node_id"] = NODE
                    result = api.call_function_dwdg(tool, args, "id", source=source)
                    series = result["nodes"][NODE]["metrics"][metric]["series"][0]
                    self.assertEqual(series["labels"]["job"], name)
                    self.assertEqual(series["sample_runs"][0][2], value)
            overview = api.call_function_dwdg("get_overview_gwdg", dict(node_id=NODE, start_time=START, end_time=END), "id", source=source)
            self.assertEqual(overview["categories"]["node"]["metrics"]["node_load1"]["series"][0]["latest"], value)
            self.assertEqual(overview["categories"]["scheduler"]["metrics"]["slurm_alloc_gpus"]["series"][0]["latest"], value)

    def test_default_source_replacement_and_unload_do_not_redirect_explicit_sources(self):
        a = api.load_data_gwdg(str(self.paths["a"]))
        api.load_data_gwdg(str(self.paths["b"]))
        api.unload_data_gwdg()
        # Query a new window to ensure this tests the reader, not just cached data.
        for source in (a, self.a, self.b):
            result = api.get_metrics_gwdg(NODE, START, "2025-03-18T09:00:10Z", source=source, metrics=["node_load1"])
            value = 9 if source is self.b else 1
            self.assertEqual(result["nodes"][NODE]["metrics"]["node_load1"]["series"][0]["sample_runs"], [[0, 1, value]])

    def test_each_source_reuses_its_own_incident_and_scheduler_windows(self):
        expected = {}
        for source in (self.a, self.b):
            for category in ("all", "scheduler"):
                expected[source, category] = self.query(source, category=category)
        with patch.object(telemetry.pd, "read_csv", side_effect=AssertionError("Unexpected reread")):
            for source in (self.a, self.b, self.a):
                for category in ("all", "scheduler"):
                    result = api.get_metrics_gwdg(NODE, "2025-03-18T10:00:00+01:00", "2025-03-18T10:00:20+01:00", category=category, source=source)
                    self.assertEqual(result, expected[source, category])
        self.a.clear_cache()
        with patch.object(telemetry.pd, "read_csv", side_effect=AssertionError("Other cache was cleared")):
            self.assertEqual(self.query(self.b), expected[self.b, "all"])
        for second in (21, 22, 23):
            api.get_metrics_gwdg(NODE, START, f"2025-03-18T09:00:{second}Z", source=self.a)
        self.assertEqual(self.a._incident_cache.cache_info().currsize, 1)

    def test_revision_tracks_paths_and_each_source_file_version(self):
        same = self.source("a")
        self.assertEqual(self.a.revision, same.revision)
        self.assertNotEqual(self.a.revision, self.b.revision)
        previous = same.revision
        for path in (same.path, same.labels_path, same.scheduler_path):
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            refreshed = self.source("a")
            self.assertNotEqual(refreshed.revision, previous)
            previous = refreshed.revision

    def test_reopening_an_edited_fixture_reads_new_values(self):
        self.query(self.a)  # Populate the previous object's cache first.
        path = self.paths["a"]
        with path.open() as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            row["value"] = "77"
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        refreshed = self.source("a")
        self.addCleanup(refreshed.clear_cache)
        self.assertNotEqual(self.a.revision, refreshed.revision)
        result = self.query(refreshed, metrics=["node_load1"])
        self.assertEqual(result["nodes"][NODE]["metrics"]["node_load1"]["series"][0]["sample_runs"], [[0, 2, 77]])

    def test_dispatch_cannot_accept_a_model_supplied_source(self):
        with self.assertRaises(TypeError):
            api.call_function_dwdg("get_resource_ids_gwdg", dict(start_time=START, end_time=END, source=self.b), "id", source=self.a)


if __name__ == "__main__":
    unittest.main()
