import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.dwdg import response_limits as limits
from src.dwdg import telemetry, tools_dwdg as api


NODE = "nodetest"
WINDOW = {"start_time": "2025-03-19T07:30:00Z", "end_time": "2025-03-19T07:30:20Z"}


def raw_result(count=1, runs=1):
    return {
        "time_axis": {"start": WINDOW["start_time"], "step_seconds": 10, "count": runs},
        "window": dict(WINDOW),
        "nodes": {NODE: {"metrics": {"node_scrape_collector_success": {
            "evidence_id": "metric-withheld",
            "series": [
                {"labels": {"collector": f"collector-{index:03}", "job": "integrations/unix"},
                 "evidence_id": f"series-withheld-{index}",
                 "sample_runs": [[index, 1, index % 2] for index in range(runs)]}
                for index in range(count)
            ],
        }}}},
    }


class ResponseLimitTests(unittest.TestCase):
    def limit(self, result, **kwargs):
        return limits.limit_tool_result(
            "get_monitoring_metrics_gwdg", dict(resource_id=NODE, **WINDOW, **kwargs), result,
        )

    def test_small_result_is_unchanged_and_thresholds_are_inclusive(self):
        for result in (raw_result(count=64), raw_result(runs=1000)):
            self.assertIs(self.limit(result), result)
        result = raw_result()
        result["padding"] = ""
        size = len(limits.serialize_tool_result(result).encode("utf-8"))
        result["padding"] = "x" * (limits.RAW_MAX_BYTES - size)
        self.assertIs(self.limit(result), result)
        result["padding"] += "x"
        self.assertEqual(self.limit(result)["exceeded"], ["response_bytes"])

    def test_too_many_series_returns_real_selectors_no_readings_and_no_mutation(self):
        result = raw_result(count=65)
        original = copy.deepcopy(result)
        response = self.limit(result)
        self.assertEqual(response["status"], "query_too_broad")
        self.assertFalse(response["data_returned"])
        self.assertIn("series", response["exceeded"])
        self.assertEqual(response["observed"]["series"], 65)
        self.assertEqual(response["window"], WINDOW)
        hint = response["available"]["metrics"][0]
        collectors = hint["label_selectors"]["collector"]
        self.assertEqual(collectors["total_values"], 65)
        self.assertEqual(collectors["omitted_values"], 65 - len(collectors["values"]))
        self.assertEqual(hint["label_selectors"]["job"]["values"], ["integrations/unix"])
        encoded = limits.serialize_tool_result(response)
        for withheld in ('"evidence_id"', '"sample_runs":[', "metric-withheld", "series-withheld"):
            self.assertNotIn(withheld, encoded)
        self.assertEqual(result, original)

    def test_run_limit_and_actual_utf8_byte_limit(self):
        response = self.limit(raw_result(runs=1001))
        self.assertIn("sample_runs", response["exceeded"])
        result = raw_result()
        result["metadata"] = "界" * 9000
        encoded = limits.serialize_tool_result(result)
        self.assertLess(len(encoded), limits.RAW_MAX_BYTES)
        response = self.limit(result)
        self.assertEqual(response["observed"]["response_bytes"], len(encoded.encode("utf-8")))
        self.assertEqual(response["exceeded"], ["response_bytes"])

    def test_retry_preserves_constraints_and_uses_an_observed_combination(self):
        result = raw_result(count=65)
        for index, series in enumerate(result["nodes"][NODE]["metrics"]["node_scrape_collector_success"]["series"]):
            series["labels"].update(job=f"job-{index}", device="eth0", mode="idle")
        args = dict(node_id=NODE, resource_id="eth0", category="all", label_filters={"mode": "idle"}, **WINDOW)
        response = limits.limit_tool_result("get_metrics_gwdg", args, result)
        retry = response["available"]["metrics"][0]["example_retry"]
        self.assertEqual(retry["tool"], "get_metrics_gwdg")
        retry_args = retry["arguments"]
        self.assertEqual(retry_args["resource_id"], "eth0")
        self.assertEqual(retry_args["category"], "all")
        self.assertEqual(retry_args["label_filters"], {
            "collector": "collector-000", "job": "job-0", "device": "eth0", "mode": "idle",
        })
        self.assertEqual({key: retry_args[key] for key in WINDOW}, WINDOW)

    def test_large_selector_catalog_is_bounded_and_reports_omissions(self):
        result = raw_result(count=200)
        metrics = result["nodes"][NODE]["metrics"]
        metric = metrics.pop("node_scrape_collector_success")
        for index, series in enumerate(metric["series"]):
            series["labels"]["long_value"] = "界" * 300 + str(index)
            series["labels"].update({f"extra-{key}": "界" * 100 + str(index) for key in range(20)})
        metrics.update({f"metric-{index}": metric for index in range(40)})
        response = self.limit(result)
        catalog = response["available"]
        self.assertLessEqual(len(limits.serialize_tool_result(response).encode("utf-8")), limits.HINT_MAX_BYTES)
        self.assertGreater(len(catalog["metrics"]), 0)
        self.assertGreater(catalog["omitted_metrics"], 0)
        self.assertEqual(catalog["total_metrics"], len(catalog["metrics"]) + catalog["omitted_metrics"])
        for hint in catalog["metrics"]:
            self.assertGreater(hint["omitted_labels"], 0)
            for entry in hint["label_selectors"].values():
                self.assertEqual(entry["total_values"], len(entry["values"]) + entry["omitted_values"])
                self.assertTrue(all(len(value) <= limits.MAX_SELECTOR_LENGTH for value in entry["values"]))

    def test_overview_has_separate_budget_and_discloses_partial_selector_scope(self):
        metric = raw_result(count=65)["nodes"][NODE]["metrics"]
        result = {"node_id": NODE, "window": WINDOW, "categories": {"monitoring": {"metrics": metric}}}
        args = dict(node_id=NODE, **WINDOW)
        self.assertIs(limits.limit_tool_result("get_overview_gwdg", args, result), result)
        result["padding"] = "x" * limits.OVERVIEW_MAX_BYTES
        response = limits.limit_tool_result("get_overview_gwdg", args, result)
        self.assertEqual(response["available"]["scope"], "returned_overview_details")
        self.assertEqual(response["limits"], {"response_bytes": limits.OVERVIEW_MAX_BYTES})
        self.assertEqual(response["available"]["metrics"][0]["example_retry"]["arguments"]["category"], "monitoring")

    def test_discovery_overflow_is_explicit(self):
        result = {"nodes": {}, "resource_ids": ["x" * limits.RAW_MAX_BYTES]}
        response = limits.limit_tool_result("get_resource_ids_gwdg", WINDOW, result)
        self.assertFalse(response["data_returned"])
        self.assertIn("resource discovery", response["retry_guidance"][0])

    def test_collector_retry_executes_through_dispatcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / f"{NODE}_incident.csv"
            metrics = ("node_scrape_collector_success", "node_scrape_collector_duration_seconds")
            with path.open("w") as stream, (root / "series_labels.tsv").open("w") as label_stream:
                writer = csv.DictWriter(stream, fieldnames=["timeUtc", "value", *telemetry.IDENTITY_COLUMNS])
                writer.writeheader()
                labels = csv.writer(label_stream, delimiter="\t")
                labels.writerow(["seriesId", "metric", "labels"])
                for name in metrics:
                    for index in range(41):
                        sid = f"collector-{index}"
                        labels.writerow([sid, name, json.dumps({"collector": sid, "job": "integrations/unix"})])
                        writer.writerow(dict(timeUtc=WINDOW["start_time"], value=0, metric=name, seriesId=sid))
            source = telemetry.GwdgDataSource(path)
            self.addCleanup(source.clear_cache)
            response = api.call_function_dwdg("get_monitoring_metrics_gwdg", dict(resource_id=NODE, **WINDOW), "broad", source=source)
            self.assertEqual(response["observed"]["series"], 82)
            for hint in response["available"]["metrics"]:
                self.assertEqual(hint["label_selectors"]["collector"]["total_values"], 41)
                self.assertEqual(hint["label_selectors"]["collector"]["omitted_values"], 0)
                retry = hint["example_retry"]
                result = api.call_function_dwdg(retry["tool"], retry["arguments"], "narrow", source=source)
                series = result["nodes"][NODE]["metrics"][hint["metric"]]["series"]
                self.assertEqual(len(series), 1)
                self.assertEqual(series[0]["labels"]["collector"], retry["arguments"]["label_filters"]["collector"])
                self.assertEqual(result["window"]["start_time"], WINDOW["start_time"])
                self.assertEqual(result["window"]["end_time"], WINDOW["end_time"])


if __name__ == "__main__":
    unittest.main()
