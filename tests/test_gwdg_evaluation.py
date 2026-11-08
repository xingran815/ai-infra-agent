import csv
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from src.dwdg import evaluation
from src.dwdg.execution import ToolArgumentError
from src.dwdg.telemetry import DATA_DIR, IDENTITY_COLUMNS, GwdgDataSource


NODE = "nodeaaaaaaaaaaaa"
OTHER = "nodebbbbbbbbbbbb"
EVENT_FIELDS = (
    "node", "machineLogDate", "slurmDetectedTime", "slurmDetectedUtcOffsetHours",
    "slurmPreviousHealthyTime", "slurmPreviousHealthyUtcOffsetHours", "description",
    "telemetryFile", "telemetryStatus",
)


def write_dataset(root: Path):
    """Minimal incident catalog: two included incidents and one without telemetry."""
    rows = [
        (NODE, "2025-03-19", "2025-03-19 09:10:00", "+1", "2025-03-18 19:00:00", "+1",
         "gpus fell off bus", f"{NODE}_2025-03-19_gpus-fell-off-bus_tidy.csv.xz", "included"),
        (OTHER, "2025-04-26", "2025-04-26 05:10:00", "+2", "2025-04-24 14:10:00", "+2",
         "gpu problem", f"{OTHER}_2025-04-26_gpu-problem_tidy.csv.xz", "included"),
        (OTHER, "2025-07-11", "", "", "", "", "gpu ecc problem", "", "not-collected"),
    ]
    with (root / "incident_events.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(EVENT_FIELDS)
        writer.writerows(rows)
    for node, start, end, name in (
        (NODE, "2025-03-18 09:00:00", "2025-03-19 11:00:00", rows[0][7]),
        (OTHER, "2025-04-25 05:00:00", "2025-04-26 07:00:00", rows[1][7]),
    ):
        meta = {"node": node, "collectStart": start, "collectEnd": end, "slurmDetectedTime": "ignored"}
        (root / name.replace("_tidy.csv.xz", "_meta.json")).write_text(json.dumps(meta))


def slurm_status(node, start, values):
    times = pd.date_range(start, periods=len(values), freq="600s", tz="UTC")
    return pd.DataFrame({"node": node, "time": times, "value": values})


def incident(**overrides):
    fields = dict(
        case_id="case", kind="incident", node=NODE, telemetry_file="file_tidy.csv.xz",
        window_start="2025-03-18T09:00:00Z", window_end="2025-03-19T11:00:00Z",
        reported_time="2025-03-19T09:10:00Z", last_healthy_time="2025-03-18T19:00:00Z",
        expected_component="gpu", onset_bracket=("2025-03-18T19:00:00Z", "2025-03-19T09:10:00Z"),
        prompt="prompt",
    )
    fields.update(overrides)
    return evaluation.EvalCase(**fields)


def healthy(**overrides):
    return incident(**{"case_id": "healthy", "kind": "healthy", "expected_component": "unknown",
                       "onset_bracket": None, **overrides})


def report(component="gpu", onset=None, trace=(), status="completed", **extra):
    return {
        "severity": "high", "confidence": 0.9, "status": status, "model_runs": 3, "tool_attempts": len(trace),
        "diagnosis": {"suspected_component": component, "onset_time": onset},
        "tool_trace": list(trace), **extra,
    }


class CaseConstructionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        write_dataset(self.root)

    def test_only_included_incidents_with_meta_windows_and_utc_bracket(self):
        cases = evaluation.load_incident_cases(self.root)
        self.assertEqual([case.node for case in cases], [NODE, OTHER])
        first = cases[0]
        self.assertEqual(first.case_id, f"{NODE}_2025-03-19_gpus-fell-off-bus")
        self.assertEqual((first.window_start, first.window_end), ("2025-03-18T09:00:00Z", "2025-03-19T11:00:00Z"))
        self.assertEqual(first.onset_bracket, ("2025-03-18T19:00:00Z", "2025-03-19T09:10:00Z"))
        self.assertEqual(first.expected_component, "gpu")

    def test_prompt_states_node_times_and_window_but_not_the_label(self):
        for case in evaluation.load_incident_cases(self.root):
            with self.subTest(case=case.node):
                for value in (case.node, case.reported_time, case.last_healthy_time, case.window_start, case.window_end):
                    self.assertIn(value, case.prompt)
                for leaked in ("fell off bus", "fell-off-bus", "gpu problem", "gpu-problem", "_tidy"):
                    self.assertNotIn(leaked, case.prompt)

    def test_longest_healthy_run_breaks_at_unhealthy_samples_and_gaps(self):
        status = pd.concat([
            slurm_status(NODE, "2025-03-18 09:00", [1, 1, 0, 1, 1, 1, 1]),
            slurm_status(NODE, "2025-03-18 11:00", [1, 1]),  # 50-minute gap after the previous run
        ], ignore_index=True)
        run = evaluation.longest_healthy_run(status, pd.Timestamp("2025-03-18", tz="UTC"), pd.Timestamp("2025-03-19", tz="UTC"))
        self.assertEqual(run, (pd.Timestamp("2025-03-18 09:30", tz="UTC"), pd.Timestamp("2025-03-18 10:00", tz="UTC")))

    def test_healthy_windows_end_before_previous_healthy_and_stay_ok(self):
        cases = evaluation.load_incident_cases(self.root)
        status = pd.concat([
            # NODE: healthy from 09:00 through 23:00; the bracket opens at 19:00.
            slurm_status(NODE, "2025-03-18 09:00", [1] * 85),
            # OTHER: an unhealthy sample splits its window into short runs.
            slurm_status(OTHER, "2025-04-25 05:00", [1] * 12 + [0] + [1] * 12),
        ], ignore_index=True)
        selected = evaluation.select_healthy_cases(cases, status, 5, min_hours=4, margin_hours=1)
        self.assertEqual(len(selected), 1)
        control = selected[0]
        self.assertEqual((control.kind, control.node, control.expected_component), ("healthy", NODE, "unknown"))
        self.assertIsNone(control.onset_bracket)
        self.assertEqual((control.window_start, control.window_end), ("2025-03-18T09:00:00Z", "2025-03-18T18:00:00Z"))
        self.assertEqual(control.reported_time, "2025-03-18T17:00:00Z")
        # Healthy prompts share the incident template, so the wording does not reveal the kind.
        self.assertEqual(control.prompt.count("\n"), cases[0].prompt.count("\n"))
        self.assertEqual(evaluation.select_healthy_cases(cases, status, 0), [])

    def test_limit_applies_to_incidents_and_keeps_controls(self):
        incidents = [incident(case_id=f"i{index}") for index in range(4)]
        controls = [healthy(case_id="h0")]
        args = evaluation.parse_args(["--limit", "2", "--negatives", "1"])
        with patch.object(evaluation, "load_incident_cases", return_value=incidents), \
                patch.object(evaluation, "load_slurm_status"), \
                patch.object(evaluation, "select_healthy_cases", return_value=controls):
            cases = evaluation.build_cases(args)
        self.assertEqual([case.case_id for case in cases], ["i0", "i1", "h0"])


class ScoringTests(unittest.TestCase):
    def test_incident_onset_inside_bracket(self):
        score = evaluation.score_run(incident(), report(onset="2025-03-19T08:19:30Z"), 12.34)
        self.assertTrue(score["correct"])
        self.assertTrue(score["onset_in_bracket"])
        self.assertEqual(score["onset_error_minutes"], 0.0)
        self.assertFalse(score["onset_echoes_prompt"])
        self.assertEqual(score["wall_seconds"], 12.3)

    def test_onset_outside_bracket_reports_signed_minutes_and_offsets_are_honoured(self):
        early = evaluation.score_run(incident(), report(onset="2025-03-18T18:30:00Z"), 0)
        self.assertFalse(early["onset_in_bracket"])
        self.assertEqual(early["onset_error_minutes"], -30.0)
        # 10:20+01:00 is 09:20 UTC, ten minutes after detection.
        late = evaluation.score_run(incident(), report(onset="2025-03-19T10:20:00+01:00"), 0)
        self.assertEqual(late["onset_error_minutes"], 10.0)

    def test_echoed_prompt_time_is_flagged_and_missing_onset_fails(self):
        echoed = evaluation.score_run(incident(), report(onset="2025-03-19 09:10:00"), 0)
        self.assertTrue(echoed["onset_in_bracket"])
        self.assertTrue(echoed["onset_echoes_prompt"])
        missing = evaluation.score_run(incident(), report(onset=None), 0)
        self.assertFalse(missing["onset_in_bracket"])
        bad = evaluation.score_run(incident(), report(onset="soon"), 0)
        self.assertFalse(bad["onset_in_bracket"])
        self.assertIn("unparseable", bad["onset_parse_error"])

    def test_healthy_window_requires_unknown(self):
        self.assertTrue(evaluation.score_run(healthy(), report("unknown"), 0)["correct"])
        wrong = evaluation.score_run(healthy(), report("gpu", onset="2025-03-18T12:00:00Z"), 0)
        self.assertFalse(wrong["correct"])
        self.assertIsNone(wrong["onset_in_bracket"])
        self.assertEqual(wrong["onset_time"], "2025-03-18T12:00:00Z")

    def test_counts_query_too_broad_errors_and_stopping_reason(self):
        trace = [
            {"output": {"status": "query_too_broad"}},
            {"output": {"status": "error", "error": {"code": "outside_available_window"}}},
            {"output": {"status": "error", "error": {"code": "outside_available_window"}}},
            {"output": {"evidence_id": "series-x"}},
        ]
        score = evaluation.score_run(incident(), report(
            "unknown", trace=trace, status="incomplete", investigation_errors=[{"code": "x"}],
            stopping_reason={"code": "tool_budget_exhausted"}), 0)
        self.assertEqual((score["query_too_broad"], score["tool_errors"]), (1, 2))
        self.assertEqual(score["tool_error_codes"], {"outside_available_window": 2})
        self.assertEqual((score["status"], score["stopping_reason"]), ("incomplete", "tool_budget_exhausted"))
        self.assertEqual(score["investigation_errors"], 1)
        self.assertEqual(score["tools_call_count"], 4)

    def test_harness_failure_report_still_scores(self):
        score = evaluation.score_run(incident(), {"status": "harness_error",
                                                  "stopping_reason": {"code": "harness_error"}}, 1)
        self.assertFalse(score["correct"])
        self.assertEqual(score["stopping_reason"], "harness_error")

    def test_summary_and_markdown_table(self):
        scores = [
            evaluation.score_run(incident(case_id="a"), report(onset="2025-03-19T08:00:00Z"), 10),
            evaluation.score_run(incident(case_id="b"), report("network"), 20),
            evaluation.score_run(healthy(case_id="h"), report("unknown"), 30),
        ]
        summary = evaluation.summarize(scores)
        self.assertEqual(summary["incident_component_accuracy"], (1, 2))
        self.assertEqual(summary["incident_onset_in_bracket"], (1, 2))
        self.assertEqual(summary["healthy_correct_unknown"], (1, 1))
        self.assertEqual(summary["mean_wall_seconds"], 20.0)
        markdown = evaluation.render_markdown(scores, summary, {"live": False})
        self.assertIn("| incident component accuracy (gpu) | 1/2 (50%) |", markdown)
        self.assertIn("| healthy windows answered unknown | 1/1 (100%) |", markdown)
        self.assertEqual(sum(line.startswith("| a |") or line.startswith("| b |") or line.startswith("| h |")
                             for line in markdown.splitlines()), 3)


class VisibleWindowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / f"{NODE}_x_tidy.csv"
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["timeUtc", "value", *IDENTITY_COLUMNS])
            writer.writeheader()
            writer.writerow(dict(timeUtc="2025-03-18 10:00:00", metric="node_load1", value=1))
        labels = Path(directory.name) / "series_labels.tsv"
        labels.write_text("seriesId\tmetric\tlabels\n")
        self.source = GwdgDataSource(self.path, series_labels_path=labels,
                                     visible_window=("2025-03-18T09:00:00Z", "2025-03-18T12:00:00Z"))

    def test_reads_inside_window_succeed(self):
        rows = self.source.read_window(NODE, "2025-03-18T09:00:00Z", "2025-03-18T12:00:00Z")
        self.assertEqual(len(rows), 1)

    def test_reads_outside_window_are_rejected_for_incident_and_scheduler(self):
        for scheduler in (False, True):
            for start, end in (("2025-03-18T08:59:50Z", "2025-03-18T10:00:00Z"),
                               ("2025-03-18T10:00:00Z", "2025-03-18T12:00:10Z")):
                with self.subTest(scheduler=scheduler, start=start, end=end), \
                        self.assertRaises(ToolArgumentError) as raised:
                    self.source.read_window(NODE, start, end, scheduler=scheduler)
                self.assertEqual(raised.exception.code, "outside_available_window")
                self.assertEqual(raised.exception.details["available_end"], "2025-03-18T12:00:00+00:00")

    def test_empty_window_is_invalid(self):
        with self.assertRaises(ValueError):
            GwdgDataSource(self.path, series_labels_path=self.source.labels_path,
                           visible_window=("2025-03-18T12:00:00Z", "2025-03-18T12:00:00Z"))


class AgentRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "test-agent-dwdg.py"
        with patch("openai.OpenAI"), patch("logging.basicConfig"):
            cls.agent = runpy.run_path(str(path))

    def final_report(self):
        return {
            "severity": "unknown", "confidence": 0,
            "diagnosis": {"suspected_component": "unknown", "summary": "s", "onset_time": None,
                          "supporting_evidence_ids": [], "missing_evidence": [], "next_check": "n"},
            "next_action": "a",
        }

    def test_telemetry_file_window_client_and_model_reach_the_run(self):
        runner = self.agent["diagnose_cluster_with_fixture"]
        source, loader, factory = Mock(), Mock(), Mock()
        loader.return_value = source
        api_client = Mock()
        api_client.responses.create.return_value = SimpleNamespace(
            status="completed", output=[], output_text=json.dumps(self.final_report()))
        window = ("2025-03-18T09:00:00Z", "2025-03-19T11:00:00Z")
        with patch.dict(runner.__globals__, load_data=loader, client=None, OpenAI=factory, logger=Mock()):
            result = runner("any", "prompt", telemetry_file="file_tidy.csv.xz", visible_window=window,
                            api_client=api_client, model="test-model")
        self.assertEqual(result["status"], "completed")
        loader.assert_called_once_with("any", telemetry_file="file_tidy.csv.xz", visible_window=window)
        factory.assert_not_called()
        api_client.close.assert_not_called()
        source.clear_cache.assert_called_once_with()
        self.assertEqual(api_client.responses.create.call_args.kwargs["model"], "test-model")
        schema = self.agent["FINAL_ANSWER_FORMAT"]["schema"]["properties"]["diagnosis"]
        self.assertIn("onset_time", schema["required"])

    def test_loader_builds_a_bounded_source_for_any_file(self):
        loader = self.agent["load_data"]
        factory = Mock()
        with patch.dict(loader.__globals__, GwdgDataSource=factory):
            source = loader("ignored", telemetry_file="file_tidy.csv.xz", visible_window=("a", "b"))
        self.assertIs(source, factory.return_value)
        factory.assert_called_once_with("file_tidy.csv.xz", series_labels_path="series_labels.tsv",
                                        visible_window=("a", "b"))

    def test_onset_time_must_be_string_or_null(self):
        validate = self.agent["validate_and_enrich_report"]
        for onset in (None, "2025-03-19T08:19:30Z"):
            report = self.final_report()
            report["diagnosis"]["onset_time"] = onset
            self.assertEqual(validate(report, [], [], eval=False)["diagnosis"]["onset_time"], onset)
        report = self.final_report()
        report["diagnosis"]["onset_time"] = 5
        with self.assertRaisesRegex(ValueError, "onset_time"):
            validate(report, [], [], eval=False)


@unittest.skipUnless((DATA_DIR / "series_labels.tsv").is_file(), "GWDG series labels are not available")
class MockBatchTests(unittest.TestCase):
    """End-to-end dry runs with the scripted client against a tiny telemetry file."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rows = []
        for second in range(0, 600, 10):
            time = pd.Timestamp("2025-03-18 09:00:00") + pd.Timedelta(seconds=second)
            rows.append(dict(timeUtc=str(time), utcOffsetHours="+1", metric="node_load1", value=1.5))
            rows.append(dict(timeUtc=str(time), utcOffsetHours="+1", metric="DCGM_FI_DEV_XID_ERRORS", value=0,
                             gpu="0", uuid="gpu000000000000"))
        frame = pd.DataFrame(rows).reindex(columns=["timeUtc", "utcOffsetHours", *IDENTITY_COLUMNS[:1], "value",
                                                    *IDENTITY_COLUMNS[1:]])
        self.telemetry = self.root / f"{NODE}_2025-03-18_test_tidy.csv.xz"
        frame.to_csv(self.telemetry, index=False)
        scheduler = pd.DataFrame([dict(timeUtc="2025-03-18 09:00:00", utcOffsetHours="+1", node=NODE,
                                       metric="slurm_status_ok", value=1, job="slurm")])
        scheduler.reindex(columns=["timeUtc", "utcOffsetHours", "node", "metric", "value", *IDENTITY_COLUMNS[1:]]) \
            .to_csv(self.root / "slurm_node_state_tidy.csv.xz", index=False)
        self.case = incident(case_id="tiny", telemetry_file=str(self.telemetry),
                             window_start="2025-03-18T09:00:00Z", window_end="2025-03-18T09:10:00Z",
                             onset_bracket=("2025-03-18T09:00:00Z", "2025-03-18T09:10:00Z"))

    def options(self, **extra):
        return {"run_dir": str(self.root / "run"), "live": False, "mock_tools": True, **extra}

    def test_inline_run_executes_real_tools_and_writes_outputs(self):
        scores = evaluation.run_batch([self.case], self.options(), workers=1)
        self.assertEqual(len(scores), 1)
        score = scores[0]
        self.assertEqual((score["status"], score["suspected_component"]), ("completed", "unknown"))
        self.assertEqual(score["tool_attempts"], 2)
        record = json.loads((self.root / "run" / "cases" / "tiny.json").read_text())
        self.assertEqual([entry["tool"] for entry in record["report"]["tool_trace"]],
                         ["get_resource_ids_gwdg", "get_overview_gwdg"])
        self.assertNotEqual(record["report"]["tool_trace"][0]["output"].get("status"), "error")
        self.assertIn(NODE, record["report"]["tool_trace"][0]["output"]["node_list"])
        markdown = evaluation.write_outputs(self.root / "run", scores, {"live": False})
        self.assertIn("| tiny | incident |", markdown)
        for name in ("summary.csv", "summary.md", "summary.json"):
            self.assertTrue((self.root / "run" / name).is_file())

    def test_process_pool_runs_cases_and_keeps_case_order(self):
        second = incident(**{**vars(self.case), "case_id": "tiny-2"})
        scores = evaluation.run_batch([self.case, second], self.options(mock_tools=False), workers=2)
        self.assertEqual([score["case_id"] for score in scores], ["tiny", "tiny-2"])
        self.assertTrue(all(score["status"] == "completed" for score in scores))
        self.assertTrue((self.root / "run" / "cases" / "tiny-2.json").is_file())

    def test_cli_main_prints_the_table(self):
        args = ["--negatives", "0", "--workers", "1", "--mock-no-tools", "--out", str(self.root / "runs")]
        with patch.object(evaluation, "load_incident_cases", return_value=[self.case]), \
                patch("logging.basicConfig") as basic_config, patch("sys.stdout") as stdout, patch("sys.stderr"):
            self.assertEqual(evaluation.main(args), 0)
        basic_config.assert_called_once()
        printed = "".join(call.args[0] for call in stdout.write.call_args_list)
        self.assertIn("incident component accuracy (gpu) | 0/1 (0%)", printed)
        self.assertEqual(len(list((self.root / "runs").glob("*-mock/summary.md"))), 1)


if __name__ == "__main__":
    unittest.main()
