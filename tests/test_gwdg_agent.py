import contextlib
import io
import json
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch


KNOWN_COMPONENTS = ("gpu", "network", "storage", "monitoring", "scheduler", "node")


class ReportValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the CLI's functions without credentials, network calls, or logging changes.
        path = Path(__file__).resolve().parents[1] / "test-agent-dwdg.py"
        with patch("openai.OpenAI"), patch("logging.basicConfig"):
            cls.agent = runpy.run_path(str(path))

    def report(self, component, evidence_ids=()):
        return {
            "severity": "high",
            "confidence": 0.99,
            "diagnosis": {
                "suspected_component": component,
                "summary": "Suspected fault based on available telemetry.",
                "supporting_evidence_ids": list(evidence_ids),
                "missing_evidence": ["Kernel logs"],
                "next_check": "Inspect kernel logs.",
            },
            "next_action": "Continue investigation.",
        }

    def validate(self, report, trace):
        return self.agent["validate_and_enrich_report"](report, trace, [], eval=False)

    def test_known_components_require_tool_calls_even_with_claimed_citations(self):
        for component in KNOWN_COMPONENTS:
            for citations in ([], ["series-observed"]):
                with self.subTest(component=component, citations=citations):
                    with self.assertRaisesRegex(ValueError, "at least one tool call"):
                        self.validate(self.report(component, citations), [])

    def test_known_components_require_citations_even_after_tool_calls(self):
        for component in KNOWN_COMPONENTS:
            for returned_ids in ([], ["series-observed"]):
                with self.subTest(component=component, returned_ids=returned_ids):
                    trace = [{"tool": "get_overview_gwdg", "evidence_ids": returned_ids}]
                    with self.assertRaisesRegex(ValueError, "requires supporting evidence IDs"):
                        self.validate(self.report(component), trace)

    def test_every_component_rejects_citations_not_returned_by_tools(self):
        trace = [{"tool": "get_overview_gwdg", "evidence_ids": ["series-observed"]}]
        for component in (*KNOWN_COMPONENTS, "unknown"):
            with self.subTest(component=component):
                report = self.report(component, ["series-observed", "series-invented"])
                with self.assertRaisesRegex(ValueError, "no tool returned: series-invented"):
                    self.validate(report, trace)

    def test_known_components_accept_returned_citations_and_enrich_actual_tool_usage(self):
        trace = [
            {"tool": "get_resource_ids_gwdg", "evidence_ids": []},
            {"tool": "get_overview_gwdg", "evidence_ids": ["series-observed"]},
            {"tool": "get_overview_gwdg", "evidence_ids": ["series-follow-up"]},
        ]
        for component in KNOWN_COMPONENTS:
            with self.subTest(component=component):
                report = self.report(component, ["series-observed", "series-follow-up"])
                report.update(tools_call_count=99, tools_used=["invented_tool"])
                result = self.validate(report, trace)
                self.assertEqual(result["tools_call_count"], 3)
                self.assertEqual(result["tools_used"], ["get_resource_ids_gwdg", "get_overview_gwdg"])

    def test_unknown_allows_no_evidence_with_or_without_tool_calls(self):
        for trace in ([], [{"tool": "get_resource_ids_gwdg", "evidence_ids": []}]):
            with self.subTest(trace=trace):
                result = self.validate(self.report("unknown"), trace)
                self.assertEqual(result["tools_call_count"], len(trace))
                self.assertEqual(result["diagnosis"]["supporting_evidence_ids"], [])

    def test_unknown_accepts_returned_citations(self):
        trace = [{"tool": "get_overview_gwdg", "evidence_ids": ["series-observed"]}]
        result = self.validate(self.report("unknown", ["series-observed"]), trace)
        self.assertEqual(result["diagnosis"]["supporting_evidence_ids"], ["series-observed"])

    def test_unknown_cannot_cite_evidence_without_tool_calls(self):
        with self.assertRaisesRegex(ValueError, "no tool returned: series-invented"):
            self.validate(self.report("unknown", ["series-invented"]), [])

    def test_gwdg_loader_constructs_an_investigation_source(self):
        loader = self.agent["load_data"]
        factory = Mock()
        with patch.dict(loader.__globals__, GwdgDataSource=factory):
            source = loader("5")
        self.assertIs(source, factory.return_value)
        factory.assert_called_once_with(
            self.agent["CASE_ID_TO_PATH"]["05"], series_labels_path="series_labels.tsv",
        )

    def test_runner_passes_its_source_to_every_tool_and_clears_cache(self):
        runner = self.agent["diagnose_cluster_with_fixture"]
        source = Mock()
        client = Mock()
        tool = Mock(return_value={"evidence_id": "series-observed"})
        function_calls = [SimpleNamespace(
            type="function_call", name="get_metrics_gwdg", arguments="{}", call_id=call_id,
        ) for call_id in ("first", "second")]
        client.responses.create.side_effect = [
            SimpleNamespace(status="completed", output=[item], output_text="")
            for item in function_calls
        ] + [SimpleNamespace(
            status="completed", output=[],
            output_text=json.dumps(self.report("gpu", ["series-observed"])),
        )]
        with patch.dict(runner.__globals__, load_data=Mock(return_value=source),
                        client=client, call_function=tool), contextlib.redirect_stdout(io.StringIO()):
            result = runner("05")
        self.assertEqual(result["tools_call_count"], 2)
        self.assertEqual(tool.call_args_list, [
            call("get_metrics_gwdg", {}, "first", source=source),
            call("get_metrics_gwdg", {}, "second", source=source),
        ])
        source.clear_cache.assert_called_once_with()

    def test_runner_clears_source_cache_on_failure(self):
        runner = self.agent["diagnose_cluster_with_fixture"]
        for error in (RuntimeError("request failed"), SystemExit(2)):
            with self.subTest(error=type(error).__name__):
                source = Mock()
                client = Mock()
                client.responses.create.side_effect = error
                with patch.dict(runner.__globals__, load_data=Mock(return_value=source), client=client):
                    with self.assertRaises(type(error)):
                        runner("05")
                source.clear_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
