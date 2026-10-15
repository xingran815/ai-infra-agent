import contextlib
import io
import json
import logging
import os
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from src.dwdg.metric_metadata import METRIC_METADATA


KNOWN_COMPONENTS = ("gpu", "network", "storage", "monitoring", "scheduler", "node")
# One realistic series evidence ID per component, built from a representative metric.
COMPONENT_EVIDENCE = {
    component: f"series-{METRIC_METADATA[metric]['short_name']}-{component}"
    for component, metric in (
        ("gpu", "DCGM_FI_DEV_XID_ERRORS"), ("network", "node_network_up"),
        ("storage", "node_filesystem_readonly"), ("monitoring", "up"),
        ("scheduler", "slurm_status_ok"), ("node", "node_load1"),
    )
}


class ModelItem(SimpleNamespace):
    def model_dump(self, mode="json"):
        return vars(self)


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
            for citations in ([], ["series-GPUXID-observed"]):
                with self.subTest(component=component, citations=citations):
                    with self.assertRaisesRegex(ValueError, "at least one tool call"):
                        self.validate(self.report(component, citations), [])

    def test_known_components_require_citations_even_after_tool_calls(self):
        for component in KNOWN_COMPONENTS:
            for returned_ids in ([], ["series-GPUXID-observed"]):
                with self.subTest(component=component, returned_ids=returned_ids):
                    trace = [{"tool": "get_overview_gwdg", "evidence_ids": returned_ids}]
                    with self.assertRaisesRegex(ValueError, "requires supporting evidence IDs"):
                        self.validate(self.report(component), trace)

    def test_every_component_rejects_citations_not_returned_by_tools(self):
        trace = [{"tool": "get_overview_gwdg", "evidence_ids": ["series-GPUXID-observed"]}]
        for component in (*KNOWN_COMPONENTS, "unknown"):
            with self.subTest(component=component):
                report = self.report(component, ["series-GPUXID-observed", "series-invented"])
                with self.assertRaisesRegex(ValueError, "no tool returned: series-invented"):
                    self.validate(report, trace)

    def test_known_components_accept_returned_citations_and_enrich_actual_tool_usage(self):
        trace = [
            {"tool": "get_resource_ids_gwdg", "evidence_ids": []},
            {"tool": "get_overview_gwdg", "evidence_ids": list(COMPONENT_EVIDENCE.values())},
            {"tool": "get_overview_gwdg", "evidence_ids": ["series-GPUTEMP-follow-up"]},
        ]
        for component in KNOWN_COMPONENTS:
            with self.subTest(component=component):
                report = self.report(component, [COMPONENT_EVIDENCE[component], "series-GPUTEMP-follow-up"])
                report.update(tools_call_count=99, tools_used=["invented_tool"])
                result = self.validate(report, trace)
                self.assertEqual(result["tools_call_count"], 3)
                self.assertEqual(result["tools_used"], ["get_resource_ids_gwdg", "get_overview_gwdg"])

    def test_known_component_requires_evidence_from_that_component(self):
        trace = [{"tool": "get_overview_gwdg", "evidence_ids": [*COMPONENT_EVIDENCE.values(), "series-observed"]}]
        for component in KNOWN_COMPONENTS:
            others = [evidence for name, evidence in COMPONENT_EVIDENCE.items() if name != component]
            with self.subTest(component=component):
                with self.assertRaisesRegex(ValueError, f"A {component} diagnosis must cite at least one {component}"):
                    self.validate(self.report(component, others), trace)

    def test_component_mismatch_message_names_the_cited_components(self):
        trace = [{"tool": "get_overview_gwdg", "evidence_ids": [COMPONENT_EVIDENCE["scheduler"], "series-observed"]}]
        with self.assertRaisesRegex(ValueError, "map to: scheduler, unrecognized"):
            self.validate(self.report("gpu", [COMPONENT_EVIDENCE["scheduler"], "series-observed"]), trace)

    def test_unknown_component_needs_no_component_evidence(self):
        trace = [{"tool": "get_overview_gwdg", "evidence_ids": [COMPONENT_EVIDENCE["scheduler"]]}]
        result = self.validate(self.report("unknown", [COMPONENT_EVIDENCE["scheduler"]]), trace)
        self.assertEqual(result["diagnosis"]["suspected_component"], "unknown")

    def test_severity_must_be_one_of_the_schema_values(self):
        severities = self.agent["SEVERITIES"]
        schema = self.agent["FINAL_ANSWER_FORMAT"]["schema"]["properties"]["severity"]
        self.assertEqual(schema["enum"], list(severities))
        for severity in severities:
            with self.subTest(severity=severity):
                report = {**self.report("unknown"), "severity": severity}
                self.assertEqual(self.validate(report, [])["severity"], severity)
        for severity in ("High", "sev1", "", None, 3):
            with self.subTest(severity=severity), self.assertRaisesRegex(ValueError, "severity must be one of"):
                self.validate({**self.report("unknown"), "severity": severity}, [])

    def test_unknown_allows_no_evidence_with_or_without_tool_calls(self):
        for trace in ([], [{"tool": "get_resource_ids_gwdg", "evidence_ids": []}]):
            with self.subTest(trace=trace):
                result = self.validate(self.report("unknown"), trace)
                self.assertEqual(result["tools_call_count"], len(trace))
                self.assertEqual(result["diagnosis"]["supporting_evidence_ids"], [])

    def test_unknown_accepts_returned_citations(self):
        trace = [{"tool": "get_overview_gwdg", "evidence_ids": ["series-GPUXID-observed"]}]
        result = self.validate(self.report("unknown", ["series-GPUXID-observed"]), trace)
        self.assertEqual(result["diagnosis"]["supporting_evidence_ids"], ["series-GPUXID-observed"])

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
        tool = Mock(return_value={"evidence_id": "series-GPUXID-observed"})
        function_calls = [SimpleNamespace(
            type="function_call", name="get_metrics_gwdg", arguments="{}", call_id=call_id,
        ) for call_id in ("first", "second")]
        client.responses.create.side_effect = [
            SimpleNamespace(status="completed", output=[item], output_text="")
            for item in function_calls
        ] + [SimpleNamespace(
            status="completed", output=[],
            output_text=json.dumps(self.report("gpu", ["series-GPUXID-observed"])),
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
                with patch.dict(runner.__globals__, load_data=Mock(return_value=source), client=client, logger=Mock()):
                    if isinstance(error, SystemExit):
                        with self.assertRaises(SystemExit):
                            runner("05")
                    else:
                        result = runner("05")
                        self.assertEqual(result["status"], "incomplete")
                        self.assertEqual(result["stopping_reason"]["code"], "model_request_failed")
                source.clear_cache.assert_called_once_with()

    def tool_call(self, call_id, arguments="{}"):
        return ModelItem(type="function_call", name="get_metrics_gwdg", arguments=arguments, call_id=call_id)

    def response(self, *calls, report=None, text=None, status="completed"):
        return SimpleNamespace(status=status, output=list(calls),
                               output_text=text if text is not None else json.dumps(report) if report else "")

    def run_responses(self, responses, dispatch=None, **kwargs):
        runner = self.agent["diagnose_cluster_with_fixture"]
        client, source = Mock(), Mock()
        client.responses.create.side_effect = responses
        with patch.dict(runner.__globals__, load_data=Mock(return_value=source), client=client,
                        call_function=dispatch or Mock(return_value={"evidence_id": "series-GPUXID-observed"}), logger=Mock()):
            result = runner("05", **kwargs)
        source.clear_cache.assert_called_once_with()
        return result, client

    def test_malformed_call_does_not_discard_successful_sibling_and_model_can_correct(self):
        dispatch = Mock(return_value={"evidence_id": "series-GPUXID-observed"})
        result, client = self.run_responses([
            self.response(self.tool_call("bad", "{"), self.tool_call("good")),
            self.response(self.tool_call("corrected")),
            self.response(report=self.report("gpu", ["series-GPUXID-observed"])),
        ], dispatch, eval=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(dispatch.call_count, 2)
        self.assertEqual(result["tools_call_count"], 3)
        self.assertEqual(result["collected_evidence_ids"], ["series-GPUXID-observed"])
        self.assertEqual(result["tool_trace"][0]["evidence_ids"], [])
        outputs = [item for item in client.responses.create.call_args_list[1].kwargs["input"]
                   if isinstance(item, dict) and item.get("type") == "function_call_output"]
        self.assertEqual([item["call_id"] for item in outputs], ["bad", "good"])
        self.assertEqual(json.loads(outputs[0]["output"])["error"]["code"], "invalid_json")
        self.assertIn("final_prompt", result)

    def test_unsupported_metric_reaches_model_then_corrected_call_succeeds(self):
        from src.dwdg.tools_dwdg import call_function_dwdg

        def dispatch(name, args, call_id, *, source):
            if args["metrics"] == ["invented"]:
                return call_function_dwdg(name, args, call_id, source=source)
            return {"evidence_id": "series-GPUXID-observed"}

        args = dict(node_id="node1", category="gpu", metrics=["invented"],
                    start_time="2025-03-19T07:30:00Z", end_time="2025-03-19T08:00:00Z")
        corrected = {**args, "metrics": ["DCGM_FI_DEV_GPU_UTIL"]}
        result, client = self.run_responses([
            self.response(self.tool_call("unsupported", json.dumps(args))),
            self.response(self.tool_call("corrected", json.dumps(corrected))),
            self.response(report=self.report("gpu", ["series-GPUXID-observed"])),
        ], dispatch)
        self.assertEqual(result["status"], "completed")
        error = result["tool_trace"][0]["output"]["error"]
        self.assertEqual(error["code"], "unsupported_metrics")
        self.assertEqual(client.responses.create.call_count, 3)

    def test_exhausted_recovery_preserves_successful_readings_and_errors(self):
        dispatch = Mock(side_effect=[{"evidence_id": "series-GPUXID-observed", "value": 7}, TimeoutError(), TimeoutError()])
        result, client = self.run_responses([
            self.response(self.tool_call("success", '{"request":1}')),
            self.response(self.tool_call("fail-1", '{"request":2}')),
            self.response(self.tool_call("fail-2", '{"request":2}')),
            self.response(self.tool_call("blocked", '{"request":2}')),
            # The final-report turn disables tools; a call here is never executed.
            self.response(self.tool_call("ignored", '{"request":3}')),
        ], dispatch, max_model_runs=5)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["stopping_reason"]["code"], "model_budget_exhausted")
        self.assertEqual(dispatch.call_count, 3)
        self.assertEqual(client.responses.create.call_count, 5)
        self.assertEqual(result["tool_trace"][0]["output"]["value"], 7)
        self.assertEqual(result["tool_trace"][-1]["output"]["error"]["code"], "retry_limit_exceeded")
        self.assertEqual(len(result["tool_trace"]), 4)
        self.assertEqual(result["investigation_errors"][-1]["code"], "tool_call_on_final_turn")
        self.assertEqual(result["collected_evidence_ids"], ["series-GPUXID-observed"])
        self.assertEqual(result["diagnosis"]["suspected_component"], "unknown")
        self.assertEqual(result["diagnosis"]["supporting_evidence_ids"], [])

    def tool_choices(self, client):
        return [entry.kwargs["tool_choice"] for entry in client.responses.create.call_args_list]

    def final_notices(self, input_items):
        notice = self.agent["FINAL_TURN_NOTICE"]
        return [item for item in input_items if isinstance(item, dict) and item.get("content") == notice]

    def test_last_model_run_is_reserved_for_the_final_report(self):
        result, client = self.run_responses([
            self.response(self.tool_call("first")),
            self.response(report=self.report("gpu", ["series-GPUXID-observed"])),
        ], max_model_runs=2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.tool_choices(client), ["auto", "none"])
        first_input, final_input = (entry.kwargs["input"] for entry in client.responses.create.call_args_list)
        self.assertEqual(self.final_notices(first_input), [])
        self.assertEqual(len(self.final_notices(final_input)), 1)
        self.assertEqual(final_input[-1]["content"], self.agent["FINAL_TURN_NOTICE"])

    def test_single_model_run_is_a_final_report_turn(self):
        dispatch = Mock()
        result, client = self.run_responses([self.response(self.tool_call("refused"))], dispatch, max_model_runs=1)
        dispatch.assert_not_called()
        self.assertEqual(self.tool_choices(client), ["none"])
        self.assertEqual(result["stopping_reason"]["code"], "model_budget_exhausted")
        self.assertEqual(result["tool_trace"], [])

    def test_model_failure_after_success_keeps_partial_evidence(self):
        result, _ = self.run_responses([
            self.response(self.tool_call("success")), RuntimeError("API unavailable"),
        ])
        self.assertEqual(result["stopping_reason"]["code"], "model_request_failed")
        self.assertEqual(result["collected_evidence_ids"], ["series-GPUXID-observed"])
        self.assertEqual(len(result["tool_trace"]), 1)

    def test_tool_budget_keeps_completed_siblings_then_model_writes_report(self):
        dispatch = Mock(return_value={"evidence_id": "series-GPUXID-observed"})
        result, client = self.run_responses([
            self.response(self.tool_call("first"), self.tool_call("second"), self.tool_call("third")),
            self.response(report=self.report("gpu", ["series-GPUXID-observed"])),
        ], dispatch, max_tool_calls=1, eval=True)
        dispatch.assert_called_once()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.tool_choices(client), ["auto", "none"])
        self.assertEqual([entry["call_id"] for entry in result["tool_trace"]], ["first", "second", "third"])
        self.assertEqual([entry["output"].get("error", {}).get("code") for entry in result["tool_trace"]],
                         [None, "tool_budget_exhausted", "tool_budget_exhausted"])
        self.assertEqual(len([entry for entry in result["final_prompt"] if entry.get("type") == "function_call_output"]), 3)
        self.assertEqual(len(self.final_notices(result["final_prompt"])), 1)
        self.assertEqual(result["collected_evidence_ids"], ["series-GPUXID-observed"])

    def test_spending_the_whole_tool_budget_forces_the_next_turn_to_report(self):
        dispatch = Mock(return_value={"evidence_id": "series-GPUXID-observed"})
        result, client = self.run_responses([
            self.response(self.tool_call("first"), self.tool_call("second")),
            self.response(report=self.report("gpu", ["series-GPUXID-observed"])),
        ], dispatch, max_tool_calls=2)
        self.assertEqual(dispatch.call_count, 2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.tool_choices(client), ["auto", "none"])

    def test_final_report_can_be_corrected_after_tool_budget_without_reenabling_tools(self):
        result, client = self.run_responses([
            self.response(self.tool_call("first")),
            self.response(report=self.report("gpu", ["series-GPUXID-invented"])),
            self.response(report=self.report("gpu", ["series-GPUXID-observed"])),
        ], max_tool_calls=1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.tool_choices(client), ["auto", "none", "none"])
        self.assertEqual(result["investigation_errors"][0]["code"], "invalid_final_report")
        self.assertEqual(len(self.final_notices(client.responses.create.call_args_list[2].kwargs["input"])), 1)

    def test_invalid_report_after_tool_budget_reports_tool_budget_stop(self):
        result, _ = self.run_responses([
            self.response(self.tool_call("first")),
            self.response(text="{}"),
        ], max_model_runs=2, max_tool_calls=1)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["stopping_reason"]["code"], "tool_budget_exhausted")
        self.assertEqual(result["collected_evidence_ids"], ["series-GPUXID-observed"])

    def test_invalid_final_reports_can_be_corrected(self):
        for invalid in ('{', 'null', '{}', json.dumps(self.report("gpu", ["invented"]))):
            with self.subTest(invalid=invalid):
                result, client = self.run_responses([
                    self.response(text=invalid), self.response(report=self.report("unknown")),
                ])
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["investigation_errors"][0]["code"], "invalid_final_report")
                self.assertIn("Correct the final report", client.responses.create.call_args_list[1].kwargs["input"][-1]["content"])

    def test_invalid_final_report_at_last_round_returns_incomplete(self):
        result, _ = self.run_responses([self.response(text='{}')], max_model_runs=1)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["investigation_errors"][0]["code"], "invalid_final_report")

    def test_unfinished_model_response_never_executes_partial_calls(self):
        dispatch = Mock()
        result, _ = self.run_responses([
            self.response(self.tool_call("partial"), status="incomplete"),
        ], dispatch)
        dispatch.assert_not_called()
        self.assertEqual(result["stopping_reason"]["code"], "model_response_incomplete")

    def test_error_output_never_adds_citable_evidence(self):
        result, _ = self.run_responses([
            self.response(self.tool_call("error")), self.response(report=self.report("unknown")),
        ], Mock(return_value={"status": "error", "evidence_id": "not-evidence"}))
        self.assertEqual(result["collected_evidence_ids"], [])

    def test_fixture_and_client_initialization_failures_return_statuses(self):
        runner = self.agent["diagnose_cluster_with_fixture"]
        for loader, factory, code in (
            (Mock(side_effect=FileNotFoundError()), Mock(), "fixture_unavailable"),
            (Mock(return_value=Mock()), Mock(side_effect=RuntimeError("no API key")), "model_unavailable"),
        ):
            with self.subTest(code=code), patch.dict(runner.__globals__, load_data=loader,
                                                   OpenAI=factory, client=None, logger=Mock()):
                result = runner("05")
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["stopping_reason"]["code"], code)

    def test_invalid_case_id_raises_typed_error_not_system_exit(self):
        for case_id in ("00", "bad", "99"):
            with self.subTest(case_id=case_id), self.assertRaises(self.agent["FixtureLoadError"]):
                self.agent["load_data"](case_id)

    def test_synthetic_cases_are_routed_to_their_own_runner(self):
        self.assertEqual(set(self.agent["CASE_ID_TO_PATH"]), {"05"})
        runner = self.agent["diagnose_cluster_with_fixture"]
        for case_id in ("01", "2", " 03 ", "04"):
            with self.subTest(case_id=case_id):
                with self.assertRaisesRegex(self.agent["FixtureLoadError"], "test-agent-functions.py"):
                    self.agent["load_data"](case_id)
                client = Mock()
                with patch.dict(runner.__globals__, client=client):
                    result = runner(case_id)
                client.responses.create.assert_not_called()
                self.assertEqual(result["stopping_reason"]["code"], "fixture_unavailable")
                self.assertIn("test-agent-functions.py", result["stopping_reason"]["message"])

    def test_import_leaves_logging_unconfigured(self):
        path = Path(__file__).resolve().parents[1] / "test-agent-dwdg.py"
        with patch("openai.OpenAI"), patch("logging.basicConfig") as basic_config:
            runpy.run_path(str(path))
        basic_config.assert_not_called()

    def test_log_level_defaults_to_info_and_reads_environment(self):
        configure = self.agent["configure_logging"]
        for value, expected in ((None, logging.INFO), ("debug", logging.DEBUG),
                                (" WARNING ", logging.WARNING), ("verbose", logging.INFO)):
            environment = {} if value is None else {"AGENT_LOG_LEVEL": value}
            with self.subTest(value=value), patch.dict(os.environ, environment, clear=True), \
                    patch("logging.basicConfig") as basic_config, \
                    patch.dict(configure.__globals__, logger=Mock()) as namespace:
                configure()
                self.assertEqual(basic_config.call_args.kwargs["level"], expected)
                if value == "verbose":
                    namespace["logger"].warning.assert_called_once()

    def test_cleanup_failure_does_not_replace_report(self):
        runner = self.agent["diagnose_cluster_with_fixture"]
        source = Mock()
        source.clear_cache.side_effect = RuntimeError("cleanup")
        client = Mock()
        client.responses.create.return_value = self.response(report=self.report("unknown"))
        with patch.dict(runner.__globals__, load_data=Mock(return_value=source), client=client, logger=Mock()):
            self.assertEqual(runner("05")["status"], "completed")

    def test_runner_closes_only_the_client_it_creates(self):
        runner = self.agent["diagnose_cluster_with_fixture"]
        for injected in (False, True):
            source, api_client = Mock(), Mock()
            api_client.responses.create.return_value = self.response(report=self.report("unknown"))
            factory = Mock(return_value=api_client)
            with self.subTest(injected=injected), patch.dict(
                runner.__globals__, load_data=Mock(return_value=source),
                client=api_client if injected else None, OpenAI=factory,
            ):
                self.assertEqual(runner("05")["status"], "completed")
                if injected:
                    factory.assert_not_called()
                    api_client.close.assert_not_called()
                else:
                    factory.assert_called_once()
                    api_client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
