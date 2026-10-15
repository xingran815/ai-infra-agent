import json
import unittest
from unittest.mock import Mock, patch

from src.dwdg.execution import DataValidationError, ToolArgumentError, ToolExecutor
from src.dwdg.tools_dwdg import call_function_dwdg


class ToolExecutionTests(unittest.TestCase):
    def setUp(self):
        self.executor = ToolExecutor()
        self.source = Mock()

    def execute(self, arguments, dispatch=call_function_dwdg, name="get_metrics_gwdg"):
        return self.executor.execute(name, arguments, "call-1", source=self.source, dispatch=dispatch)

    def test_unsupported_metric_returns_choices_without_reading_data(self):
        result = self.execute(json.dumps({
            "node_id": "node1", "category": "gpu", "metrics": ["invented_metric"],
            "start_time": "2025-03-19T07:30:00Z", "end_time": "2025-03-19T08:00:00Z",
        }))
        self.assertEqual(result["error"]["code"], "unsupported_metrics")
        self.assertTrue(result["error"]["retryable"])
        self.assertIn("DCGM_FI_DEV_GPU_UTIL", result["error"]["details"]["available_metrics"])
        self.source.read_window.assert_not_called()

    def test_invalid_argument_shapes_and_signature_errors_are_recoverable(self):
        for arguments in ('{', '[]', 'null', '1', '"string"', '{"node_id": NaN}', '{}',
                          '{"source": "other"}', '{"unexpected": true}'):
            with self.subTest(arguments=arguments):
                result = self.execute(arguments)
                self.assertTrue(result["error"]["retryable"])
                self.assertEqual(result["error"]["action"], "correct_arguments")
        self.source.read_window.assert_not_called()

    def test_unknown_tool_returns_allowlist(self):
        result = self.execute('{}', name="not_a_tool")
        self.assertEqual(result["error"]["code"], "unknown_tool")
        self.assertIn("get_metrics_gwdg", result["error"]["details"]["available_tools"])

    def test_error_classes_and_nonretryable_errors_do_not_leak_exception_text(self):
        cases = [
            (TimeoutError("private path"), "temporarily_unavailable", True),
            (ConnectionError("private path"), "temporarily_unavailable", True),
            (FileNotFoundError("private path"), "data_source_unavailable", False),
            (PermissionError("private path"), "data_source_unavailable", False),
            (DataValidationError("private path"), "invalid_source_data", False),
            (ValueError("private path"), "internal_tool_error", False),
            (TypeError("private path"), "internal_tool_error", False),
            (RuntimeError("private path"), "internal_tool_error", False),
        ]
        for index, (error, code, retryable) in enumerate(cases):
            with self.subTest(code=code), patch("src.dwdg.execution.logger"):
                result = self.execute(json.dumps({"request": index}), dispatch=Mock(side_effect=error))
                self.assertEqual(result["error"]["code"], code)
                self.assertEqual(result["error"]["retryable"], retryable)
                self.assertFalse(result["data_returned"])
                self.assertNotIn("private path", json.dumps(result))

    def test_bad_tool_output_becomes_internal_error(self):
        for index, value in enumerate((None, {"value": float("nan")}, {"value": object()})):
            with self.subTest(value=value), patch("src.dwdg.execution.logger"):
                result = self.execute(json.dumps({"request": index}), dispatch=Mock(return_value=value))
                self.assertEqual(result["error"]["code"], "internal_tool_error")

    def test_identical_failures_are_bounded_but_corrected_request_runs(self):
        dispatch = Mock(side_effect=TimeoutError())
        for index, arguments in enumerate(('{"a": 1, "b": 2}', '{"b":2,"a":1}')):
            error = self.execute(arguments, dispatch)["error"]
            self.assertEqual(error["code"], "temporarily_unavailable")
            self.assertEqual(error["remaining_identical_attempts"], 1 - index)
            self.assertEqual(error["retryable"], index == 0)
        blocked = self.execute('{"a":1,"b":2}', dispatch)
        self.assertEqual(blocked["error"]["code"], "retry_limit_exceeded")
        self.assertEqual(dispatch.call_count, 2)
        dispatch.side_effect = None
        dispatch.return_value = {"nodes": {}}
        self.assertEqual(self.execute('{"a":2,"b":2}', dispatch), {"nodes": {}})
        self.assertEqual(dispatch.call_count, 3)

    def test_malformed_json_retries_are_also_bounded(self):
        dispatch = Mock()
        for _ in range(2):
            self.assertEqual(self.execute('{', dispatch)["error"]["code"], "invalid_json")
        self.assertEqual(self.execute('{', dispatch)["error"]["code"], "retry_limit_exceeded")
        dispatch.assert_not_called()

    def test_nonretryable_error_blocks_immediate_repeat(self):
        dispatch = Mock(side_effect=FileNotFoundError())
        self.execute('{}', dispatch)
        self.assertEqual(self.execute('{}', dispatch)["error"]["code"], "retry_limit_exceeded")
        dispatch.assert_called_once()

    def test_success_after_transient_error_preserves_result_and_resets_failure_count(self):
        reading = {"evidence_id": "series-1", "value": 1}
        dispatch = Mock(side_effect=[TimeoutError(), reading, TimeoutError(), reading])
        for _ in range(2):
            self.assertEqual(self.execute('{}', dispatch)["status"], "error")
            self.assertIs(self.execute('{}', dispatch), reading)

    def test_process_control_is_not_swallowed(self):
        for error in (KeyboardInterrupt(), SystemExit(2)):
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                self.execute('{}', Mock(side_effect=error))


if __name__ == "__main__":
    unittest.main()
