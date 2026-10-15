"""Recovery boundary for model-issued, read-only telemetry tool calls."""

import json
import logging

from src.dwdg.response_limits import serialize_tool_result


logger = logging.getLogger(__name__)


class ToolArgumentError(ValueError):
    """A request the model can repair, distinct from bad data or a tool bug."""

    def __init__(self, message, *, code="invalid_arguments", details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class DataValidationError(ValueError):
    """Invalid source data; changing request syntax cannot repair it."""


class FixtureLoadError(ValueError):
    """The investigation's configured fixture could not be selected."""


def tool_error(code, message, *, retryable=False, action="use_other_evidence", details=None):
    result = {
        "status": "error", "data_returned": False,
        "error": {"code": code, "message": message[:512],
                  "retryable": retryable, "action": action},
    }
    if details:
        result["error"]["details"] = details
    return result


def _reject_constant(value):
    raise ToolArgumentError("Arguments must be standard JSON; NaN and Infinity are not supported.")


class ToolExecutor:
    """Keep failures local to a call and bound repeated failed requests per run.

    The model decides how to retry. Two failed attempts at identical arguments
    block further identical requests; a corrected request has its own budget.
    Non-retryable failures block that request immediately. User cancellation
    (KeyboardInterrupt/SystemExit) is deliberately not swallowed.
    """

    def __init__(self, max_failed_attempts=2):
        self.max_failed_attempts = max_failed_attempts
        self.failures = {}

    def _blocked(self, key):
        previous = self.failures.get(key)
        if previous and (not previous["retryable"] or previous["attempts"] >= self.max_failed_attempts):
            return tool_error(
                "retry_limit_exceeded", "This failed request is blocked for this investigation. Change the request or use other evidence.",
                action="change_request_or_use_other_evidence",
                details={"last_error_code": previous["code"], "failed_attempts": previous["attempts"]},
            )

    def execute(self, name, arguments, call_id, *, source, dispatch):
        key = (name, arguments)
        try:
            if not isinstance(arguments, str):
                raise ToolArgumentError("Tool arguments must be a JSON object encoded as a string.")
            blocked = self._blocked(key)
            if blocked:
                return blocked
            try:
                args = json.loads(arguments, parse_constant=_reject_constant)
            except json.JSONDecodeError as error:
                raise ToolArgumentError(
                    f"Invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}",
                    code="invalid_json",
                ) from error
            if not isinstance(args, dict):
                raise ToolArgumentError("Tool arguments must be a JSON object, not an array or scalar.")
            key = (name, json.dumps(args, sort_keys=True, separators=(",", ":")))
            blocked = self._blocked(key)
            if blocked:
                return blocked
            result = dispatch(name, args, call_id, source=source)
            # Serialization failures must not escape after a successful tool call.
            if not isinstance(result, dict):
                raise TypeError("Telemetry tools must return an object")
            serialize_tool_result(result)
            self.failures.pop(key, None)
            return result
        except ToolArgumentError as error:
            result = tool_error(error.code, str(error), retryable=True,
                                action="correct_arguments", details=error.details)
        except (TimeoutError, ConnectionError):
            result = tool_error("temporarily_unavailable", "The telemetry request timed out or lost its connection.",
                                retryable=True, action="retry_request")
        except (FileNotFoundError, PermissionError):
            result = tool_error("data_source_unavailable", "The selected data source is missing or inaccessible. Use other available evidence and report the gap.")
        except DataValidationError:
            logger.exception("Invalid telemetry data for tool %s, call %s", name, call_id)
            result = tool_error("invalid_source_data", "The selected telemetry data failed validation. Use other evidence and request a data-source check.")
        except Exception:
            logger.exception("Tool failure for %s, call %s", name, call_id)
            result = tool_error("internal_tool_error", "The tool failed unexpectedly. Use another tool or report the missing evidence; an operator must inspect the tool logs.")
        # Retain only failure metadata, never exception text or partial readings.
        # Malformed non-string payloads are not valid dictionary keys.
        if isinstance(arguments, str):
            previous = self.failures.get(key, {})
            attempts = previous.get("attempts", 0) + 1
            remaining = max(0, self.max_failed_attempts - attempts) if result["error"]["retryable"] else 0
            result["error"].update(failed_attempts=attempts, remaining_identical_attempts=remaining)
            if result["error"]["retryable"] and remaining == 0:
                result["error"].update(retryable=False, action="change_request_or_use_other_evidence")
            self.failures[key] = {
                "attempts": attempts,
                "code": result["error"]["code"], "retryable": result["error"]["retryable"],
            }
        return result
