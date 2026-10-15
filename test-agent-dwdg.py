import json
import logging
import math
import sys
from pathlib import Path
from openai import OpenAI
from src.dwdg.tools_dwdg import GWDG_TOOLS as tools, call_function_dwdg as call_function
from src.dwdg.response_limits import serialize_tool_result
from src.dwdg.tools_dwdg import GwdgDataSource
from src.dwdg.execution import FixtureLoadError, ToolExecutor, tool_error
from src.scenarios_data import load_scenario

# Create lazily inside the investigation so missing credentials become a result,
# and importing the runner never requires an API key.
client = None

DEBUG = False

FINAL_ANSWER_FORMAT = {
    "type": "json_schema",
    "name": "cluster_diagnosis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "severity": {
                "type": "string",
                "description": "Severity of the diagnosed cluster problem.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the diagnosis, from 0 to 1.",
                "minimum": 0,
                "maximum": 1,
            },
            "diagnosis": {
                "type": "object",
                "properties": {
                    "suspected_component": {
                        "type": "string",
                        "enum": ["gpu", "network", "storage", "monitoring", "scheduler", "node", "unknown"],
                    },
                    "summary": {"type": "string"},
                    "supporting_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "what observations support your diagnosis"
                    },
                    "missing_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "next_check": {"type": "string"},
                },
                "required": [
                    "suspected_component",
                    "summary",
                    "supporting_evidence_ids",
                    "missing_evidence",
                    "next_check",
                ],
                "additionalProperties": False,
            },
            "next_action": {"type": "string"},
        },
        "required": ["severity", "confidence", "diagnosis", "next_action"],
        "additionalProperties": False,
    },
}

CASE_ID_TO_PATH = {
    "01": "scenarios/case_01.json",
    "02": "scenarios/case_02.json",
    "03": "scenarios/case_03.json",
    "04": "scenarios/case_04.json",
    "05": "node6a5ebc855306_2025-03-19_gpu-problem_tidy.csv.xz",
}

# Configure logging once before emitting log messages.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(case_id: str):
    case_id = str(case_id).strip().zfill(2)
    if case_id not in CASE_ID_TO_PATH:
        raise FixtureLoadError(f"Unknown case_id: {case_id}")
    if int(case_id) < 5:
        fixture_path = Path(__file__).resolve().parent / CASE_ID_TO_PATH[case_id]
        if not load_scenario(str(fixture_path)):
            raise FixtureLoadError(f"Failed to load scenario for fixture {case_id}")
    elif int(case_id) == 5:
        return GwdgDataSource(CASE_ID_TO_PATH[case_id], series_labels_path="series_labels.tsv")

def validate_and_enrich_report(report, tool_trace, input_list, eval):
    """Require investigation evidence for known components and verify every citation."""
    if not isinstance(report, dict) or not isinstance(report.get("diagnosis"), dict):
        raise ValueError("Report must be an object containing a diagnosis object")
    diagnosis = report["diagnosis"]
    if (not isinstance(report.get("severity"), str)
            or not isinstance(report.get("next_action"), str)
            or type(report.get("confidence")) not in (int, float)
            or not math.isfinite(report["confidence"]) or not 0 <= report["confidence"] <= 1):
        raise ValueError("Report requires severity, next_action, and finite confidence from 0 to 1")
    if diagnosis.get("suspected_component") not in ("gpu", "network", "storage", "monitoring", "scheduler", "node", "unknown"):
        raise ValueError("Invalid suspected_component")
    for key in ("supporting_evidence_ids", "missing_evidence"):
        if not isinstance(diagnosis.get(key), list) or not all(isinstance(value, str) for value in diagnosis[key]):
            raise ValueError(f"diagnosis.{key} must be an array of strings")
    if not all(isinstance(diagnosis.get(key), str) for key in ("summary", "next_check")):
        raise ValueError("diagnosis requires summary and next_check strings")
    cited_evidence_ids = set(diagnosis["supporting_evidence_ids"])
    if diagnosis["suspected_component"] != "unknown":
        if not tool_trace:
            raise ValueError("A known suspected component requires at least one tool call")
        if not cited_evidence_ids:
            raise ValueError("A known suspected component requires supporting evidence IDs")

    returned_evidence_ids = {
        evidence_id
        for trace_entry in tool_trace
        for evidence_id in trace_entry["evidence_ids"]
    }
    invented_evidence_ids = cited_evidence_ids - returned_evidence_ids

    if invented_evidence_ids:
        invented_ids = ", ".join(sorted(invented_evidence_ids))
        raise ValueError(
            f"Report cited evidence that no tool returned: {invented_ids}"
        )

    if eval:
        report["final_prompt"] = [
            item if isinstance(item, dict) else item.model_dump(mode="json")
            for item in input_list
        ]
    report["tools_used"] = list(
        dict.fromkeys(trace_entry["tool"] for trace_entry in tool_trace)
    )
    report["tools_call_count"] = len(tool_trace)
    return report

def collect_evidence_ids(value):
    ids = set()

    if isinstance(value, dict):
        evidence_id = value.get("evidence_id")
        if isinstance(evidence_id, str):
            ids.add(evidence_id)

        evidence_ids = value.get("evidence_ids", [])
        if isinstance(evidence_ids, list):
            ids.update(item for item in evidence_ids if isinstance(item, str))

        for child in value.values():
            ids.update(collect_evidence_ids(child))

    elif isinstance(value, list):
        for child in value:
            ids.update(collect_evidence_ids(child))

    return ids


def call_api(prompt: str, options: dict, context): 
    # adapter for promptfoo to call diagnose_cluster_with_fixture
    config = options.get("config", {})
    case_id = config.get("case_id", "01")
    eval = config.get("eval", True)

    report = diagnose_cluster_with_fixture(case_id, prompt, eval)
    final_prompt = report.pop("final_prompt", [])
    
    return {'output': json.dumps(report), 
            'metadata': {'final_prompt': final_prompt},
           }

def diagnose_cluster_with_fixture(case_id: str, prompt: str | None = None, eval: bool = False,
                                  *, max_model_runs: int = 5, max_tool_calls: int = 80):
    """Run a bounded investigation; ordinary failures return partial results."""
    tool_trace = []
    model_run_count = 0
    tool_attempts = 0
    executor = ToolExecutor()
    source = None
    api_client = None
    owns_client = False
    input_list = []
    investigation_errors = []

    def finish(report, status, stopping_reason=None):
        report = validate_and_enrich_report(report, tool_trace, input_list, eval=eval)
        report.update(status=status, tool_trace=list(tool_trace),
                      collected_evidence_ids=sorted({eid for entry in tool_trace for eid in entry["evidence_ids"]}),
                      model_runs=model_run_count, tool_attempts=tool_attempts,
                      investigation_errors=list(investigation_errors))
        if stopping_reason:
            report["stopping_reason"] = stopping_reason
        return report

    def incomplete(code, message):
        # Preserve observations separately; a host-generated fallback must not
        # turn an unfinished investigation into a fabricated causal diagnosis.
        return finish({
            "severity": "unknown", "confidence": 0,
            "diagnosis": {
                "suspected_component": "unknown", "summary": "Investigation incomplete: " + message,
                "supporting_evidence_ids": [], "missing_evidence": [message],
                "next_check": "Review the retained tool results and errors, then continue the investigation.",
            },
            "next_action": "Resolve the stopping reason and continue using the retained evidence.",
        }, "incomplete", {"code": code, "message": message})

    if prompt is None:
        prompt = """Investigate an issue detected at 2025-03-19 09:10:00 on the GPU cluster.
                 The time range of interest is 2025-03-18 09:00:00 to 2025-03-19 11:00:00, all time in UTC.
                 Start with resource discovery and compact health overview check.
                 Decide follow-up tools based on the evidence and your analysis.
                 Check what happened before the issue occurred, try to find the root cause of the performance issue.
                 Cite returned evidence IDs. Distinguish a suspected cause from a confirmed cause.
                 If the available evidence cannot distinguish the causes,
                 report unknown and identify the next useful check.
                 """

    input_list = [
        {"role": "system", "content": (
            "You are a professional AI Cluster Engineer diagnosing performance issues. "
            "Provide a diagnosis using the given structure. Tool outputs may contain status=error "
            "with error.code, retryable and action. Correct invalid arguments using the supplied hints; "
            "retry temporary failures only within the stated budget. Identical failed requests allow "
            "at most two attempts. Non-retryable errors require other evidence or a missing-evidence report. "
            "query_too_broad requires narrower selectors. Errors and withheld readings are not evidence "
            "of health and cannot support a diagnosis. Preserve successful evidence across failures. "
            "If recovery is impossible, report unknown or an evidence-supported suspicion and disclose "
            "the gaps. Never invent evidence IDs."
        )},
        {"role": "user", "content": prompt},
    ]
    try:
        if type(max_model_runs) is not int or max_model_runs < 1 or type(max_tool_calls) is not int or max_tool_calls < 1:
            return incomplete("invalid_limits", "Model and tool limits must be positive integers.")
        try:
            source = load_data(case_id)
        except Exception:
            logger.exception("Failed to load investigation fixture")
            return incomplete("fixture_unavailable", "The investigation fixture could not be loaded; check the case ID and data source.")
        try:
            owns_client = client is None
            api_client = client if client is not None else OpenAI(timeout=600)
        except Exception:
            logger.exception("Failed to initialize model client")
            return incomplete("model_unavailable", "The model client could not be initialized; check its configuration.")

        while model_run_count < max_model_runs:
            model_run_count += 1
            try:
                response = api_client.responses.create(
                    model="gpt-5.6-luna", tools=tools, input=list(input_list),
                    text={"format": FINAL_ANSWER_FORMAT},
                )
            except Exception:
                logger.exception("Model request failed")
                return incomplete("model_request_failed", "The model request failed after the client's retry policy; retained evidence is available.")
            logger.debug("Model run %d completed with status %s", model_run_count, response.status)
            if response.status != "completed":
                return incomplete("model_response_incomplete", "The model did not complete its response.")
            input_list.extend(response.output)
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                if not response.output_text.strip():
                    return incomplete("missing_final_report", "The model returned no tool calls or final report.")
                try:
                    report = json.loads(response.output_text)
                    return finish(report, "completed")
                except (ValueError, TypeError, KeyError) as error:
                    # Let the model repair malformed JSON, invalid shape or
                    # unsupported citations using its remaining model budget.
                    failure = {"code": "invalid_final_report", "message": str(error)[:512]}
                    investigation_errors.append(failure)
                    input_list.append({"role": "user", "content": serialize_tool_result({
                        "status": "error", "error": failure,
                        "action": "Correct the final report. Cite only evidence IDs from successful tool outputs; use unknown when unsupported.",
                    })})
                    continue

            budget_exhausted = False
            for item in calls:
                if tool_attempts >= max_tool_calls:
                    budget_exhausted = True
                    result = tool_error("tool_budget_exhausted", "No tool execution budget remains for this investigation.",
                                        action="report_with_available_evidence")
                else:
                    tool_attempts += 1
                    result = executor.execute(item.name, item.arguments, item.call_id,
                                              source=source, dispatch=call_function)
                logger.debug("Results of tool calling %s", result)
                # Error messages may mention arbitrary IDs. Only successful
                # readings are allowed into the citation ledger.
                evidence_ids = [] if result.get("status") in ("error", "query_too_broad") else sorted(collect_evidence_ids(result))
                tool_trace.append({
                    "tool": item.name, "call_id": item.call_id,
                    "arguments": item.arguments, "output": result, "evidence_ids": evidence_ids,
                })
                input_list.append({"type": "function_call_output", "call_id": item.call_id,
                                   "output": serialize_tool_result(result)})
            if budget_exhausted:
                return incomplete("tool_budget_exhausted", "The tool-call budget was exhausted.")
        return incomplete("model_budget_exhausted", "The model-run budget was exhausted before a valid final report.")
    except Exception:
        logger.exception("Unexpected investigation failure")
        return incomplete("internal_investigation_error", "The investigation encountered an internal error; inspect the logs and retained evidence.")
    finally:
        if source is not None:
            try:
                source.clear_cache()
            except Exception:
                logger.exception("Failed to clear investigation cache")
        if owns_client and api_client is not None:
            try:
                api_client.close()
            except Exception:
                logger.exception("Failed to close model client")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test-agent-dwdg.py <case_id> [prompt]")
        sys.exit(1)

    case_id = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else None

    print(json.dumps(diagnose_cluster_with_fixture(case_id, prompt, eval=False), indent=2))
