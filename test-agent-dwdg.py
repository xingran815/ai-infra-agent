import json
import logging
import sys
from pathlib import Path
from openai import OpenAI
from src.dwdg.tools_dwdg import GWDG_TOOLS as tools, call_function_dwdg as call_function
from src.dwdg.tools_dwdg import load_data_gwdg
from src.scenarios_data import load_scenario

client = OpenAI(timeout=600)

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
                        "enum": ["gpu", "network", "storage", "monitoring", "scheduler", "unknown"],
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
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(case_id: str):
    case_id = str(case_id).strip().zfill(2)
    if int(case_id) < 5:
        fixture_path = Path(__file__).resolve().parent / CASE_ID_TO_PATH[case_id]
        if not load_scenario(str(fixture_path)):
            logger.error("Failed to load scenario for fixture %s", case_id)
            sys.exit(1)
    elif int(case_id) == 5:
        return load_data_gwdg("node6a5ebc855306_2025-03-19_gpu-problem_tidy.csv.xz", series_labels_path="series_labels.tsv")
    else:
        logger.error("Unknown case_id: %s", case_id)
        sys.exit(1)

def validate_and_enrich_report(report, tool_trace, input_list, eval):
    returned_evidence_ids = {
        evidence_id
        for trace_entry in tool_trace
        for evidence_id in trace_entry["evidence_ids"]
    }
    cited_evidence_ids = set(
        report["diagnosis"]["supporting_evidence_ids"]
    )
    invented_evidence_ids = cited_evidence_ids - returned_evidence_ids

    if invented_evidence_ids:
        invented_ids = ", ".join(sorted(invented_evidence_ids))
        raise ValueError(
            f"Report cited evidence that no tool returned: {invented_ids}"
        )

    if eval:
        report["final_prompt"] = list(
            item if isinstance(item, dict) else item.model_dump(mode="json")
            for item in input_list
        )
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

def diagnose_cluster_with_fixture(case_id: str, prompt: str | None = None, eval: bool = False):
    # case_id: The ID of the test case to diagnose, "01", "02", etc.
    # exit values: 
    # 1: fixture, validation, or incomplete-investigation failures.
    # 2: model or tool limits.
    # 0: successful completion.

    func_call_output_list = []
    # model-run limit and tool call limit
    model_run_limit = 5
    tool_call_limit = 80
    model_run_cnt = 0
    tool_call_cnt = 0

    if prompt is None:
        prompt = """Investigate an issue detected at 2025-03-19 09:10:00 on the GPU cluster.
                 The time range of interest is 2025-03-18 09:00:00 to 2025-03-19 11:00:00, Europe/Berlin timezone.
                 Discover resources and then check compact health overview. Decide follow-up tools based on the evidence and your analysis.
                 Cite returned evidence IDs. Distinguish a suspected cause from a confirmed cause.
                 If the available evidence cannot distinguish the causes,
                 report unknown and identify the next useful check."""

    input_list = [  {"role": "system", "content": ("You are a professional AI Cluster Engineer at diagnosing performance issues. "
                                             "You will be given a description of a performance issue and you will provide a diagnosis with given structure."),},
                    {"role": "user", "content": prompt},
        ]

    load_data(case_id)

    while True:
        if model_run_cnt >= model_run_limit:
            logger.error("Model run limit exceeded.")
            sys.exit(2)
        model_run_cnt += 1
        response = client.responses.create(
            model="gpt-5.6-luna",
            tools=tools,
            input=input_list,
            text={"format": FINAL_ANSWER_FORMAT},
        )
        logger.debug("Model run %d completed with status %s", model_run_cnt, response.status)
        # append all model output to the conversation
        event = response.output
        input_list += event

        # put all function calls in the event into a list and count them
        local_func_call_cnt = 0
        func_call_list = []
        for item in event:
            if item.type == "function_call":
                local_func_call_cnt += 1
                func_call_list.append(item)

        if local_func_call_cnt == 0:
            if response.status == "completed" and response.output_text.strip():
                final_answer = json.loads(response.output_text)
                try:
                    final_answer = validate_and_enrich_report(
                        final_answer,
                        func_call_output_list,
                        input_list,
                        eval=eval
                    )
                except ValueError as error:
                    logger.error("Invalid final report: %s", error)
                    sys.exit(1)
                print("\n" + json.dumps(final_answer, indent=2))
                return final_answer
            else:
                print("\n" + "incomplete investigation, stop now")
                sys.exit(1)

        if local_func_call_cnt + tool_call_cnt > tool_call_limit:
            logger.error("Tool call limit exceeded.")
            sys.exit(2)
        for item in func_call_list:
            args = json.loads(item.arguments)
            logger.debug("Calling tool ## %d %s, args %s", tool_call_cnt, item.name, args)
            result = call_function(item.name, args, item.call_id)
            logger.debug("Results of tool calling %s", result)

            func_call_output_list.append(
                {
                    "tool": item.name,
                    "call_id": item.call_id,
                    "arguments": item.arguments,
                    "output": result,
                    "evidence_ids": sorted(collect_evidence_ids(result)),
                }
            )
            tool_call_cnt += 1
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result),
                }
            )

    if DEBUG:
        print("tools_trace: {}", func_call_output_list)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test-agent-dwdg.py <case_id> [prompt]")
        sys.exit(1)

    case_id = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else None

    diagnose_cluster_with_fixture(case_id, prompt, eval=False)
