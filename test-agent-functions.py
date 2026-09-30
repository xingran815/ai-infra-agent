import json
import logging
from openai import OpenAI
from src.tools import tools, call_function
from src.scenarios_data import load_scenario
import argparse


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
                        "enum": ["gpu", "network", "storage", "unknown"],
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


# Configure logging once before emitting log messages.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

func_call_output_list = []
input_list = [  {"role": "system", "content": ("You are a professional AI Cluster Engineer at diagnosing performance issues. "
                                         "You will be given a description of a performance issue and you will provide a diagnosis with given structure."),},
                {"role": "user", "content": """Investigate node-2 from 2026-09-23T10:00:00Z to 2026-09-23T10:30:00Z.
                                            Investigate the specified node and time window.
                                            Start with node metrics. Choose follow-up tools based on the evidence.
                                            Cite observation IDs. Distinguish a suspected cause from a confirmed cause.
                                            If the available evidence cannot distinguish the causes,
                                            report unknown and identify the next useful check."""},
    ]
# model-run limit and tool call limit
model_run_limit = 5
tool_call_limit = 80
model_run_cnt = 0
tool_call_cnt = 0

def parse_args():
    parser = argparse.ArgumentParser(
        description="Investigate an AI cluster fixture."
    )
    parser.add_argument(
        "fixture",
        help="case number of the fixture"
    )
    return parser.parse_args()


def validate_and_enrich_report(report, tool_trace):
    returned_evidence_ids = {
        evidence_id
        for trace_entry in tool_trace
        for evidence_id in trace_entry["observation_id"]
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

    report["tools_used"] = list(
        dict.fromkeys(trace_entry["tool"] for trace_entry in tool_trace)
    )
    return report


if __name__ == "__main__":
    args = parse_args()
    load_scenario("scenarios/case_" + str(args.fixture) + ".json")

    while True:
        if model_run_cnt >= model_run_limit:
            logger.error("Model run limit exceeded.")
            exit(2)
        model_run_cnt += 1
        response = client.responses.create(
            model="gpt-5.6-luna",
            tools=tools,
            input=input_list,
            text={"format": FINAL_ANSWER_FORMAT},
        )
        logger.info("Model run %d completed with status %s", model_run_cnt, response.status)
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
                    )
                except ValueError as error:
                    logger.error("Invalid final report: %s", error)
                    exit(1)
                print("\n" + json.dumps(final_answer, indent=2))
                break
            else:
                print("\n" + "incomplete investigation, stop now")
                exit(1)

        if local_func_call_cnt + tool_call_cnt > tool_call_limit:
            logger.error("Tool call limit exceeded.")
            exit(2)
        for item in func_call_list:
            args = json.loads(item.arguments)
            logger.info("Calling tool ## %d %s, args %s", tool_call_cnt, item.name, args)
            result = call_function(item.name, args, item.call_id)
            logger.info("Results of tool calling %s", result)

            func_call_output_list.append(
                {
                    "tool": item.name,
                    "call_id": item.call_id,
                    "arguments": item.arguments,
                    "output": result,
                    "observation_id": [
                        obs["evidence_id"]
                        for obs in (
                            result.get("observations", [])
                            + result.get("historical_events", [])
                        )
                    ],
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
