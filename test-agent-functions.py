from openai import OpenAI
import json
import logging
from src.tools import tools, call_function


client = OpenAI(timeout=600)

# Configure logging once before emitting log messages.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

input_list = [  {"role": "system", "content": ("You are a professional AI Cluster Engineer at diagnosing performance issues. "
                                         "You will be given a description of a performance issue and you will provide a diagnosis with given structure."),},
                {"role": "user", "content": """Investigate node-2 from 2026-09-23T10:00:00Z to 2026-09-23T10:30:00Z.
                                            Start with node metrics.
                                            Cite the evidence and distinguish a suspected cause from a confirmed cause"""},
    ]
# model-run limit and tool call limit
model_run_limit = 5
tool_call_limit = 8
model_run_cnt = 0
tool_call_cnt = 0


if __name__ == "__main__":
    while True:
        if model_run_cnt >= model_run_limit:
            logger.error("Model run limit exceeded.")
            exit(2)
        model_run_cnt += 1
        response = client.responses.create(
            model="gpt-5.6-luna",
            tools=tools,
            input=input_list,
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
            if response.status == "completed" and response.output_text.strip().strip():
                print("\n" + response.output_text)
                break
            else:
                print("\n" + "incomplete investigation, stop now")
                exit(1)

        if local_func_call_cnt + tool_call_cnt > tool_call_limit:
            logger.error("Tool call limit exceeded.")
            exit(2)
        for item in func_call_list:
            args = json.loads(item.arguments)
            logger.info("Calling tool %s, args %s", item.name, args)
            result = call_function(item.name, args, item.call_id)
            logger.info("Results of tool calling %s", result)
            
            tool_call_cnt += 1
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result),
                }
            )
