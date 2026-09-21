from openai import OpenAI
from pydantic import BaseModel
import json
import src.tools_prometheus as tools_prom

client = OpenAI()

tools = tools_prom.tools
input_list = [  {"role": "system", "content": ("You are a professional AI Cluster Engineer at diagnosing performance issues. "
                                         "You will be given a description of a performance issue and you will provide a diagnosis with given structure."),},
                {"role": "user", "content": """the GPU utilization on node-1 is consistently below 50% during training, while other nodes are at 90% utilization. The training job is running on 4 nodes with 8 GPUs each. The network is InfiniBand."""},
    ]

response = client.responses.create(
    model="gpt-5.6-luna",
    tools=tools,
    input=input_list,
)

event = response.output
input_list += response.output

for item in event:
    if item.type == "function_call":
        if item.name == "query_prometheus":
            json_arg = json.loads(item.arguments)
            node = json_arg["node"]
            metric = json_arg["metric"]
            result = tools_prom.query_prometheus(node=node, metric=metric)

            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result),
                }
            )
            
print("Final Input List:")
print(input_list)


response = client.responses.create(
    model="gpt-5.6-luna",
    tools=tools,
    input=input_list,
)

print("Final output:")
print(response.model_dump_json(indent=2))
print("\n" + response.output_text)