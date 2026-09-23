from openai import OpenAI
import json
import src.tools_cluster as tools_cluster

client = OpenAI()

tools = tools_cluster.tools
input_list = [  {"role": "system", "content": ("You are a professional AI Cluster Engineer at diagnosing performance issues. "
                                         "You will be given a description of a performance issue and you will provide a diagnosis with given structure."),},
                {"role": "user", "content": """Use get_node_metrics to inspect node-2 from 2026-09-23T10:00:00Z to 2026-09-23T10:30:00Z"""},
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
        if item.name == "get_node_metrics":
            json_arg = json.loads(item.arguments)
            node_id = json_arg["node_id"]
            start_time = json_arg["start_time"]
            end_time = json_arg["end_time"]
            result = tools_cluster.get_node_metrics(node_id=node_id, start_time=start_time, end_time=end_time)

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