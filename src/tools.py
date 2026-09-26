from src.tools_cluster import get_node_metrics
from src.tools_gpu import get_gpu_health
from src.tools_network import get_network_health
from src.tools_storage import get_storage_health

tools = [
    {
        "type": "function",
        "name": "get_storage_health",
        "description": "Read storage health metrics for a specific storage resource.",
        "parameters": {
            "type": "object",
            "properties": {
                "storage_id": {
                    "type": "string",
                    "description": "The storage resource ID to query, e.g. 'storage-2'.",
                },
            },
            "required": ["storage_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_gpu_health",
        "description": "Read GPU health metrics for a GPU on a specific node.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The node that owns the GPU, e.g. 'node-2'.",
                },
                "gpu_id": {
                    "type": "string",
                    "description": "The GPU ID returned by node metrics, e.g. 'node-2-gpu-0'.",
                },
            },
            "required": ["node_id", "gpu_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_network_health",
        "description": "Query network health metrics for a specific interface.",
        "parameters": {
            "type": "object",
            "properties": {
                "interface_id": {
                    "type": "string",
                    "description": "reading a synthetic network snapshot for an interface ID returned by node metrics.",
                }
            },
            "required": ["interface_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_node_metrics",
        "description": "Query node metrics for a specific node at a given time range.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "the id of the node to query, e.g. 'node-1', 'node-2'",
                },
                "start_time": {
                    "type": "string",
                    "description": "the start time of the node metric. The format should be ISO 8601, e.g., '2026-09-23T10:00:00Z'.",
                },
                "end_time": {
                    "type": "string",
                    "description": "the end time of the node metric. The format should be ISO 8601, e.g., '2026-09-23T11:00:00Z'.",
                }
            },
            "required": ["node_id", "start_time", "end_time"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

def call_function(name, args, id):
    if name == "get_node_metrics":
        return get_node_metrics(**args)
    elif name == "get_gpu_health":
        return get_gpu_health(**args)
    elif name == "get_network_health":
        return get_network_health(**args)
    elif name == "get_storage_health":
        return get_storage_health(**args)
    raise ValueError(f"Unknown function: {name}, call_id: {id}")
