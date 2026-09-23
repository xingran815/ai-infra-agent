
from datetime import datetime

NODE_ID_LIST = ["node-1", "node-2", "node-3", "node-4"]

OBS_START_DT = datetime.strptime("2026-09-23T10:00:00Z", "%Y-%m-%dT%H:%M:%SZ")
OBS_END_DT = datetime.strptime("2026-09-23T11:00:00Z", "%Y-%m-%dT%H:%M:%SZ")

tools = [
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

def get_node_metrics(node_id: str, start_time: str, end_time: str) -> dict:
    """Simulate fetching metrics for a specific node from Prometheus."""
    # start_time and end_time can have this kind of format: "2026-09-23T10:00:00Z"
    # convert them to datetime objects for comparison

    start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ")
    end_dt = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%SZ")

    if node_id not in NODE_ID_LIST:
        return {
            "status": "error",
            "error": f"Unknown node: {node_id}",
        }
    elif start_dt > end_dt:
        return {
            "status": "error",
            "error": f"Invalid time range: start_time {start_time} is after end_time {end_time}",
        }
    elif start_dt > OBS_END_DT or end_dt < OBS_START_DT:
        return {
            "status": "error",
            "error": f"No data available for the specified time range: {start_time} to {end_time}",
        }

    return {
        "source": "synthetic",
        "observations": [
            {
                "evidence_id": "obs-001",
                "resource_id": node_id,
                "metric_name": "gpu_utilization",
                "value": 98.0,
                "unit": "percent",
                "timestamp": "2026-09-23T10:00:30Z",
            }
        ]
    }