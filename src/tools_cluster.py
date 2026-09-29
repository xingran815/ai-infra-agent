from datetime import datetime

from src import scenarios_data


def get_node_metrics(node_id: str, start_time: str, end_time: str) -> dict:
    """Simulate fetching metrics for a specific node from Prometheus."""
    # start_time and end_time can have this kind of format: "2026-09-23T10:00:00Z"
    # convert them to datetime objects for comparison

    start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ")
    end_dt = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%SZ")

    loaded_data = scenarios_data.load_fixture()
    observation_start = datetime.strptime(
        loaded_data["start_time"], "%Y-%m-%dT%H:%M:%SZ"
    )
    observation_end = datetime.strptime(
        loaded_data["end_time"], "%Y-%m-%dT%H:%M:%SZ"
    )
    node_ids = loaded_data["node_list"]

    if node_id not in node_ids:
        return {
            "status": "error",
            "error": f"Unknown node: {node_id}",
        }
    elif start_dt > end_dt:
        return {
            "status": "error",
            "error": f"Invalid time range: start_time {start_time} is after end_time {end_time}",
        }
    elif start_dt > observation_end or end_dt < observation_start:
        return {
            "source": loaded_data["source"],
            "observations": [],
        }

    node_data = loaded_data.get(node_id)
    if not node_data:
        return {
            "source": loaded_data["source"],
            "observations": [],
        }

    observations = [
        observation
        for observation in node_data["node_metrics"]
        if start_dt
        <= datetime.strptime(observation["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
        < end_dt
    ]

    return {
        "source": loaded_data["source"],
        "network_interface_ids": node_data["network_interface_ids"],
        "gpu_ids": node_data["gpu_ids"],
        "storage_id": node_data["storage_id"],
        "observations": observations,
        "historical_events": node_data["historical_events"],
    }
