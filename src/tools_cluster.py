
from datetime import datetime
from src.scenarios_data import load_scenario
from pathlib import Path



def get_node_metrics(node_id: str, start_time: str, end_time: str) -> dict:
    """Simulate fetching metrics for a specific node from Prometheus."""
    # start_time and end_time can have this kind of format: "2026-09-23T10:00:00Z"
    # convert them to datetime objects for comparison

    start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ")
    end_dt = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%SZ")

    scenario_path = Path(__file__).resolve().parent.parent / "scenarios" / "case_01.json"
    scenario_data = load_scenario(str(scenario_path))
    OBS_START_DT = datetime.strptime(scenario_data["start_time"], "%Y-%m-%dT%H:%M:%SZ")
    OBS_END_DT = datetime.strptime(scenario_data["end_time"], "%Y-%m-%dT%H:%M:%SZ")
    NODE_ID_LIST = scenario_data["node_list"]

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
            "source": "synthetic",
            "observations": []
        }

    if node_id != scenario_data["node_id"]: 
        return {
            "source": "synthetic",
            "observations": [],
        }

    observations = [
        observation
        for observation in scenario_data["node_metrics"]
        if start_dt <= datetime.strptime(observation["timestamp"], "%Y-%m-%dT%H:%M:%SZ") < end_dt
    ]

    return {
        "source": "synthetic",
        "network_interface_id": scenario_data["network_interface_id"],
        "gpu_id": scenario_data["gpu_id"],
        "storage_id": scenario_data["storage_id"],
        "observations": observations,
    }
