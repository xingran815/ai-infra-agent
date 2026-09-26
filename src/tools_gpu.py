from pathlib import Path
from src.scenarios_data import load_scenario


def get_gpu_health(node_id: str, gpu_id: str) -> dict:
    """Return synthetic GPU health observations for the scenario's node and GPU."""
    scenario_path = Path(__file__).resolve().parent.parent / "scenarios" / "case_01.json"
    scenario_data = load_scenario(str(scenario_path))

    if node_id == scenario_data["node_id"] and gpu_id == scenario_data["gpu_id"]:
        return {
            "source": "synthetic",
            "observations": scenario_data["gpu_health"],
        }
    else:
        return {
            "source": "synthetic",
            "observations": [],
        }
