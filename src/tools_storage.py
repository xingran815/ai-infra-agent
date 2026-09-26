from pathlib import Path

from src.scenarios_data import load_scenario


def get_storage_health(storage_id: str) -> dict:
    """Return synthetic storage health observations for the specified storage ID."""
    scenario_path = Path(__file__).resolve().parent.parent / "scenarios" / "case_01.json"
    scenario_data = load_scenario(str(scenario_path))

    if storage_id == scenario_data["storage_id"]:
        return {
            "source": "synthetic",
            "observations": scenario_data["storage_health"],
        }
    else:
        return {
            "source": "synthetic",
            "observations": [],
        }
