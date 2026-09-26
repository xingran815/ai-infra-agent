from src.scenarios_data import load_scenario
from pathlib import Path

def get_network_health(interface_id: str) -> dict:
    """Simulate fetching network health metrics for a specific interface."""
    scenario_path = Path(__file__).resolve().parent.parent / "scenarios" / "case_01.json"
    scenario_data = load_scenario(str(scenario_path))
    if interface_id == scenario_data["network_interface_id"]:
        return {
            "source": "synthetic",
            "observations": scenario_data["network_health"],
        }
        # return {
        #     "source": "synthetic",
        #     "observations": [
        #         {
        #             "evidence_id": "net-9-errors-001",
        #             "resource_id": "net-9",
        #             "metric_name": "link_errors_in_window",
        #             "value": 12,
        #             "unit": "errors",
        #             "window_start": "2026-09-23T10:00:00Z",
        #             "window_end": "2026-09-23T10:30:00Z"
        #         }
        #     ]
        # }
    else:
        return {
            "source": "synthetic",
            "observations": [],
        }
