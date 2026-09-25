
def get_network_health(interface_id: str) -> dict:
    """Simulate fetching network health metrics for a specific interface."""
    if interface_id == "net-9":
        return {
            "source": "synthetic",
            "observations": [
                {
                    "evidence_id": "net-9-errors-001",
                    "resource_id": "net-9",
                    "metric_name": "link_errors_in_window",
                    "value": 12,
                    "unit": "errors",
                    "window_start": "2026-09-23T10:00:00Z",
                    "window_end": "2026-09-23T10:30:00Z"
                }
            ]
        }
    else:
        return {
            "status": "error",
            "error": f"Unknown network interface: {interface_id}",
        }