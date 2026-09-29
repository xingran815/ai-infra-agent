from src import scenarios_data


def get_network_health(interface_id: str) -> dict:
    """Simulate fetching network health metrics for a specific interface."""
    loaded_data = scenarios_data.load_fixture()

    for node_id in loaded_data["node_list"]:
        node_data = loaded_data.get(node_id)
        if node_data and interface_id in node_data["network_interface_ids"]:
            return {
                "source": loaded_data["source"],
                "observations": [
                    observation
                    for observation in node_data["network_health"]
                    if observation["resource_id"] == interface_id
                ],
            }

    return {
        "source": loaded_data["source"],
        "observations": [],
    }
