from src import scenarios_data


def get_storage_health(storage_id: str) -> dict:
    """Return synthetic storage health observations for the specified storage ID."""
    loaded_data = scenarios_data.load_fixture()

    for node_id in loaded_data["node_list"]:
        node_data = loaded_data.get(node_id)
        if node_data and storage_id == node_data["storage_id"]:
            return {
                "source": loaded_data["source"],
                "observations": [
                    observation
                    for observation in node_data["storage_health"]
                    if observation["resource_id"] == storage_id
                ],
            }

    return {
        "source": loaded_data["source"],
        "observations": [],
    }
