from src import scenarios_data


def get_gpu_health(node_id: str, gpu_id: str) -> dict:
    """Return synthetic GPU health observations for the scenario's node and GPU."""
    loaded_data = scenarios_data.load_fixture()
    empty_result = {
        "source": loaded_data["source"],
        "observations": [],
    }

    if node_id not in loaded_data["node_list"]:
        return empty_result

    node_data = loaded_data.get(node_id)
    if not node_data or gpu_id not in node_data["gpu_ids"]:
        return empty_result

    return {
        "source": loaded_data["source"],
        "observations": [
            observation
            for observation in node_data["gpu_health"]
            if observation["resource_id"] == gpu_id
        ],
    }
