import unittest

from src.scenarios_data import load_scenario
from src.tools_cluster import get_node_metrics
from src.tools_gpu import get_gpu_health
from src.tools_network import get_network_health
from src.tools_storage import get_storage_health


class ToolFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_scenario("scenarios/case_01.json")

    def test_node_metrics_returns_node_resources_and_filtered_metrics(self):
        result = get_node_metrics(
            "node-2", "2026-09-23T10:00:00Z", "2026-09-23T10:30:00Z"
        )

        self.assertEqual(
            result["gpu_ids"],
            [
                "node-2-gpu-0",
                "node-2-gpu-1",
                "node-2-gpu-2",
                "node-2-gpu-3",
            ],
        )
        self.assertEqual(
            result["network_interface_ids"],
            ["net-9", "net-10", "net-11", "net-12"],
        )
        self.assertEqual(result["storage_id"], "storage-2")
        self.assertEqual(len(result["observations"]), 2)

    def test_node_metrics_filters_out_observations_outside_query(self):
        result = get_node_metrics(
            "node-2", "2026-09-23T10:01:00Z", "2026-09-23T10:02:00Z"
        )

        self.assertEqual(result["observations"], [])

    def test_unknown_node_returns_an_error(self):
        result = get_node_metrics(
            "node-3", "2026-09-23T10:00:00Z", "2026-09-23T10:30:00Z"
        )

        self.assertEqual(result["status"], "error")

    def test_gpu_health_returns_only_requested_gpu(self):
        result = get_gpu_health("node-2", "node-2-gpu-1")

        self.assertEqual(len(result["observations"]), 1)
        self.assertEqual(result["observations"][0]["resource_id"], "node-2-gpu-1")

    def test_gpu_health_rejects_gpu_from_another_node(self):
        result = get_gpu_health("node-1", "node-2-gpu-0")

        self.assertEqual(result["observations"], [])

    def test_network_health_returns_only_requested_interface(self):
        result = get_network_health("net-9")

        self.assertEqual(len(result["observations"]), 1)
        self.assertEqual(result["observations"][0]["resource_id"], "net-9")

    def test_storage_health_finds_storage_below_its_node(self):
        result = get_storage_health("storage-1")

        self.assertEqual(len(result["observations"]), 1)
        self.assertEqual(result["observations"][0]["resource_id"], "storage-1")

    def test_unknown_resources_return_empty_observations(self):
        self.assertEqual(get_gpu_health("node-2", "missing")["observations"], [])
        self.assertEqual(get_network_health("missing")["observations"], [])
        self.assertEqual(get_storage_health("missing")["observations"], [])


if __name__ == "__main__":
    unittest.main()
