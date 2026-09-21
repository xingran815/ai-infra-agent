SUPPORTED_METRICS = (
    "gpu_utilization_percent",
    "cpu_utilization_percent",
    "data_loader_wait_seconds",
    "infiniband_throughput_gbps",
    "infiniband_symbol_errors_total",
    "gpu_temperature_celsius",
)


# A deterministic snapshot of a four-node training cluster.  The values point
# to a node-local input-pipeline bottleneck on node-1: its CPUs are saturated
# and workers wait for batches, while its InfiniBand link and GPU temperature
# look healthy.
_CLUSTER_SNAPSHOT = {
    "node-1": {
        "gpu_utilization_percent": 43.0,
        "cpu_utilization_percent": 96.0,
        "data_loader_wait_seconds": 0.42,
        "infiniband_throughput_gbps": 93.0,
        "infiniband_symbol_errors_total": 0,
        "gpu_temperature_celsius": 66.0,
    },
    "node-2": {
        "gpu_utilization_percent": 91.0,
        "cpu_utilization_percent": 58.0,
        "data_loader_wait_seconds": 0.03,
        "infiniband_throughput_gbps": 94.0,
        "infiniband_symbol_errors_total": 0,
        "gpu_temperature_celsius": 72.0,
    },
    "node-3": {
        "gpu_utilization_percent": 89.0,
        "cpu_utilization_percent": 61.0,
        "data_loader_wait_seconds": 0.04,
        "infiniband_throughput_gbps": 92.0,
        "infiniband_symbol_errors_total": 0,
        "gpu_temperature_celsius": 71.0,
    },
    "node-4": {
        "gpu_utilization_percent": 92.0,
        "cpu_utilization_percent": 55.0,
        "data_loader_wait_seconds": 0.02,
        "infiniband_throughput_gbps": 95.0,
        "infiniband_symbol_errors_total": 0,
        "gpu_temperature_celsius": 73.0,
    },
}

_METRIC_UNITS = {
    "gpu_utilization_percent": "percent",
    "cpu_utilization_percent": "percent",
    "data_loader_wait_seconds": "seconds_per_batch",
    "infiniband_throughput_gbps": "Gbps",
    "infiniband_symbol_errors_total": "errors",
    "gpu_temperature_celsius": "celsius",
}


tools = [
    {
        "type": "function",
        "name": "query_prometheus",
        "description": "Query Prometheus for a specific metric on a given node.",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "string",
                    "description": "the name of the node to query",
                },
                "metric": {
                    "type": "string",
                    "enum": list(SUPPORTED_METRICS),
                    "description": "The simulated Prometheus metric to retrieve.",
                },
            },
            "required": ["node", "metric"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def query_prometheus(node: str, metric: str) -> dict:
    """Return one metric from the deterministic simulated cluster snapshot."""
    if node not in _CLUSTER_SNAPSHOT:
        return {
            "status": "error",
            "error": f"Unknown node: {node}",
            "available_nodes": sorted(_CLUSTER_SNAPSHOT),
        }

    if metric not in SUPPORTED_METRICS:
        return {
            "status": "error",
            "error": f"Unsupported metric: {metric}",
            "available_metrics": list(SUPPORTED_METRICS),
        }

    return {
        "status": "success",
        "node": node,
        "metric": metric,
        "value": _CLUSTER_SNAPSHOT[node][metric],
        "unit": _METRIC_UNITS[metric],
        "labels": {
            "job": "distributed-training",
            "gpu_count": "8",
            "network": "InfiniBand",
            "aggregation": "node_average",
        },
        "timestamp": "2026-09-21T12:00:00Z",
    }
