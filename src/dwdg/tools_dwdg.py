"""GWDG tool entry points; query implementation lives in telemetry.py.

Call get_resource_ids_gwdg before metric tools to discover nodes and devices.
"""

from src.dwdg.telemetry import (
    DATA_DIR, SOURCE, SAMPLE_INTERVAL_SECONDS, IDENTITY_COLUMNS,
    METRIC_CATEGORIES, load_data_gwdg, load_scheduler_data_gwdg,
    unload_data_gwdg, get_resource_ids_gwdg, get_metrics_gwdg,
)


def _window_properties():
    return {
        "start_time": {"type": "string", "description": "Inclusive ISO 8601 timestamp with timezone."},
        "end_time": {"type": "string", "description": "Exclusive ISO 8601 timestamp with timezone."},
    }


def _category_tool(name, description, resource_description, *, device=False):
    """Build strict category schemas with mandatory resource and window IDs."""
    properties = {
        "resource_id": {"type": "string", "minLength": 1, "description": resource_description},
        **_window_properties(),
    }
    if device:
        properties["node_id"] = {
            "type": "string", "minLength": 1,
            "description": "Owning node ID returned by get_resource_ids_gwdg.",
        }
    return {
        "type": "function", "name": name,
        "description": "Call get_resource_ids_gwdg first. " + description,
        "parameters": {
            "type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False,
        },
        "strict": True,
    }


GWDG_TOOLS = [{
    "type": "function",
    "name": "get_resource_ids_gwdg",
    "description": (
        "Call this first, before retrieving metrics. Discover all node and device "
        "resource IDs observed in the selected source and UTC window, including "
        "GPU, network, InfiniBand, disk, filesystem, NVMe, and logical CPU IDs. "
        "Use the returned node and resource IDs for subsequent metric queries."
    ),
    "parameters": {
        "type": "object", "properties": _window_properties(),
        "required": ["start_time", "end_time"], "additionalProperties": False,
    },
    "strict": True,
}, {
    "type": "function",
    "name": "get_metrics_gwdg",
    "description": (
        "Call get_resource_ids_gwdg first. Read raw metrics by category and optional "
        "metric names, resource ID, and exact label matches. 'all' reads all incident "
        "metrics; 'scheduler' reads Slurm state at its native 600-second cadence."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "minLength": 1, "description": "Discovered node ID."},
            **_window_properties(),
            "metrics": {
                "type": ["array", "null"], "items": {"type": "string"},
                "description": "Dataset metric names; omit or null for the complete category; [] selects none.",
            },
            "category": {
                "type": "string", "enum": list(METRIC_CATEGORIES),
                "description": "Metric category; defaults to all. monitor is an alias for monitoring.",
            },
            "resource_id": {
                "type": ["string", "null"], "minLength": 1,
                "description": "Discovered GPU UUID or exact device ID; omit for all resources. Node categories use the node ID.",
            },
            "label_filters": {
                "type": ["object", "null"], "additionalProperties": {"type": "string"},
                "description": "Exact label matches, all required; for example cpu=0 and mode=idle.",
            },
        },
        "required": ["node_id", "start_time", "end_time"],
        "additionalProperties": False,
    },
    "strict": False,
},
    _category_tool(
        "get_node_metrics_gwdg",
        "Read all node metrics. Interpret CPU times and other counters using changes or rates.",
        "Node ID from discovery.",
    ),
    _category_tool(
        "get_gpu_metrics_gwdg", "Read all GPU metrics for one GPU.",
        "GPU ID from resources.gpu_ids; matches the uuid label.", device=True,
    ),
    _category_tool(
        "get_network_metrics_gwdg",
        "Read network metrics for one interface or InfiniBand adapter, retaining node-wide statistics.",
        "Exact device ID from network_interface_ids or infiniband_adapter_ids.", device=True,
    ),
    _category_tool(
        "get_storage_metrics_gwdg",
        "Read storage metrics for one device, retaining node-wide NFS counters; no partition mapping is inferred.",
        "Exact disk, filesystem, or NVMe device ID from storage_ids.", device=True,
    ),
    _category_tool(
        "get_scheduler_metrics_gwdg", "Read all scheduler metrics for one node.",
        "Node ID from discovery.",
    ),
    _category_tool(
        "get_monitoring_metrics_gwdg",
        "Read all monitoring metrics for one node, preserving job and collector series.",
        "Node ID from discovery.",
    ),
]


def _require_resource_id(resource_id: str) -> str:
    if not isinstance(resource_id, str) or not resource_id:
        raise ValueError("resource_id must be a non-empty string")
    return resource_id


def get_node_metrics_gwdg(resource_id: str, start_time: str, end_time: str) -> dict:
    """Return all node metrics for a discovered node ID."""
    return get_metrics_gwdg(resource_id, start_time, end_time, category="node", resource_id=_require_resource_id(resource_id))


def get_gpu_metrics_gwdg(node_id: str, resource_id: str, start_time: str, end_time: str) -> dict:
    """Return all GPU metrics for a discovered GPU UUID and its owning node."""
    return get_metrics_gwdg(node_id, start_time, end_time, category="gpu", resource_id=_require_resource_id(resource_id))


def get_network_metrics_gwdg(node_id: str, resource_id: str, start_time: str, end_time: str) -> dict:
    """Return interface/adapter metrics plus node-wide network series."""
    return get_metrics_gwdg(node_id, start_time, end_time, category="network", resource_id=_require_resource_id(resource_id))


def get_storage_metrics_gwdg(node_id: str, resource_id: str, start_time: str, end_time: str) -> dict:
    """Return exact device metrics plus node-wide NFS series."""
    return get_metrics_gwdg(node_id, start_time, end_time, category="storage", resource_id=_require_resource_id(resource_id))


def get_scheduler_metrics_gwdg(resource_id: str, start_time: str, end_time: str) -> dict:
    """Return all scheduler metrics for a discovered node ID."""
    return get_metrics_gwdg(resource_id, start_time, end_time, category="scheduler", resource_id=_require_resource_id(resource_id))


def get_monitoring_metrics_gwdg(resource_id: str, start_time: str, end_time: str) -> dict:
    """Return all monitoring metrics for a discovered node ID."""
    return get_metrics_gwdg(resource_id, start_time, end_time, category="monitoring", resource_id=_require_resource_id(resource_id))
