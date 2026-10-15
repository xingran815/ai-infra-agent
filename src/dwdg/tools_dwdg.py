"""GWDG tool entry points; query implementation lives in telemetry.py.

Call get_resource_ids_gwdg before metric tools to discover nodes and devices.
"""

from inspect import signature

from src.dwdg.execution import ToolArgumentError
from src.dwdg.overview import build_overview
from src.dwdg.response_limits import limit_tool_result
from src.dwdg.telemetry import (
    GwdgDataSource, DATA_DIR, SOURCE, SAMPLE_INTERVAL_SECONDS, IDENTITY_COLUMNS,
    METRIC_CATEGORIES, load_data_gwdg, load_scheduler_data_gwdg,
    unload_data_gwdg, get_resource_ids_gwdg, get_metrics_gwdg,
)


RAW_SAMPLE_DESCRIPTION = (
    "Raw responses use time_axis {start, step_seconds, count} and sample_runs "
    "[first_index, repeat_count, value]. A run represents equal values at consecutive "
    "indices first_index through first_index + repeat_count - 1. Timestamp = start + "
    "step_seconds * index. count is the nominal step count in the exclusive-end window, "
    "rounded up for partial steps. Fractional first_index preserves off-grid times and "
    "unaligned bounds. 'miss' represents missing observations on the series' inferred "
    "native schedule, never zero; skip these runs in calculations. Duplicate observations "
    "retain repeated indices. Coverage uses the modal observed sampling phase; "
    "returned_samples counts observed rows including duplicates and excludes 'miss'. "
    "Resource inventories come from get_resource_ids_gwdg; metric responses retain series labels. "
    "Oversized tool output returns status=query_too_broad without readings or evidence IDs. "
    "Use its available metrics, collector/job/device labels and example_retry arguments "
    "to retry get_metrics_gwdg with narrower selectors. Hints report omissions. "
    "The tool never silently shortens the time window."
)


def _window_properties():
    return {
        "start_time": {"type": "string", "description": "Inclusive ISO 8601 timestamp with timezone."},
        "end_time": {"type": "string", "description": "Exclusive ISO 8601 timestamp with timezone."},
    }


def _category_tool(name, description, resource_description, *, device=False):
    """Build strict schemas; models pass null explicitly to select all metrics."""
    properties = {
        "resource_id": {"type": "string", "minLength": 1, "description": resource_description},
        **_window_properties(),
        "metrics": {
            "type": ["array", "null"], "items": {"type": "string"},
            "description": "Metric names in this category. null selects all; [] selects none.",
        },
    }
    if device:
        properties["node_id"] = {
            "type": "string", "minLength": 1,
            "description": "Owning node ID returned by get_resource_ids_gwdg.",
        }
    return {
        "type": "function", "name": name,
        "description": "Call get_resource_ids_gwdg first. " + description + " " + RAW_SAMPLE_DESCRIPTION,
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
        "metrics; 'scheduler' reads Slurm state at its native 600-second cadence. "
        + RAW_SAMPLE_DESCRIPTION
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
        "Read selected node metrics. Interpret CPU times and other counters using changes or rates.",
        "Node ID from discovery.",
    ),
    _category_tool(
        "get_gpu_metrics_gwdg", "Read selected GPU metrics for one GPU.",
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
        "get_scheduler_metrics_gwdg", "Read selected scheduler metrics for one node.",
        "Node ID from discovery.",
    ),
    _category_tool(
        "get_monitoring_metrics_gwdg",
        "Read selected monitoring metrics for one node, preserving job and collector series.",
        "Node ID from discovery.",
    ),
]


def _require_resource_id(resource_id: str) -> str:
    if not isinstance(resource_id, str) or not resource_id:
        raise ToolArgumentError("resource_id must be a non-empty string")
    return resource_id


def get_node_metrics_gwdg(resource_id: str, start_time: str, end_time: str, metrics: list[str] | None = None, *, source: GwdgDataSource | None = None) -> dict:
    """Return selected node metrics; None selects all, [] selects none."""
    return get_metrics_gwdg(resource_id, start_time, end_time, metrics=metrics, category="node", resource_id=_require_resource_id(resource_id), source=source)


def get_gpu_metrics_gwdg(node_id: str, resource_id: str, start_time: str, end_time: str, metrics: list[str] | None = None, *, source: GwdgDataSource | None = None) -> dict:
    """Return selected GPU metrics for a GPU UUID; None selects all, [] none."""
    return get_metrics_gwdg(node_id, start_time, end_time, metrics=metrics, category="gpu", resource_id=_require_resource_id(resource_id), source=source)


def get_network_metrics_gwdg(node_id: str, resource_id: str, start_time: str, end_time: str, metrics: list[str] | None = None, *, source: GwdgDataSource | None = None) -> dict:
    """Return selected interface/adapter and node-wide metrics; None selects all."""
    return get_metrics_gwdg(node_id, start_time, end_time, metrics=metrics, category="network", resource_id=_require_resource_id(resource_id), source=source)


def get_storage_metrics_gwdg(node_id: str, resource_id: str, start_time: str, end_time: str, metrics: list[str] | None = None, *, source: GwdgDataSource | None = None) -> dict:
    """Return selected device and node-wide NFS metrics; None selects all."""
    return get_metrics_gwdg(node_id, start_time, end_time, metrics=metrics, category="storage", resource_id=_require_resource_id(resource_id), source=source)


def get_scheduler_metrics_gwdg(resource_id: str, start_time: str, end_time: str, metrics: list[str] | None = None, *, source: GwdgDataSource | None = None) -> dict:
    """Return selected scheduler metrics; None selects all, [] selects none."""
    return get_metrics_gwdg(resource_id, start_time, end_time, metrics=metrics, category="scheduler", resource_id=_require_resource_id(resource_id), source=source)


def get_monitoring_metrics_gwdg(resource_id: str, start_time: str, end_time: str, metrics: list[str] | None = None, *, source: GwdgDataSource | None = None) -> dict:
    """Return selected monitoring metrics; None selects all, [] selects none."""
    return get_metrics_gwdg(resource_id, start_time, end_time, metrics=metrics, category="monitoring", resource_id=_require_resource_id(resource_id), source=source)


def get_overview_gwdg(node_id: str, start_time: str, end_time: str, *, source: GwdgDataSource | None = None) -> dict:
    """Return bounded health summaries and derived values, without raw samples."""
    return build_overview(node_id, start_time, end_time, source=source)


GWDG_TOOLS.append({
    "type": "function", "name": "get_overview_gwdg",
    "description": (
        "Start here after resource discovery for a compact node health overview. "
        "Checks 28 critical metrics: host load, blocked tasks, OOM and uncorrectable memory "
        "errors, system state, GPU XID/remap/PCIe errors, GPU temperature/utilization, "
        "network link states/errors/drops, filesystem capacity/errors, NFS retries, "
        "monitoring failures and Slurm health/allocations. Also returns memory available "
        "percentage and disk read latency, coverage and evidence IDs. Other metrics are "
        "not checked here; retrieve them with category tools. Raw samples are not returned. "
        "Detail lists are bounded with omitted counts; use category tools or get_metrics_gwdg "
        "for follow-up. Missing evidence is not healthy evidence."
    ),
    "parameters": {
        "type": "object", "properties": {
            "node_id": {"type": "string", "minLength": 1}, **_window_properties(),
        },
        "required": ["node_id", "start_time", "end_time"], "additionalProperties": False,
    },
    "strict": True,
})


def call_function_dwdg(name, args, id, *, source: GwdgDataSource | None = None):
    """Dispatch a GWDG tool call using the same interface as tools.call_function."""
    functions = {
        "get_resource_ids_gwdg": get_resource_ids_gwdg,
        "get_metrics_gwdg": get_metrics_gwdg,
        "get_node_metrics_gwdg": get_node_metrics_gwdg,
        "get_gpu_metrics_gwdg": get_gpu_metrics_gwdg,
        "get_network_metrics_gwdg": get_network_metrics_gwdg,
        "get_storage_metrics_gwdg": get_storage_metrics_gwdg,
        "get_scheduler_metrics_gwdg": get_scheduler_metrics_gwdg,
        "get_monitoring_metrics_gwdg": get_monitoring_metrics_gwdg,
        "get_overview_gwdg": get_overview_gwdg,
    }
    function = functions.get(name)
    if function is None:
        raise ToolArgumentError("Unknown tool. Select one of the available tools.", code="unknown_tool",
                                details={"available_tools": list(functions)})
    try:
        signature(function).bind(**args, source=source)
    except TypeError as error:
        raise ToolArgumentError(str(error)) from error
    result = function(**args, source=source)
    return limit_tool_result(name, args, result)
