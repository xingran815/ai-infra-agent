"""Bound model-facing tool output and offer selectors for an explicit retry.

This is an output budget, not a CSV scan or memory budget. Internal queries (in
particular the overview's inputs) remain complete. Never truncate raw evidence
or silently change the requested window to make a response fit.
"""

import json


RAW_MAX_BYTES = 24 * 1024
OVERVIEW_MAX_BYTES = 48 * 1024
RAW_MAX_SERIES = 64
RAW_MAX_RUNS = 1000
HINT_MAX_BYTES = 16 * 1024
MAX_HINT_METRICS = 32
MAX_HINT_LABELS = 12
MAX_HINT_VALUES = 64
MAX_SELECTOR_LENGTH = 256
LABEL_PRIORITY = ("collector", "job", "uuid", "device", "cpu", "mode", "sensor", "mountpoint")


def serialize_tool_result(result):
    """Use the same UTF-8 JSON representation for measurement and model input."""
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _size(result):
    return len(serialize_tool_result(result).encode("utf-8"))


def _metric_entries(result, category):
    for node_id, node in result.get("nodes", {}).items():
        for name, metric in node.get("metrics", {}).items():
            yield node_id, category, name, metric
    for group, content in result.get("categories", {}).items():
        for name, metric in content.get("metrics", {}).items():
            yield result["node_id"], group, name, metric


def _label_order(name):
    return (LABEL_PRIORITY.index(name) if name in LABEL_PRIORITY else len(LABEL_PRIORITY), name)


def _metric_hint(node_id, category, name, metric, args, window):
    series = metric.get("series", [])
    values = {}
    for item in series:
        for label, value in item.get("labels", {}).items():
            if isinstance(value, str):
                values.setdefault(label, set()).add(value)
    selectors = {}
    for label in sorted(values, key=_label_order):
        if len(selectors) == MAX_HINT_LABELS:
            break
        if len(label) > MAX_SELECTOR_LENGTH:
            continue
        choices = sorted(value for value in values[label] if len(value) <= MAX_SELECTOR_LENGTH)[:MAX_HINT_VALUES]
        selectors[label] = {
            "values": choices, "total_values": len(values[label]),
            "omitted_values": len(values[label]) - len(choices),
        }
    hint = {
        "metric": name, "node_id": node_id, "category": category,
        "series_count": len(series), "label_selectors": selectors,
        "omitted_labels": len(values) - len(selectors),
    }
    # Keep the original resource and label constraints. Category wrappers cannot
    # accept label_filters, so every example uses the generic tool instead.
    retry_args = {
        "node_id": node_id, "start_time": window["start_time"],
        "end_time": window["end_time"], "category": category, "metrics": [name],
    }
    if args.get("resource_id") is not None:
        retry_args["resource_id"] = args["resource_id"]
    filters = dict(args.get("label_filters") or {})
    if series:
        # Use one actual label combination, never a cross product of the catalog.
        for label, value in sorted(series[0].get("labels", {}).items(), key=lambda pair: _label_order(pair[0])):
            if label in selectors and value in selectors[label]["values"]:
                filters[label] = value
    if filters:
        retry_args["label_filters"] = filters
    retry = {"tool": "get_metrics_gwdg", "arguments": retry_args}
    if _size(retry) <= 2048:
        hint["example_retry"] = retry
    # Leave room for other metrics. Long/high-cardinality labels must not consume
    # the entire hint response; values are omitted whole with explicit counts.
    while _size(hint) > 4096 and selectors:
        label = max(selectors, key=lambda key: _size(selectors[key]))
        entry = selectors[label]
        if entry["values"]:
            entry["values"] = entry["values"][:len(entry["values"]) // 2]
            entry["omitted_values"] = entry["total_values"] - len(entry["values"])
        else:
            del selectors[label]
            hint["omitted_labels"] += 1
    return hint


def limit_tool_result(tool_name, args, result):
    """Return a complete result or a bounded, evidence-free narrowing response."""
    overview = tool_name == "get_overview_gwdg"
    raw = "time_axis" in result
    byte_limit = OVERVIEW_MAX_BYTES if overview else RAW_MAX_BYTES
    category = args.get("category", "all")
    if tool_name not in ("get_metrics_gwdg", "get_resource_ids_gwdg", "get_overview_gwdg"):
        category = tool_name.removeprefix("get_").removesuffix("_metrics_gwdg")
    entries = list(_metric_entries(result, category))
    series_count = sum(len(metric.get("series", [])) for _, _, _, metric in entries)
    run_count = sum(len(series.get("sample_runs", [])) for _, _, _, metric in entries for series in metric.get("series", []))
    limits = {"response_bytes": byte_limit}
    observed = {"response_bytes": _size(result)}
    if raw:
        limits.update(series=RAW_MAX_SERIES, sample_runs=RAW_MAX_RUNS)
        observed.update(metrics=len(entries), series=series_count, sample_runs=run_count)
    exceeded = [key for key, maximum in limits.items() if observed[key] > maximum]
    if not exceeded:
        return result

    response = {
        "status": "query_too_broad", "data_returned": False,
        "message": "No readings or evidence IDs returned. Narrow the query explicitly and retry.",
        "limits": limits, "observed": observed, "exceeded": exceeded,
        "available": {
            "scope": "returned_overview_details" if overview else "matched_query",
            "metrics": [], "total_metrics": len(entries), "omitted_metrics": len(entries),
        },
        "retry_guidance": [
            "Use get_metrics_gwdg with selected metrics and exact label_filters from the catalog. All label filters must match.",
            "Examples select observed label combinations; they are not a health ranking or a guarantee the retry will fit.",
            "Select a metric first if hints are omitted. If one series is still too large, explicitly request a shorter time window.",
            "Missing or omitted readings are not evidence of health. Do not infer values from these selectors.",
        ],
    }
    # Complete entries only: never cut JSON, metric names, or selector values.
    # Counts expose every omission, including values too long to display safely.
    window = result.get("window")
    if window:
        response["window"] = window
        for node_id, group, name, metric in entries:
            if len(response["available"]["metrics"]) == MAX_HINT_METRICS:
                break
            hint = _metric_hint(node_id, group, name, metric, args, window)
            response["available"]["metrics"].append(hint)
            response["available"]["omitted_metrics"] -= 1
            if _size(response) > HINT_MAX_BYTES:
                response["available"]["metrics"].pop()
                response["available"]["omitted_metrics"] += 1
    else:
        # Resource discovery has no metric selectors. A smaller explicit window
        # can reduce its inventory; it must never masquerade as a complete list.
        response["retry_guidance"] = ["Retry resource discovery with an explicitly shorter time window."]
    return response
