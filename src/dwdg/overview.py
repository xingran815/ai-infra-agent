"""Small, evidence-linked summaries of selected incident metrics.

No health thresholds are invented. Counter increases exclude resets, duplicate
timestamps and gaps; their valid duration is reported rather than extrapolated.
"""

from collections import defaultdict
from statistics import fmean

from src.dwdg import telemetry


DETAIL_LIMIT = 3
OVERVIEW_METRICS = {
    "node": """
        node_cpu_seconds_total node_load1 node_load5 node_procs_running node_procs_blocked
        node_memory_MemAvailable_bytes node_memory_MemTotal_bytes
        node_memory_SwapFree_bytes node_memory_SwapTotal_bytes node_vmstat_oom_kill
        node_vmstat_pswpin node_vmstat_pswpout node_memory_HardwareCorrupted_bytes
        node_edac_correctable_errors_total node_edac_uncorrectable_errors_total
        node_edac_csrow_correctable_errors_total node_edac_csrow_uncorrectable_errors_total
        node_boot_time_seconds ipmi_chassis_power_state ipmi_fan_speed_state
        ipmi_power_state ipmi_sensor_state ipmi_temperature_state ipmi_voltage_state
        node_hwmon_temp_celsius node_hwmon_temp_max_celsius node_hwmon_temp_crit_celsius
        node_systemd_system_running node_systemd_units
    """.split(),
    "gpu": """
        DCGM_FI_DEV_XID_ERRORS DCGM_FI_DEV_ROW_REMAP_FAILURE
        DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS
        DCGM_FI_DEV_PCIE_REPLAY_COUNTER DCGM_FI_DEV_GPU_TEMP DCGM_FI_DEV_MEMORY_TEMP
        DCGM_FI_DEV_GPU_UTIL DCGM_FI_DEV_MEM_COPY_UTIL DCGM_FI_DEV_FB_USED
        DCGM_FI_DEV_FB_FREE DCGM_FI_DEV_POWER_USAGE
    """.split(),
    "network": """
        node_network_up node_network_carrier node_network_carrier_changes_total
        node_network_receive_errs_total node_network_transmit_errs_total
        node_network_receive_drop_total node_network_transmit_drop_total
        node_network_receive_bytes_total node_network_transmit_bytes_total
        node_infiniband_state_id node_infiniband_physical_state_id
        node_infiniband_link_downed_total node_infiniband_symbol_error_total
        node_infiniband_port_errors_received_total node_infiniband_port_discards_transmitted_total
        node_infiniband_port_transmit_wait_total node_infiniband_port_data_received_bytes_total
        node_infiniband_port_data_transmitted_bytes_total
        node_netstat_Tcp_RetransSegs node_netstat_Tcp_OutSegs
    """.split(),
    "storage": """
        node_filesystem_avail_bytes node_filesystem_size_bytes node_filesystem_files_free
        node_filesystem_files node_filesystem_readonly node_filesystem_device_error
        node_disk_read_time_seconds_total node_disk_reads_completed_total
        node_disk_write_time_seconds_total node_disk_writes_completed_total
        node_disk_read_bytes_total node_disk_written_bytes_total node_disk_io_now
        node_nfs_rpc_retransmissions_total node_nfs_rpcs_total
    """.split(),
    "scheduler": "slurm_status_ok slurm_total_gpus slurm_alloc_gpus".split(),
    "monitoring": """
        up ipmi_up node_scrape_collector_success node_textfile_scrape_error scrape_samples_scraped
    """.split(),
}

# These codes/indicators must not be averaged or interpreted as severity rankings.
STATE_NORMAL = {
    "DCGM_FI_DEV_XID_ERRORS": 0, "DCGM_FI_DEV_ROW_REMAP_FAILURE": 0,
    "ipmi_chassis_power_state": 1, "node_systemd_system_running": 1,
    "node_network_up": 1, "node_network_carrier": 1,
    "node_infiniband_state_id": 4, "node_infiniband_physical_state_id": 5,
    "node_filesystem_readonly": 0, "node_filesystem_device_error": 0,
    "slurm_status_ok": 1, "up": 1, "ipmi_up": 1,
    "node_scrape_collector_success": 1, "node_textfile_scrape_error": 0,
    **{f"ipmi_{kind}_state": 0 for kind in ("fan_speed", "power", "sensor", "temperature", "voltage")},
}


def _gauge_statistics(points):
    """Summarize a {timestamp: value} mapping."""
    if not points:
        return {"latest": None, "min": None, "max": None, "mean": None}
    latest_time = max(points)
    min_time = min(points, key=points.get)
    max_time = max(points, key=points.get)
    return {
        "latest": points[latest_time], "latest_time": latest_time,
        "min": points[min_time], "min_time": min_time,
        "max": points[max_time], "max_time": max_time,
        "mean": fmean(points.values()),
    }


def _counter_changes(points, interval_seconds):
    """Return increases by interval, excluding counter resets and missing scrapes."""
    increases = {}
    resets = gaps = 0
    samples = sorted(points.items())
    for (start, before), (end, after) in zip(samples, samples[1:]):
        elapsed = (telemetry.utc_time(end) - telemetry.utc_time(start)).total_seconds()
        if after < before:
            resets += 1
        if elapsed != interval_seconds:
            gaps += 1
        if after >= before and elapsed == interval_seconds:
            increases[(start, end)] = after - before
    return {"increases": increases, "resets": resets, "gaps": gaps}


def _summarize_metrics(metrics, interval_seconds):
    """Summarize each metric's series, then keep the three most relevant series."""
    summaries = {}
    for name, metric in metrics.items():
        series_summaries = []
        for series in metric["series"]:
            points = series["points"]
            summary = {
                "labels": series["labels"], "evidence_id": series["evidence_id"],
                "coverage": series["coverage"], "unique_samples": series["unique_samples"],
                "ambiguous_timestamps": series["unique_samples"] - len(points),
            }
            if not points:
                summary["status"] = "insufficient_data"
            elif metric["type"] == "counter":
                changes = _counter_changes(points, interval_seconds)
                increases = changes["increases"]
                duration = len(increases) * interval_seconds
                increase = sum(increases.values()) if increases else None
                summary.update(
                    observed_increase=increase, valid_seconds=duration,
                    rate_per_second=increase / duration if duration else None,
                    resets=changes["resets"], excluded_gap_intervals=changes["gaps"],
                )
            elif name in STATE_NORMAL or name == "node_boot_time_seconds":
                samples = sorted(points.items())
                transitions = [
                    {"time": time, "from": before, "to": after}
                    for (_, before), (time, after) in zip(samples, samples[1:])
                    if before != after
                ]
                summary.update(
                    latest=samples[-1][1], latest_time=samples[-1][0],
                    transition_count=len(transitions), transitions=transitions[:DETAIL_LIMIT],
                    transitions_omitted=max(0, len(transitions) - DETAIL_LIMIT),
                )
                if name in STATE_NORMAL:
                    abnormal = [(time, value) for time, value in samples if value != STATE_NORMAL[name]]
                    codes = sorted(set(points.values()))
                    summary.update(
                        non_nominal_samples=len(abnormal),
                        first_non_nominal_time=abnormal[0][0] if abnormal else None,
                        observed_codes=codes[:DETAIL_LIMIT],
                        codes_omitted=max(0, len(codes) - DETAIL_LIMIT),
                    )
            else:
                summary.update(_gauge_statistics(points))
            series_summaries.append(summary)

        # Show non-nominal states first, then counter activity and gauge peaks.
        series_summaries.sort(key=lambda summary: (
            summary.get("non_nominal_samples", 0),
            summary.get("observed_increase") or 0,
            summary.get("max") or 0,
        ), reverse=True)
        summaries[name] = {
            "unit": metric["unit"], "type": metric["type"],
            "series_count": len(series_summaries), "series": series_summaries[:DETAIL_LIMIT],
            "series_omitted": max(0, len(series_summaries) - DETAIL_LIMIT),
        }
        if name in STATE_NORMAL:
            summaries[name]["non_nominal_series"] = sum(
                summary.get("non_nominal_samples", 0) > 0 for summary in series_summaries
            )
    return summaries


def _pair_series(metrics, numerator, denominator):
    """Match complete label sets; a missing or ambiguous side is returned as None."""
    left, right = defaultdict(list), defaultdict(list)
    for name, target in ((numerator, left), (denominator, right)):
        for series in metrics.get(name, {}).get("series", []):
            target[tuple(sorted(series["labels"].items()))].append(series)
    for labels in sorted(left.keys() | right.keys()):
        numerator_series = left[labels][0] if len(left[labels]) == 1 else None
        denominator_series = right[labels][0] if len(right[labels]) == 1 else None
        entry = {"labels": dict(labels)}
        if numerator_series is None or denominator_series is None:
            entry.update(status="missing_or_ambiguous_pair", value=None)
        else:
            entry["evidence_ids"] = [numerator_series["evidence_id"], denominator_series["evidence_id"]]
            entry["source_coverage"] = {
                numerator: numerator_series["coverage"], denominator: denominator_series["coverage"],
            }
        yield entry, numerator_series, denominator_series


def _memory_available_percent(metrics):
    """Calculate available memory as a percentage of total at matching timestamps."""
    entries = []
    for entry, available, total in _pair_series(
        metrics, "node_memory_MemAvailable_bytes", "node_memory_MemTotal_bytes"
    ):
        entry["unit"] = "percent"
        if available is not None and total is not None:
            available_points, total_points = available["points"], total["points"]
            percentages = {}
            for time in sorted(available_points.keys() & total_points.keys()):
                free, capacity = available_points[time], total_points[time]
                if capacity > 0 and 0 <= free <= capacity:
                    percentages[time] = 100 * free / capacity
            entry.update(_gauge_statistics(percentages))
            entry["matched_samples"] = len(percentages)
            entry["status"] = "ok" if percentages else "no_valid_paired_samples"
        entries.append(entry)
    return {
        "series": entries[:DETAIL_LIMIT], "series_omitted": max(0, len(entries) - DETAIL_LIMIT),
        "status": "available" if entries else "missing_metrics",
        "formula": "100 * MemAvailable_bytes / MemTotal_bytes at matching timestamps",
    }


def _disk_read_latency(metrics, interval_seconds):
    """Calculate milliseconds per completed read over shared counter intervals."""
    time_metric = "node_disk_read_time_seconds_total"
    reads_metric = "node_disk_reads_completed_total"
    entries = []
    for entry, read_time, read_count in _pair_series(metrics, time_metric, reads_metric):
        entry["unit"] = "milliseconds"
        if read_time is not None and read_count is not None:
            time_changes = _counter_changes(read_time["points"], interval_seconds)
            read_changes = _counter_changes(read_count["points"], interval_seconds)
            shared_intervals = sorted(time_changes["increases"].keys() & read_changes["increases"].keys())
            seconds = sum(time_changes["increases"][interval] for interval in shared_intervals)
            reads = sum(read_changes["increases"][interval] for interval in shared_intervals)
            entry.update(
                value=1000 * seconds / reads if reads else None,
                status="ok" if reads else "no_completed_reads_or_valid_intervals",
                completed_reads=reads, valid_seconds=len(shared_intervals) * interval_seconds,
                reset_intervals={time_metric: time_changes["resets"], reads_metric: read_changes["resets"]},
                excluded_gap_intervals={time_metric: time_changes["gaps"], reads_metric: read_changes["gaps"]},
            )
        entries.append(entry)
    entries.sort(key=lambda entry: entry.get("value") or 0, reverse=True)
    return {
        "series": entries[:DETAIL_LIMIT], "series_omitted": max(0, len(entries) - DETAIL_LIMIT),
        "status": "available" if entries else "missing_metrics",
        "formula": ("1000 * sum(delta(read_time_seconds)) / sum(delta(reads_completed)); "
                    "only shared consecutive native intervals without resets"),
    }


def build_overview(node_id, start_time, end_time):
    """Fetch categories, decode samples, summarize metrics, and assemble the overview."""
    # Fetch each category once, including all its devices.
    results = {}
    for category in ("node", "gpu", "network", "storage", "monitoring"):
        results[category] = telemetry.get_metrics_gwdg(
            node_id, start_time, end_time, category=category, metrics=OVERVIEW_METRICS[category]
        )
    try:
        results["scheduler"] = telemetry.get_metrics_gwdg(
            node_id, start_time, end_time, category="scheduler", metrics=OVERVIEW_METRICS["scheduler"]
        )
    except FileNotFoundError:
        pass  # An optional scheduler source may not be installed.

    # Decode samples once. All calculations below use {timestamp: value} maps.
    metrics_by_category = {}
    categories = {}
    for category, result in results.items():
        raw_metrics = result["nodes"].get(node_id, {}).get("metrics", {})
        metrics = {}
        for name, metric in raw_metrics.items():
            series_list = []
            for series in metric["series"]:
                values_by_time = defaultdict(list)
                for index, value in series["samples"]:
                    values_by_time[result["timestamps"][index]].append(value)
                series_list.append({
                    "labels": series["labels"], "evidence_id": series["evidence_id"],
                    "coverage": series["coverage"], "unique_samples": len(values_by_time),
                    "points": {time: values[0] for time, values in values_by_time.items() if len(values) == 1},
                })
            metrics[name] = {**metric, "series": series_list}
        metrics_by_category[category] = metrics
        interval_seconds = result["window"]["sample_interval_seconds"]
        categories[category] = {
            "metrics": _summarize_metrics(metrics, interval_seconds),
            "missing_metrics": sorted(set(OVERVIEW_METRICS[category]) - metrics.keys()),
            "sample_interval_seconds": interval_seconds,
        }
    if "scheduler" not in categories:
        categories["scheduler"] = {
            "status": "source_unavailable", "metrics": {},
            "missing_metrics": OVERVIEW_METRICS["scheduler"],
        }

    # Keep controller totals and chip-select-row detail separate.
    node_metrics = categories["node"]["metrics"]
    categories["node"]["edac_controller"] = {}
    categories["node"]["edac_csrow"] = {}
    for name in list(node_metrics):
        if name.startswith("node_edac_"):
            group = "edac_csrow" if name.startswith("node_edac_csrow_") else "edac_controller"
            categories["node"][group][name] = node_metrics.pop(name)

    node = results["node"]
    return {
        "source": node["source"], "node_id": node_id, "window": node["window"],
        "status": "observed" if node_id in node["nodes"] else "no_incident_observations",
        "summary_policy": {
            "series_limit_per_metric": DETAIL_LIMIT,
            "gauges": "Sample mean and observed extrema; no interpolation or invented thresholds.",
            "counters": "Increases/rates on consecutive native intervals only; gaps and resets excluded.",
            "states": "Observed transitions are not event counts; non-nominal states need workload/configuration context.",
            "missing": "Missing or omitted evidence is not a zero value or proof of health.",
            "follow_up": "Use metric names, labels and evidence IDs with category tools or get_metrics_gwdg.",
        },
        "categories": categories,
        "derived": {
            "memory_available_percent": _memory_available_percent(metrics_by_category["node"]),
            "average_disk_read_latency_ms": _disk_read_latency(
                metrics_by_category["storage"], results["storage"]["window"]["sample_interval_seconds"]
            ),
        },
    }
