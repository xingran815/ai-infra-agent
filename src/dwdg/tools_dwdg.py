"""GWDG tool entry points. Shared data access lives in telemetry.py."""

from src.dwdg.metric_metadata import NODE_METRICS, NODE_METRIC_METADATA
from src.dwdg.telemetry import (
    DATA_DIR,
    SOURCE,
    SAMPLE_INTERVAL_SECONDS,
    IDENTITY_COLUMNS,
    load_data_gwdg,
    unload_data_gwdg,
    get_metrics_gwdg,
)

# This tool can be supplied separately from the existing synthetic-fixture tools.
GWDG_TOOLS = [{
    "type": "function",
    "name": "get_node_metrics_gwdg",
    "description": (
        "Read raw GWDG node telemetry for a UTC window with an exclusive end. "
        "Select a few node metrics for manageable evidence. "
        "CPU times and other counters require changes or rates for interpretation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "Original v2 node pseudonym."},
            "start_time": {"type": "string", "description": "Inclusive ISO 8601 timestamp with timezone."},
            "end_time": {"type": "string", "description": "Exclusive ISO 8601 timestamp with timezone."},
            "metrics": {"type": ["array", "null"], "items": {"type": "string", "enum": list(NODE_METRICS)}},
        },
        "required": ["node_id", "start_time", "end_time"],
        "additionalProperties": False,
    },
    "strict": False,
}, {
    "type": "function",
    "name": "get_metrics_gwdg",
    "description": "Read named GWDG metrics from any category in a UTC window, optionally matching resource labels.",
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "Original v2 node pseudonym."},
            "start_time": {"type": "string", "description": "Inclusive ISO 8601 timestamp with timezone."},
            "end_time": {"type": "string", "description": "Exclusive ISO 8601 timestamp with timezone."},
            "metrics": {"type": "array", "items": {"type": "string"}, "description": "Original dataset metric names."},
            "label_filters": {
                "type": ["object", "null"], "additionalProperties": {"type": "string"},
                "description": "Exact label matches, all required; for example cpu=0 and mode=idle.",
            },
        },
        "required": ["node_id", "start_time", "end_time", "metrics"],
        "additionalProperties": False,
    },
    "strict": False,
}]


def get_node_metrics_gwdg(
    node_id: str,
    start_time: str,
    end_time: str,
    metrics: list[str] | None = None,
) -> dict:
    """Return raw node metrics in [start_time, end_time).

    Load an incident file first. None selects all node metrics; [] selects none.
    Resources describe the full window and supply IDs for follow-up queries.
    Use get_metrics_gwdg for direct queries with optional label filters.
    """
    if metrics is None:
        metrics = list(NODE_METRICS)
    if set(metrics) - set(NODE_METRICS):
        raise ValueError(f"Unsupported node metrics: {sorted(set(metrics) - set(NODE_METRICS))}")
    return get_metrics_gwdg(node_id, start_time, end_time, metrics)


def get_gpu_metrics_gwdg():
    """
    Planned return: DCGM metrics for a GPU on a node and UTC time range.

    Includes utilization, memory, temperature, power, clocks, NVLink/PCIe
    indicators, XID errors, and row-remapping state. Identify the GPU by its
    pseudonymized UUID or GPU index. Some incidents have no DCGM telemetry.

    Source: per-incident *_tidy.csv.xz telemetry, sampled every 10 seconds.

    Preserve timestamps, values, seriesId, and distinguishing labels.
    Availability varies by incident; missing scrapes are absent rows.

    Metric names:
        - "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS",
        - "DCGM_FI_DEV_DEC_UTIL",
        - "DCGM_FI_DEV_ENC_UTIL",
        - "DCGM_FI_DEV_FB_FREE",
        - "DCGM_FI_DEV_FB_USED",
        - "DCGM_FI_DEV_GPU_TEMP",
        - "DCGM_FI_DEV_GPU_UTIL",
        - "DCGM_FI_DEV_MEMORY_TEMP",
        - "DCGM_FI_DEV_MEM_CLOCK",
        - "DCGM_FI_DEV_MEM_COPY_UTIL",
        - "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL",
        - "DCGM_FI_DEV_PCIE_REPLAY_COUNTER",
        - "DCGM_FI_DEV_POWER_USAGE",
        - "DCGM_FI_DEV_ROW_REMAP_FAILURE",
        - "DCGM_FI_DEV_SM_CLOCK",
        - "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
        - "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS",
        - "DCGM_FI_DEV_VGPU_LICENSE_STATUS",
        - "DCGM_FI_DEV_XID_ERRORS",
    """
    return

def get_network_metrics_gwdg():
    """
    Planned return: network metrics for a node and UTC time range.

    Includes network interfaces, InfiniBand ports, protocol statistics,
    connection tracking, socket usage, UDP queues, and kernel packet
    processing. An optional interface filter applies to interface/port
    series; connection tracking and socket statistics are node-wide,
    and softnet series are per CPU.

    Source: per-incident *_tidy.csv.xz telemetry, sampled every 10 seconds.

    Preserve timestamps, values, seriesId, and distinguishing labels.
    Availability varies by incident; missing scrapes are absent rows.

    Metric names:
        - "node_infiniband_excessive_buffer_overrun_errors_total",
        - "node_infiniband_info",
        - "node_infiniband_link_downed_total",
        - "node_infiniband_link_error_recovery_total",
        - "node_infiniband_local_link_integrity_errors_total",
        - "node_infiniband_multicast_packets_received_total",
        - "node_infiniband_multicast_packets_transmitted_total",
        - "node_infiniband_physical_state_id",
        - "node_infiniband_port_constraint_errors_received_total",
        - "node_infiniband_port_constraint_errors_transmitted_total",
        - "node_infiniband_port_data_received_bytes_total",
        - "node_infiniband_port_data_transmitted_bytes_total",
        - "node_infiniband_port_discards_transmitted_total",
        - "node_infiniband_port_errors_received_total",
        - "node_infiniband_port_packets_received_total",
        - "node_infiniband_port_packets_transmitted_total",
        - "node_infiniband_port_receive_remote_physical_errors_total",
        - "node_infiniband_port_receive_switch_relay_errors_total",
        - "node_infiniband_port_transmit_wait_total",
        - "node_infiniband_rate_bytes_per_second",
        - "node_infiniband_state_id",
        - "node_infiniband_symbol_error_total",
        - "node_infiniband_unicast_packets_received_total",
        - "node_infiniband_unicast_packets_transmitted_total",
        - "node_infiniband_vl15_dropped_total",
        - "node_netstat_Icmp6_InErrors",
        - "node_netstat_Icmp6_InMsgs",
        - "node_netstat_Icmp6_OutMsgs",
        - "node_netstat_Icmp_InErrors",
        - "node_netstat_Icmp_InMsgs",
        - "node_netstat_Icmp_OutMsgs",
        - "node_netstat_Ip6_InOctets",
        - "node_netstat_Ip6_OutOctets",
        - "node_netstat_IpExt_InOctets",
        - "node_netstat_IpExt_OutOctets",
        - "node_netstat_Ip_Forwarding",
        - "node_netstat_TcpExt_ListenDrops",
        - "node_netstat_TcpExt_ListenOverflows",
        - "node_netstat_TcpExt_SyncookiesFailed",
        - "node_netstat_TcpExt_SyncookiesRecv",
        - "node_netstat_TcpExt_SyncookiesSent",
        - "node_netstat_TcpExt_TCPSynRetrans",
        - "node_netstat_TcpExt_TCPTimeouts",
        - "node_netstat_Tcp_ActiveOpens",
        - "node_netstat_Tcp_CurrEstab",
        - "node_netstat_Tcp_InErrs",
        - "node_netstat_Tcp_InSegs",
        - "node_netstat_Tcp_OutRsts",
        - "node_netstat_Tcp_OutSegs",
        - "node_netstat_Tcp_PassiveOpens",
        - "node_netstat_Tcp_RetransSegs",
        - "node_netstat_Udp6_InDatagrams",
        - "node_netstat_Udp6_InErrors",
        - "node_netstat_Udp6_NoPorts",
        - "node_netstat_Udp6_OutDatagrams",
        - "node_netstat_Udp6_RcvbufErrors",
        - "node_netstat_Udp6_SndbufErrors",
        - "node_netstat_UdpLite6_InErrors",
        - "node_netstat_UdpLite_InErrors",
        - "node_netstat_Udp_InDatagrams",
        - "node_netstat_Udp_InErrors",
        - "node_netstat_Udp_NoPorts",
        - "node_netstat_Udp_OutDatagrams",
        - "node_netstat_Udp_RcvbufErrors",
        - "node_netstat_Udp_SndbufErrors",
        - "node_network_address_assign_type",
        - "node_network_carrier",
        - "node_network_carrier_changes_total",
        - "node_network_carrier_down_changes_total",
        - "node_network_carrier_up_changes_total",
        - "node_network_device_id",
        - "node_network_dormant",
        - "node_network_flags",
        - "node_network_iface_id",
        - "node_network_iface_link",
        - "node_network_iface_link_mode",
        - "node_network_info",
        - "node_network_mtu_bytes",
        - "node_network_name_assign_type",
        - "node_network_net_dev_group",
        - "node_network_protocol_type",
        - "node_network_receive_bytes_total",
        - "node_network_receive_compressed_total",
        - "node_network_receive_drop_total",
        - "node_network_receive_errs_total",
        - "node_network_receive_fifo_total",
        - "node_network_receive_frame_total",
        - "node_network_receive_multicast_total",
        - "node_network_receive_packets_total",
        - "node_network_speed_bytes",
        - "node_network_transmit_bytes_total",
        - "node_network_transmit_carrier_total",
        - "node_network_transmit_colls_total",
        - "node_network_transmit_compressed_total",
        - "node_network_transmit_drop_total",
        - "node_network_transmit_errs_total",
        - "node_network_transmit_fifo_total",
        - "node_network_transmit_packets_total",
        - "node_network_transmit_queue_length",
        - "node_network_up",
        - "node_nf_conntrack_entries",
        - "node_nf_conntrack_entries_limit",
        - "node_nf_conntrack_stat_drop",
        - "node_nf_conntrack_stat_early_drop",
        - "node_nf_conntrack_stat_found",
        - "node_nf_conntrack_stat_ignore",
        - "node_nf_conntrack_stat_insert",
        - "node_nf_conntrack_stat_insert_failed",
        - "node_nf_conntrack_stat_invalid",
        - "node_nf_conntrack_stat_search_restart",
        - "node_sockstat_FRAG6_inuse",
        - "node_sockstat_FRAG6_memory",
        - "node_sockstat_FRAG_inuse",
        - "node_sockstat_FRAG_memory",
        - "node_sockstat_RAW6_inuse",
        - "node_sockstat_RAW_inuse",
        - "node_sockstat_TCP6_inuse",
        - "node_sockstat_TCP_alloc",
        - "node_sockstat_TCP_inuse",
        - "node_sockstat_TCP_mem",
        - "node_sockstat_TCP_mem_bytes",
        - "node_sockstat_TCP_orphan",
        - "node_sockstat_TCP_tw",
        - "node_sockstat_UDP6_inuse",
        - "node_sockstat_UDPLITE6_inuse",
        - "node_sockstat_UDPLITE_inuse",
        - "node_sockstat_UDP_inuse",
        - "node_sockstat_UDP_mem",
        - "node_sockstat_UDP_mem_bytes",
        - "node_sockstat_sockets_used",
        - "node_softnet_backlog_len",
        - "node_softnet_cpu_collision_total",
        - "node_softnet_dropped_total",
        - "node_softnet_flow_limit_count_total",
        - "node_softnet_processed_total",
        - "node_softnet_received_rps_total",
        - "node_softnet_times_squeezed_total",
        - "node_udp_queues",
    """
    return

def get_storage_metrics_gwdg():
    """
    Planned return: storage and I/O metrics for a node and UTC time range.

    Includes disk I/O, filesystem capacity and state, NVMe information,
    and NFS client activity. Optional device or series filters select
    specific resources where labels are available. NFS counters describe
    activity observed by this node; filesystem paths are sanitized.

    Source: per-incident *_tidy.csv.xz telemetry, sampled every 10 seconds.

    Preserve timestamps, values, seriesId, and distinguishing labels.
    Availability varies by incident; missing scrapes are absent rows.

    Metric names:
        - "node_disk_discard_time_seconds_total",
        - "node_disk_discarded_sectors_total",
        - "node_disk_discards_completed_total",
        - "node_disk_discards_merged_total",
        - "node_disk_filesystem_info",
        - "node_disk_info",
        - "node_disk_io_now",
        - "node_disk_io_time_seconds_total",
        - "node_disk_io_time_weighted_seconds_total",
        - "node_disk_read_bytes_total",
        - "node_disk_read_time_seconds_total",
        - "node_disk_reads_completed_total",
        - "node_disk_reads_merged_total",
        - "node_disk_write_time_seconds_total",
        - "node_disk_writes_completed_total",
        - "node_disk_writes_merged_total",
        - "node_disk_written_bytes_total",
        - "node_filesystem_avail_bytes",
        - "node_filesystem_device_error",
        - "node_filesystem_files",
        - "node_filesystem_files_free",
        - "node_filesystem_free_bytes",
        - "node_filesystem_readonly",
        - "node_filesystem_size_bytes",
        - "node_nfs_connections_total",
        - "node_nfs_packets_total",
        - "node_nfs_requests_total",
        - "node_nfs_rpc_authentication_refreshes_total",
        - "node_nfs_rpc_retransmissions_total",
        - "node_nfs_rpcs_total",
        - "node_nvme_info",
    """
    return

def get_scheduler_metrics_gwdg():
    """
    Planned return: Slurm scheduler state for a node and UTC time range.

    Source: slurm_node_state_tidy.csv.xz, sampled every 10 minutes.
    This separate table has a node column and must be filtered by node.
    The incident telemetry file does not contain these metrics.
    slurm_total_gpus: GPU count known to the scheduler.
    slurm_alloc_gpus: GPUs allocated to jobs.
    slurm_status_ok: 1 when the node is in a healthy scheduler state.

    Preserve timestamps, values, seriesId, and distinguishing labels.
    Node coverage varies; missing scheduler observations are absent rows.

    Metric names:
        - "slurm_total_gpus",
        - "slurm_alloc_gpus",
        - "slurm_status_ok",
    """
    return

def get_monitoring_metrics_gwdg():
    """
    Planned return: monitoring pipeline metrics for a node and UTC time range.

    Includes exporter information, collector success, scrape durations,
    sample and series counts, and IPMI collection status. ipmi_up reports
    IPMI scrape success; chassis power state is a node hardware metric.
    Preserve job and collector labels to distinguish monitoring sources.

    Source: per-incident *_tidy.csv.xz telemetry, sampled every 10 seconds.

    Preserve timestamps, values, seriesId, and distinguishing labels.
    Availability varies by incident; missing scrapes are absent rows.

    Metric names:
        - "ipmi_scrape_duration_seconds",
        - "ipmi_up",
        - "node_exporter_build_info",
        - "node_scrape_collector_duration_seconds",
        - "node_scrape_collector_success",
        - "node_textfile_scrape_error",
        - "promhttp_metric_handler_errors_total",
        - "scrape_duration_seconds",
        - "scrape_samples_post_metric_relabeling",
        - "scrape_samples_scraped",
        - "scrape_series_added",
        - "up",
    """
    return
