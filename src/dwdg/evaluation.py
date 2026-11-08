"""Batch evaluation of the GWDG diagnosis agent in test-agent-dwdg.py.

Incident cases come from incident_events.csv rows with telemetryStatus
"included"; each telemetry file's _meta.json supplies the collection window.
Healthy cases (negative controls) are windows inside those same files, ending
before the Slurm previous-healthy time, in which slurm_status_ok stays 1.

Every case runs against a GwdgDataSource bounded to its window, so a healthy
case cannot read the later failure. Runs execute in a process pool because the
xz/pandas scans are CPU-bound. Model calls use a scripted mock client unless
--live is given.

    ai-infra-agent/bin/python -m src.dwdg.evaluation --limit 3 --negatives 2
    ai-infra-agent/bin/python -m src.dwdg.evaluation --live --workers 3

Outputs go to runs/<timestamp>/: cases/<case_id>.json (case, score, report),
logs/<case_id>.log, summary.csv and summary.md.
"""

import argparse
import csv
import json
import logging
import os
import runpy
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import cache
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.dwdg.telemetry import DATA_DIR


ROOT = Path(__file__).resolve().parents[2]
AGENT_PATH = ROOT / "test-agent-dwdg.py"
RUNS_DIR = ROOT / "runs"
SLURM_FILE = "slurm_node_state_tidy.csv.xz"
SLURM_INTERVAL = pd.Timedelta(seconds=600)
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

PROMPT_TEMPLATE = """Investigate a possible problem on GPU node {node}. All times are UTC.
A problem was reported at {reported}. Slurm last confirmed the node healthy at {last_healthy}.
Telemetry for this investigation covers {start} to {end}; requests outside that window are rejected.
Start with resource discovery and a compact health overview, then decide follow-up tools based on the evidence.
Determine whether a fault occurred and which component is responsible. Report onset_time as the earliest
UTC time at which the fault is visible in the evidence, or null if the evidence shows no fault.
Cite returned evidence IDs. Distinguish a suspected cause from a confirmed cause.
If the evidence shows no fault, or cannot distinguish the causes, report unknown and identify the next useful check.
"""

SUMMARY_FIELDS = (
    "case_id", "kind", "node", "expected_component", "suspected_component", "correct",
    "severity", "confidence", "onset_time", "onset_in_bracket", "onset_error_minutes",
    "onset_echoes_prompt", "bracket_starts_before_window", "status", "stopping_reason",
    "model_runs", "tool_attempts", "tools_call_count", "query_too_broad", "tool_errors",
    "investigation_errors", "wall_seconds",
)


@dataclass
class EvalCase:
    case_id: str
    kind: str  # "incident" or "healthy"
    node: str
    telemetry_file: str
    window_start: str
    window_end: str
    reported_time: str
    last_healthy_time: str
    expected_component: str
    # Ground-truth onset bracket for incidents; None for healthy windows.
    onset_bracket: tuple | None
    prompt: str


def utc(value) -> pd.Timestamp:
    """Parse a dataset time; naive values are UTC, as SCHEMA.md specifies."""
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def iso(value) -> str:
    return utc(value).strftime(TIME_FORMAT)


def build_prompt(node, reported, last_healthy, start, end) -> str:
    return PROMPT_TEMPLATE.format(node=node, reported=iso(reported), last_healthy=iso(last_healthy),
                                  start=iso(start), end=iso(end))


def load_incident_cases(data_dir: Path = DATA_DIR) -> list[EvalCase]:
    """Included incidents, in catalog order; the operator description never reaches the prompt."""
    events = pd.read_csv(Path(data_dir) / "incident_events.csv", dtype=str, na_filter=False)
    cases = []
    for row in events.loc[events["telemetryStatus"] == "included"].itertuples(index=False):
        meta_path = Path(data_dir) / row.telemetryFile.replace("_tidy.csv.xz", "_meta.json")
        meta = json.loads(meta_path.read_text())
        start, end = utc(meta["collectStart"]), utc(meta["collectEnd"])
        healthy, detected = utc(row.slurmPreviousHealthyTime), utc(row.slurmDetectedTime)
        cases.append(EvalCase(
            case_id=row.telemetryFile.removesuffix("_tidy.csv.xz"), kind="incident", node=row.node,
            telemetry_file=row.telemetryFile, window_start=iso(start), window_end=iso(end),
            reported_time=iso(detected), last_healthy_time=iso(healthy), expected_component="gpu",
            onset_bracket=(iso(healthy), iso(detected)),
            prompt=build_prompt(row.node, detected, healthy, start, end),
        ))
    return cases


def load_slurm_status(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """slurm_status_ok rows as node, time (UTC) and value, sorted per node."""
    frame = pd.read_csv(Path(data_dir) / SLURM_FILE, dtype=str, na_filter=False,
                        usecols=["timeUtc", "node", "metric", "value"])
    frame = frame.loc[frame["metric"] == "slurm_status_ok"]
    return pd.DataFrame({
        "node": frame["node"],
        "time": pd.to_datetime(frame["timeUtc"], utc=True),
        "value": pd.to_numeric(frame["value"], errors="coerce"),
    }).sort_values(["node", "time"], ignore_index=True)


def longest_healthy_run(status: pd.DataFrame, start, end):
    """Longest [first, last] run of consecutive 600 s samples with value 1 inside [start, end]."""
    rows = status.loc[(status["time"] >= start) & (status["time"] <= end)]
    best, run_start, previous = None, None, None
    for time_value, value in zip(rows["time"], rows["value"]):
        if value != 1:
            run_start = None
        elif run_start is None or time_value - previous != SLURM_INTERVAL:
            run_start = time_value
        if run_start is not None and (best is None or time_value - run_start > best[1] - best[0]):
            best = (run_start, time_value)
        previous = time_value
    return best


def select_healthy_cases(incidents: list[EvalCase], slurm_status: pd.DataFrame, count: int, *,
                         min_hours: float = 4, max_hours: float = 24, margin_hours: float = 1) -> list[EvalCase]:
    """Pick up to count healthy windows, one per node, longest first.

    A window lies inside an incident file's collection window, ends at least
    margin_hours before that incident's previous-healthy time, and every Slurm
    sample in it (10-minute cadence, no gaps) reports slurm_status_ok == 1.
    """
    candidates = {}
    for case in incidents:
        healthy = utc(case.last_healthy_time)
        start, end = utc(case.window_start), min(utc(case.window_end), healthy - pd.Timedelta(hours=margin_hours))
        if end - start < pd.Timedelta(hours=min_hours):
            continue
        run = longest_healthy_run(slurm_status.loc[slurm_status["node"] == case.node], start, end)
        if run is None or run[1] - run[0] < pd.Timedelta(hours=min_hours):
            continue
        run_start, run_end = run[0], min(run[1], run[0] + pd.Timedelta(hours=max_hours))
        best = candidates.get(case.node)
        if best is None or run_end - run_start > best[2] - best[1]:
            candidates[case.node] = (case, run_start, run_end)

    ranked = sorted(candidates.values(), key=lambda item: (-(item[2] - item[1]), item[0].node))
    healthy_cases = []
    for case, start, end in ranked[:max(0, count)]:
        # Mirror the incident layout: the report falls one hour before the window end.
        reported = end - pd.Timedelta(hours=1)
        last_healthy = max(start, reported - pd.Timedelta(hours=12))
        healthy_cases.append(EvalCase(
            case_id=f"healthy_{case.node}_{start.strftime('%Y-%m-%dT%H%M')}", kind="healthy", node=case.node,
            telemetry_file=case.telemetry_file, window_start=iso(start), window_end=iso(end),
            reported_time=iso(reported), last_healthy_time=iso(last_healthy), expected_component="unknown",
            onset_bracket=None, prompt=build_prompt(case.node, reported, last_healthy, start, end),
        ))
    return healthy_cases


def parse_onset(value):
    """Return (timestamp or None, error message or None)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, None
    try:
        return utc(value), None
    except (ValueError, TypeError) as error:
        return None, f"unparseable onset_time {value!r}: {error}"[:200]


def score_run(case: EvalCase, report: dict, wall_seconds: float) -> dict:
    diagnosis = report.get("diagnosis") if isinstance(report.get("diagnosis"), dict) else {}
    component = diagnosis.get("suspected_component")
    onset, onset_error = parse_onset(diagnosis.get("onset_time"))
    trace = report.get("tool_trace") or []
    statuses = [entry.get("output", {}).get("status") for entry in trace]
    error_codes = Counter(
        entry["output"].get("error", {}).get("code", "unknown")
        for entry in trace if entry.get("output", {}).get("status") == "error"
    )

    onset_in_bracket = onset_error_minutes = onset_echoes_prompt = bracket_before_window = None
    if case.onset_bracket is not None:
        low, high = (utc(value) for value in case.onset_bracket)
        bracket_before_window = low < utc(case.window_start)
        if onset is not None:
            onset_in_bracket = bool(low <= onset <= high)
            # Signed distance outside the bracket: negative is too early, positive too late.
            if onset < low:
                onset_error_minutes = round((onset - low).total_seconds() / 60, 1)
            elif onset > high:
                onset_error_minutes = round((onset - high).total_seconds() / 60, 1)
            else:
                onset_error_minutes = 0.0
            # Echoing a prompt time lands inside the bracket without finding anything.
            onset_echoes_prompt = onset in (low, high)
        else:
            onset_in_bracket = False

    stopping = report.get("stopping_reason")
    return {
        "case_id": case.case_id, "kind": case.kind, "node": case.node,
        "expected_component": case.expected_component, "suspected_component": component,
        "correct": component == case.expected_component,
        "severity": report.get("severity"), "confidence": report.get("confidence"),
        "onset_time": onset.strftime(TIME_FORMAT) if onset is not None else None,
        "onset_parse_error": onset_error,
        "onset_in_bracket": onset_in_bracket, "onset_error_minutes": onset_error_minutes,
        "onset_echoes_prompt": onset_echoes_prompt, "bracket_starts_before_window": bracket_before_window,
        "status": report.get("status"),
        "stopping_reason": stopping.get("code") if isinstance(stopping, dict) else None,
        "model_runs": report.get("model_runs"), "tool_attempts": report.get("tool_attempts"),
        "tools_call_count": report.get("tools_call_count", len(trace)),
        "query_too_broad": statuses.count("query_too_broad"),
        "tool_errors": statuses.count("error"), "tool_error_codes": dict(error_codes),
        "investigation_errors": len(report.get("investigation_errors") or []),
        "wall_seconds": round(wall_seconds, 1),
    }


class ScriptedResponses:
    """Offline stand-in for client.responses: discovery, overview, then an unknown report.

    It exercises the real tools and data scans without any API call. With
    use_tools=False it answers immediately.
    """

    def __init__(self, case: EvalCase, use_tools: bool = True):
        self.case = case
        self.plan = ["get_resource_ids_gwdg", "get_overview_gwdg"] if use_tools else []
        self.calls = 0

    def create(self, *, tool_choice="auto", **_):
        self.calls += 1
        window = {"start_time": self.case.window_start, "end_time": self.case.window_end}
        if self.plan and tool_choice != "none":
            name = self.plan.pop(0)
            arguments = window if name == "get_resource_ids_gwdg" else {"node_id": self.case.node, **window}
            item = SimpleNamespace(type="function_call", name=name, arguments=json.dumps(arguments),
                                   call_id=f"mock-{self.calls}")
            item.model_dump = lambda mode="json", item=item: {
                key: value for key, value in vars(item).items() if key != "model_dump"}
            return SimpleNamespace(status="completed", output=[item], output_text="")
        report = {
            "severity": "unknown", "confidence": 0,
            "diagnosis": {
                "suspected_component": "unknown", "summary": "Mock client: no diagnosis was attempted.",
                "onset_time": None, "supporting_evidence_ids": [],
                "missing_evidence": ["A live model run"], "next_check": "Run with --live.",
            },
            "next_action": "Run the evaluation with --live.",
        }
        return SimpleNamespace(status="completed", output=[], output_text=json.dumps(report))


class ScriptedClient:
    def __init__(self, case: EvalCase, use_tools: bool = True):
        self.responses = ScriptedResponses(case, use_tools)

    def close(self):
        pass


@cache
def load_agent() -> dict:
    """Load test-agent-dwdg.py once per process (its file name is not importable)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return runpy.run_path(str(AGENT_PATH))


def run_case(case: EvalCase, options: dict) -> dict:
    """Run and score one case, writing its JSON; never raises for ordinary failures."""
    run_dir = Path(options["run_dir"])
    log_handler = logging.FileHandler(run_dir / "logs" / f"{case.case_id}.log")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    root_logger.setLevel(options.get("log_level", "INFO"))
    for noisy in ("openai", "httpx", "httpcore", "httpcore2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    started = time.perf_counter()
    try:
        agent = load_agent()
        api_client = None if options.get("live") else ScriptedClient(case, options.get("mock_tools", True))
        kwargs = dict(
            telemetry_file=case.telemetry_file, visible_window=(case.window_start, case.window_end),
            api_client=api_client, max_model_runs=options.get("max_model_runs", 5),
            max_tool_calls=options.get("max_tool_calls", 80),
        )
        if options.get("model"):
            kwargs["model"] = options["model"]
        report = agent["diagnose_cluster_with_fixture"](case.case_id, case.prompt, **kwargs)
    except Exception as error:
        logging.getLogger(__name__).exception("Harness failure for %s", case.case_id)
        report = {"status": "harness_error", "stopping_reason": {"code": "harness_error", "message": repr(error)[:512]}}
    finally:
        root_logger.removeHandler(log_handler)
        log_handler.close()
    score = score_run(case, report, time.perf_counter() - started)
    record = {"case": asdict(case), "score": score, "report": report}
    (run_dir / "cases" / f"{case.case_id}.json").write_text(json.dumps(record, indent=2, default=str))
    return score


def run_batch(cases: list[EvalCase], options: dict, *, workers: int = 1, progress=None) -> list[dict]:
    """Run cases inline (workers == 1) or in a spawn-based process pool; return scores in case order."""
    run_dir = Path(options["run_dir"])
    (run_dir / "cases").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    order = {case.case_id: index for index, case in enumerate(cases)}
    scores = []
    if workers <= 1:
        for case in cases:
            scores.append(run_case(case, options))
            if progress:
                progress(scores[-1], len(scores), len(cases))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(run_case, case, options): case for case in cases}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    score = future.result()
                except Exception as error:  # e.g. a worker killed by the OS
                    score = score_run(case, {"status": "harness_error", "stopping_reason": {
                        "code": "worker_failed", "message": repr(error)[:512]}}, 0.0)
                scores.append(score)
                if progress:
                    progress(score, len(scores), len(cases))
    return sorted(scores, key=lambda score: order[score["case_id"]])


def _rate(values):
    values = [value for value in values if value is not None]
    return (sum(bool(value) for value in values), len(values))


def _fmt_rate(pair):
    hits, total = pair
    return f"{hits}/{total} ({hits / total:.0%})" if total else "n/a"


def _mean(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return round(sum(values) / len(values), 1) if values else None


def summarize(scores: list[dict]) -> dict:
    incidents = [score for score in scores if score["kind"] == "incident"]
    healthy = [score for score in scores if score["kind"] == "healthy"]
    return {
        "runs": len(scores),
        "incident_component_accuracy": _rate(score["correct"] for score in incidents),
        "incident_onset_reported": _rate(score["onset_time"] is not None for score in incidents),
        "incident_onset_in_bracket": _rate(score["onset_in_bracket"] for score in incidents),
        "incident_onset_echoes_prompt": _rate(score["onset_echoes_prompt"] for score in incidents
                                              if score["onset_time"] is not None),
        "healthy_correct_unknown": _rate(score["correct"] for score in healthy),
        "healthy_false_onset": _rate(score["onset_time"] is not None for score in healthy),
        "completed": _rate(score["status"] == "completed" for score in scores),
        "mean_model_runs": _mean(score["model_runs"] for score in scores),
        "mean_tool_attempts": _mean(score["tool_attempts"] for score in scores),
        "mean_wall_seconds": _mean(score["wall_seconds"] for score in scores),
        "total_query_too_broad": sum(score["query_too_broad"] for score in scores),
        "total_tool_errors": sum(score["tool_errors"] for score in scores),
        "total_investigation_errors": sum(score["investigation_errors"] for score in scores),
        "stopping_reasons": dict(Counter(score["stopping_reason"] for score in scores if score["stopping_reason"])),
    }


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace("|", "\\|")


def render_markdown(scores: list[dict], summary: dict, config: dict) -> str:
    lines = [
        f"# GWDG agent evaluation ({'live' if config.get('live') else 'mock client'})", "",
        "| metric | value |", "|---|---|",
        f"| incident component accuracy (gpu) | {_fmt_rate(summary['incident_component_accuracy'])} |",
        f"| incident onset reported | {_fmt_rate(summary['incident_onset_reported'])} |",
        f"| incident onset inside Slurm bracket | {_fmt_rate(summary['incident_onset_in_bracket'])} |",
        f"| onset equals a prompt time (of reported) | {_fmt_rate(summary['incident_onset_echoes_prompt'])} |",
        f"| healthy windows answered unknown | {_fmt_rate(summary['healthy_correct_unknown'])} |",
        f"| healthy windows with an onset | {_fmt_rate(summary['healthy_false_onset'])} |",
        f"| completed runs | {_fmt_rate(summary['completed'])} |",
        f"| mean model runs / tool attempts | {summary['mean_model_runs']} / {summary['mean_tool_attempts']} |",
        f"| mean wall seconds | {summary['mean_wall_seconds']} |",
        f"| query_too_broad / tool errors / report errors | {summary['total_query_too_broad']} / "
        f"{summary['total_tool_errors']} / {summary['total_investigation_errors']} |",
        f"| stopping reasons | {_cell(json.dumps(summary['stopping_reasons'])) or '-'} |",
        "",
        "| case | kind | expected | answer | ok | onset | in bracket | error min | status | runs | tools | broad | errors | wall s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for score in scores:
        lines.append("| " + " | ".join(_cell(value) for value in (
            score["case_id"], score["kind"], score["expected_component"], score["suspected_component"],
            score["correct"], score["onset_time"], score["onset_in_bracket"], score["onset_error_minutes"],
            score["status"] if not score["stopping_reason"] else f"{score['status']} ({score['stopping_reason']})",
            score["model_runs"], score["tool_attempts"], score["query_too_broad"], score["tool_errors"],
            score["wall_seconds"],
        )) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(run_dir: Path, scores: list[dict], config: dict) -> str:
    summary = summarize(scores)
    with (run_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scores)
    (run_dir / "summary.json").write_text(json.dumps({"config": config, "summary": summary}, indent=2))
    markdown = render_markdown(scores, summary, config)
    (run_dir / "summary.md").write_text(markdown)
    return markdown


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N included incidents.")
    parser.add_argument("--negatives", type=int, default=3, help="Healthy-window controls to add (0 disables).")
    parser.add_argument("--cases", default=None, help="Comma-separated case IDs to run (after building all cases).")
    parser.add_argument("--workers", type=int, default=2,
                        help="Worker processes; each can hold a full incident window (several GB) in memory.")
    parser.add_argument("--live", action="store_true", help="Call the model API; without it a scripted mock client is used.")
    parser.add_argument("--mock-no-tools", action="store_true", help="Mock client answers at once, skipping data scans.")
    parser.add_argument("--model", default=None, help="Model name for live runs (default: the agent's).")
    parser.add_argument("--max-model-runs", type=int, default=5)
    parser.add_argument("--max-tool-calls", type=int, default=80)
    parser.add_argument("--min-healthy-hours", type=float, default=4)
    parser.add_argument("--out", type=Path, default=RUNS_DIR, help="Parent directory for run outputs.")
    parser.add_argument("--log-level", default="INFO", help="Level for the per-case log files.")
    return parser.parse_args(argv)


def build_cases(args) -> list[EvalCase]:
    incidents = load_incident_cases()
    cases = list(incidents)
    if args.negatives > 0:
        cases += select_healthy_cases(incidents, load_slurm_status(), args.negatives,
                                      min_hours=args.min_healthy_hours)
    if args.cases:
        wanted = [value.strip() for value in args.cases.split(",") if value.strip()]
        known = {case.case_id for case in cases}
        missing = [value for value in wanted if value not in known]
        if missing:
            raise SystemExit(f"Unknown case IDs: {', '.join(missing)}")
        cases = [case for case in cases if case.case_id in wanted]
    if args.limit is not None:
        kept = [case for case in cases if case.kind == "incident"][:max(0, args.limit)]
        cases = kept + [case for case in cases if case.kind == "healthy"]
    return cases


def main(argv=None) -> int:
    args = parse_args(argv)
    # Console logging follows the agent's AGENT_LOG_LEVEL; per-case files use --log-level.
    load_agent()["configure_logging"]()
    if args.live and not os.environ.get("OPENAI_API_KEY"):
        print("warning: OPENAI_API_KEY is not set; live runs will end as model_unavailable", file=sys.stderr)
    cases = build_cases(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / f"{stamp}-{'live' if args.live else 'mock'}"
    config = {
        "live": args.live, "model": args.model, "workers": args.workers, "limit": args.limit,
        "negatives": args.negatives, "max_model_runs": args.max_model_runs,
        "max_tool_calls": args.max_tool_calls, "case_ids": [case.case_id for case in cases],
    }
    options = {
        "run_dir": str(run_dir), "live": args.live, "mock_tools": not args.mock_no_tools,
        "model": args.model, "max_model_runs": args.max_model_runs,
        "max_tool_calls": args.max_tool_calls, "log_level": args.log_level.upper(),
    }
    print(f"Running {len(cases)} cases ({sum(c.kind == 'healthy' for c in cases)} healthy) "
          f"with {args.workers} workers -> {run_dir}", file=sys.stderr)

    def progress(score, done, total):
        print(f"[{done}/{total}] {score['case_id']}: {score['suspected_component']} "
              f"({'ok' if score['correct'] else 'wrong'}), {score['status']}, {score['wall_seconds']} s",
              file=sys.stderr)

    scores = run_batch(cases, options, workers=args.workers, progress=progress)
    print(write_outputs(run_dir, scores, config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
