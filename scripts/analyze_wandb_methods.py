import argparse
import math
from datetime import datetime, timezone
from statistics import mean, pstdev

import wandb


DEFAULT_METHOD_GROUPS = {
    "rewind": ["rewind_iql_window-close-v2"],
    "rewind_freeze": ["rewind_freeze_iql_window-close-v2"],
    "flow": ["rewind_flow_iql_window-close-v2", "flow_window-close-v2"],
    "flow_freeze": ["rewind_flow_freeze_iql_window-close-v2", "flow_freeze_window-close-v2"],
}


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def _extract_metric_series(run, metric_name):
    values = []
    steps = []
    for row in run.scan_history(keys=[metric_name, "_step"]):
        value = _safe_float(row.get(metric_name))
        if value is None:
            continue
        values.append(value)
        steps.append(row.get("_step"))
    return steps, values


def _summarize_run(run, metric_name, last_n):
    steps, values = _extract_metric_series(run, metric_name)
    if not values:
        return None

    seed = (
        run.config.get("general_training", {}).get("seed")
        if isinstance(run.config.get("general_training"), dict)
        else None
    )
    if seed is None:
        seed = run.config.get("seed")

    final_value = values[-1]
    tail_values = values[-last_n:] if len(values) >= last_n else values
    return {
        "run_id": run.id,
        "run_name": run.name,
        "group": run.group,
        "state": run.state,
        "seed": seed,
        "num_points": len(values),
        "final_step": steps[-1] if steps else None,
        "final_value": final_value,
        "last_n_mean": mean(tail_values),
    }


def _aggregate_method(run_summaries):
    final_values = [item["final_value"] for item in run_summaries]
    tail_values = [item["last_n_mean"] for item in run_summaries]
    return {
        "num_runs": len(run_summaries),
        "final_mean": mean(final_values),
        "final_std": pstdev(final_values) if len(final_values) > 1 else 0.0,
        "last_n_mean": mean(tail_values),
        "last_n_std": pstdev(tail_values) if len(tail_values) > 1 else 0.0,
    }


def _parse_run_time(run):
    for attr in ("created_at", "heartbeat_at", "updated_at"):
        value = getattr(run, attr, None)
        if not value:
            continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
    return datetime.min.replace(tzinfo=timezone.utc)


def main():
    parser = argparse.ArgumentParser(description="Aggregate W&B eval metrics across methods and seeds.")
    parser.add_argument("--entity", required=True, help="W&B entity/team name")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--metric", default="eval/success_rate", help="Metric to aggregate")
    parser.add_argument("--last-n", type=int, default=10, help="Average over the last N logged points")
    parser.add_argument(
        "--method-group",
        action="append",
        default=[],
        help="Override mapping as method=group_name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Include runs that are still running. By default only finished runs are used.",
    )
    parser.add_argument(
        "--max-runs-per-method",
        type=int,
        default=6,
        help="Use only the most recent N matching runs per method before aggregation.",
    )
    args = parser.parse_args()

    method_groups = dict(DEFAULT_METHOD_GROUPS)
    for item in args.method_group:
        if "=" not in item:
            raise ValueError(f"Invalid --method-group value: {item}")
        method, group = item.split("=", 1)
        method_groups[method] = [value.strip() for value in group.split(",") if value.strip()]

    api = wandb.Api()
    repo = f"{args.entity}/{args.project}"

    print(f"Project: {repo}")
    print(f"Metric: {args.metric}")
    print(f"Last-N window: {args.last_n}")
    print(f"Max runs per method: {args.max_runs_per_method}")
    print("")

    method_results = {}

    for method, group_names in method_groups.items():
        collected_runs = {}
        for group_name in group_names:
            filters = {"group": group_name}
            for run in api.runs(repo, filters=filters):
                collected_runs[run.id] = run
        runs = list(collected_runs.values())
        if not args.include_running:
            runs = [run for run in runs if run.state == "finished"]
        runs = sorted(runs, key=_parse_run_time, reverse=True)[: args.max_runs_per_method]

        run_summaries = []
        for run in runs:
            summary = _summarize_run(run, args.metric, args.last_n)
            if summary is not None:
                run_summaries.append(summary)

        method_results[method] = {
            "group": ",".join(group_names),
            "runs": run_summaries,
            "aggregate": _aggregate_method(run_summaries) if run_summaries else None,
        }

    print("Method comparison")
    print("method\tgroup\tn_runs\tfinal_mean\tfinal_std\tlast_n_mean\tlast_n_std")
    for method, result in method_results.items():
        aggregate = result["aggregate"]
        if aggregate is None:
            print(f"{method}\t{result['group']}\t0\tNA\tNA\tNA\tNA")
            continue
        print(
            f"{method}\t{result['group']}\t{aggregate['num_runs']}\t"
            f"{aggregate['final_mean']:.4f}\t{aggregate['final_std']:.4f}\t"
            f"{aggregate['last_n_mean']:.4f}\t{aggregate['last_n_std']:.4f}"
        )

    print("")
    print("Per-run details")
    print("method\tseed\trun_name\tfinal_step\tfinal_value\tlast_n_mean")
    for method, result in method_results.items():
        for run_summary in sorted(result["runs"], key=lambda item: (str(item["seed"]), item["run_name"])):
            print(
                f"{method}\t{run_summary['seed']}\t{run_summary['run_name']}\t"
                f"{run_summary['final_step']}\t{run_summary['final_value']:.4f}\t"
                f"{run_summary['last_n_mean']:.4f}"
            )


if __name__ == "__main__":
    main()
