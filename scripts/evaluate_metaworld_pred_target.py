import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_flow_checkpoint import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot checkpoint prediction-vs-target curves on MetaWorld trajectories. "
            "Linear checkpoints are compared to linear progress; optical-flow checkpoints "
            "are compared to stored flow_progress targets."
        )
    )
    parser.add_argument(
        "--h5-path",
        default="datasets_rewind_scratch/metaworld_embeddings_eval.h5",
        help="MetaWorld embedding H5 to evaluate.",
    )
    parser.add_argument("--linear-checkpoints", nargs="*", default=[])
    parser.add_argument("--flow-checkpoints", nargs="*", default=[])
    parser.add_argument("--output-dir", default="metaworld_pred_target_eval")
    parser.add_argument("--max-groups", type=int, default=-1)
    parser.add_argument("--max-trajs-per-group", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint_path in args.linear_checkpoints:
        evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            h5_path=args.h5_path,
            output_dir=output_dir,
            max_groups=args.max_groups,
            max_trajs_per_group=args.max_trajs_per_group,
            target_type="linear",
        )

    for checkpoint_path in args.flow_checkpoints:
        evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            h5_path=args.h5_path,
            output_dir=output_dir,
            max_groups=args.max_groups,
            max_trajs_per_group=args.max_trajs_per_group,
            target_type="optical_flow",
        )

    if not args.linear_checkpoints and not args.flow_checkpoints:
        raise SystemExit("Pass at least one --linear-checkpoints or --flow-checkpoints path.")


if __name__ == "__main__":
    main()
