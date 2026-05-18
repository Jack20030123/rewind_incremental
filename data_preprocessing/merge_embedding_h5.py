import argparse
from pathlib import Path

import h5py


def is_trajectory_key(key):
    return (
        "lang" not in key
        and not key.startswith("flow_progress_")
        and not key.startswith("flow_signal_")
    )


def copy_dataset(src_group, dst_group, src_key, dst_key):
    if dst_key in dst_group:
        del dst_group[dst_key]
    src_group.copy(src_key, dst_group, name=dst_key)


def merge_file(src_path, dst_h5, require_flow):
    copied = 0
    with h5py.File(src_path, "r") as src_h5:
        for group_name, src_group in src_h5.items():
            if not isinstance(src_group, h5py.Group):
                continue

            dst_group = dst_h5.require_group(group_name)
            if "minilm_lang_embedding" in src_group and "minilm_lang_embedding" not in dst_group:
                copy_dataset(src_group, dst_group, "minilm_lang_embedding", "minilm_lang_embedding")

            next_id = sum(1 for key in dst_group.keys() if is_trajectory_key(key))
            for traj_key in src_group.keys():
                if not is_trajectory_key(traj_key):
                    continue

                flow_progress_key = f"flow_progress_{traj_key}"
                flow_signal_key = f"flow_signal_{traj_key}"
                has_flow = flow_progress_key in src_group and flow_signal_key in src_group
                if require_flow and not has_flow:
                    continue

                dst_traj_key = str(next_id)
                next_id += 1
                copy_dataset(src_group, dst_group, traj_key, dst_traj_key)
                if has_flow:
                    copy_dataset(src_group, dst_group, flow_progress_key, f"flow_progress_{dst_traj_key}")
                    copy_dataset(src_group, dst_group, flow_signal_key, f"flow_signal_{dst_traj_key}")
                copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(description="Merge ReWiND embedding H5 files.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--require-flow", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with h5py.File(output, "w") as dst_h5:
        for src in args.inputs:
            src_path = Path(src)
            try:
                copied = merge_file(src_path, dst_h5, args.require_flow)
                total += copied
                print(f"{src_path.name}: copied {copied} trajectories")
            except Exception as exc:
                if not args.skip_errors:
                    raise
                print(f"{src_path.name}: skipped ({exc})")

    print(f"Wrote {total} trajectories to {output}")


if __name__ == "__main__":
    main()
