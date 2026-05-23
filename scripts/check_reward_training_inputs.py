import argparse
import os
from pathlib import Path

import h5py


def summarize(path):
    path = Path(path)
    resolved = Path(os.path.realpath(path))
    if not path.exists():
        return {"path": str(path), "exists": False}

    groups = trajectories = flow = 0
    examples = []
    with h5py.File(path, "r") as h5_file:
        for group_name, group in h5_file.items():
            if not isinstance(group, h5py.Group):
                continue
            groups += 1
            if len(examples) < 5:
                examples.append(group_name)
            trajs = [
                key
                for key in group.keys()
                if "lang" not in key
                and not key.startswith("flow_progress_")
                and not key.startswith("flow_signal_")
            ]
            trajectories += len(trajs)
            flow += sum(
                f"flow_progress_{traj}" in group
                and f"flow_signal_{traj}" in group
                for traj in trajs
            )
    return {
        "path": str(path),
        "resolved": str(resolved),
        "exists": True,
        "groups": groups,
        "trajectories": trajectories,
        "flow": flow,
        "examples": examples,
    }


def print_summary(label, info):
    if not info["exists"]:
        print(f"{label}: MISSING {info['path']}")
        return
    print(
        f"{label}: {info['path']} -> {info['resolved']} "
        f"groups={info['groups']} trajs={info['trajectories']} "
        f"flow={info['flow']}/{info['trajectories']} examples={info['examples']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-folder-path", default="datasets")
    parser.add_argument(
        "--openx-embedding-path",
        default="datasets/full_openx_embeddings_v2_train.h5",
    )
    parser.add_argument(
        "--flow-openx-embedding-path",
        default="datasets/full_openx_flow_embeddings_train.h5",
    )
    args = parser.parse_args()

    h5_folder = Path(args.h5_folder_path)
    paths = {
        "MetaWorld train": h5_folder / "metaworld_embeddings_train.h5",
        "MetaWorld eval": h5_folder / "metaworld_embeddings_eval.h5",
        "MetaWorld close success": h5_folder / "metaworld_dino_embeddings_eval_close_succ.h5",
        "MetaWorld all fail": h5_folder / "metaworld_dino_embeddings_eval_all_fail.h5",
        "OpenX DINO": Path(args.openx_embedding_path),
        "OpenX flow": Path(args.flow_openx_embedding_path),
    }

    summaries = {label: summarize(path) for label, path in paths.items()}
    for label, info in summaries.items():
        print_summary(label, info)

    train = summaries["MetaWorld train"]
    openx = summaries["OpenX DINO"]
    if train["exists"] and openx["exists"] and train["resolved"] == openx["resolved"]:
        print(
            "ERROR: MetaWorld train resolves to the OpenX embedding file. "
            "Regenerate MetaWorld embeddings before training."
        )

    flow_openx = summaries["OpenX flow"]
    if flow_openx["exists"] and flow_openx["flow"] != flow_openx["trajectories"]:
        print("ERROR: OpenX flow file has incomplete flow coverage.")


if __name__ == "__main__":
    main()
