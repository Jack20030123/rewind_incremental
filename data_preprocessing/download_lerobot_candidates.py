import argparse
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download


CANDIDATES = {
    "dobbe": [
        "IPEC-COMMUNITY/dobbe_lerobot",
        "FedorX8/dobbe_lerobot",
        "lerobot/dobbe",
    ],
    "roboset": [
        "FedorX8/roboset_lerobot",
        "IPEC-COMMUNITY/roboset_lerobot",
        "lerobot/roboset",
    ],
    "rh20t": [
        "FedorX8/rh20t_lerobot",
        "IPEC-COMMUNITY/rh20t_lerobot",
        "lerobot/rh20t",
    ],
    "tdroid_carrot_in_bowl": [
        "FedorX8/tdroid_carrot_in_bowl_lerobot",
        "IPEC-COMMUNITY/tdroid_carrot_in_bowl_lerobot",
        "lerobot/tdroid_carrot_in_bowl",
    ],
    "tdroid_pour_corn_in_pot": [
        "FedorX8/tdroid_pour_corn_in_pot_lerobot",
        "IPEC-COMMUNITY/tdroid_pour_corn_in_pot_lerobot",
        "lerobot/tdroid_pour_corn_in_pot",
    ],
    "tdroid_flip_pot_upright": [
        "FedorX8/tdroid_flip_pot_upright_lerobot",
        "IPEC-COMMUNITY/tdroid_flip_pot_upright_lerobot",
        "lerobot/tdroid_flip_pot_upright",
    ],
    "tdroid_move_object_onto_plate": [
        "FedorX8/tdroid_move_object_onto_plate_lerobot",
        "IPEC-COMMUNITY/tdroid_move_object_onto_plate_lerobot",
        "lerobot/tdroid_move_object_onto_plate",
    ],
    "tdroid_knock_object_over": [
        "FedorX8/tdroid_knock_object_over_lerobot",
        "IPEC-COMMUNITY/tdroid_knock_object_over_lerobot",
        "lerobot/tdroid_knock_object_over",
    ],
    "tdroid_cover_object_with_towel": [
        "FedorX8/tdroid_cover_object_with_towel_lerobot",
        "IPEC-COMMUNITY/tdroid_cover_object_with_towel_lerobot",
        "lerobot/tdroid_cover_object_with_towel",
    ],
    "droid_wipe": [
        "FedorX8/droid_wipe_lerobot",
        "IPEC-COMMUNITY/droid_wipe_lerobot",
        "lerobot/droid_wipe",
    ],
    "rl_bench_v1": [
        "FedorX8/rl_bench_v1_lerobot",
        "IPEC-COMMUNITY/rl_bench_v1_lerobot",
        "lerobot/rl_bench_v1",
    ],
    "bridge_oxe": [
        "FedorX8/bridge_oxe_lerobot",
        "IPEC-COMMUNITY/bridge_oxe_lerobot",
        "lerobot/bridge_oxe",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--lerobot-dir", default=os.environ.get("LEROBOT_DIR", "/scratch1/chunheil/lerobot"))
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("HF_MAX_WORKERS", "1")))
    parser.add_argument("--sleep", type=float, default=10.0)
    args = parser.parse_args()

    local_dir = Path(args.lerobot_dir) / args.dataset
    local_dir.mkdir(parents=True, exist_ok=True)

    last_error = None
    for repo_id in CANDIDATES[args.dataset]:
        try:
            print(f"Trying {repo_id} -> {local_dir} with max_workers={args.max_workers}", flush=True)
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                max_workers=args.max_workers,
            )
            print(f"Downloaded {args.dataset} from {repo_id}", flush=True)
            return
        except Exception as exc:
            last_error = exc
            print(f"Failed {repo_id}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(args.sleep)

    raise SystemExit(f"No candidate repo worked for {args.dataset}. Last error: {last_error}")


if __name__ == "__main__":
    main()
