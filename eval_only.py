import torch
import h5py
import argparse
import json
from model import ReWiNDTransformer
from utils.eval_utils import compute_metrics_multi

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model_args = checkpoint["args"]

    model = ReWiNDTransformer(
        args=model_args,
        video_dim=768,
        text_dim=384,
        hidden_dim=512
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load eval data
    h5_eval = h5py.File(args.h5_path, "r")
    h5_close = h5py.File("datasets/metaworld_dino_embeddings_eval_close_succ.h5", "r")
    h5_fail = h5py.File("datasets/metaworld_dino_embeddings_eval_all_fail.h5", "r")
    task_list = json.load(open("utils/new_task_v2.json", "r"))

    with torch.no_grad():
        compute_metrics_multi(
            model_args,
            model,
            gt_data=h5_eval,
            close_success_data=h5_close,
            all_fail_data=h5_fail,
            task_list=task_list,
            epoch=None
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--h5_path", type=str, default="datasets/metaworld_embeddings_eval.h5")
    args = parser.parse_args()
    main(args)
