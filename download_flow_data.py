import argparse

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_path",
        type=str,
        default="datasets",
        help="Path to download the datasets.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="lincolnlam/rewind_openx_flow_embeddings",
        help="Hugging Face dataset repo ID.",
    )
    args = parser.parse_args()

    load_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.download_path,
    )
    print(f"Downloaded dataset to {load_dir}")


if __name__ == "__main__":
    main()
