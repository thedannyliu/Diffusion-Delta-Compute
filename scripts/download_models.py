#!/usr/bin/env python
import argparse
import os
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model repo id")
    parser.add_argument("--target_dir", type=str, required=True, help="Local directory to store snapshot")
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    snapshot_download(repo_id=args.model_id, local_dir=args.target_dir, local_dir_use_symlinks=False, revision=args.revision)
    print(f"Downloaded {args.model_id} to {args.target_dir}")


if __name__ == "__main__":
    main()



