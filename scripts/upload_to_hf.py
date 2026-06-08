"""
scripts/upload_to_hf.py

Upload model weights + model card to HF Hub, then create/update the Space.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login          # or set HF_TOKEN env var

Usage:
    python scripts/upload_to_hf.py --user edarsem
    python scripts/upload_to_hf.py --user edarsem --ckpt checkpoints/chessformer_v0.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user",      required=True, help="Your HF username, e.g. edarsem")
    parser.add_argument("--ckpt",      default="checkpoints/chessformer_v0.pt",
                        help="Path to the exported (slim) checkpoint")
    parser.add_argument("--model-repo", default="chessformer",    help="Model repo name")
    parser.add_argument("--space-repo", default="chessformer-demo", help="Space repo name")
    parser.add_argument("--private",   action="store_true", help="Create repos as private")
    parser.add_argument("--skip-space", action="store_true", help="Skip Space upload")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api        = HfApi()
    model_id   = f"{args.user}/{args.model_repo}"
    space_id   = f"{args.user}/{args.space_repo}"
    card_path  = "hf_space/model_card.md"
    space_dir  = "hf_space"

    assert os.path.exists(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    assert os.path.exists(card_path), f"Model card not found: {card_path}"

    # ── Model repo ──────────────────────────────────────────────────────
    print(f"\n── Model repo: {model_id} ──")
    create_repo(model_id, repo_type="model", private=args.private, exist_ok=True)
    print("  Uploading model card …")
    api.upload_file(
        path_or_fileobj = card_path,
        path_in_repo    = "README.md",
        repo_id         = model_id,
        repo_type       = "model",
        commit_message  = "Add model card",
    )
    ckpt_mb = os.path.getsize(args.ckpt) / 1e6
    print(f"  Uploading checkpoint ({ckpt_mb:.0f} MB) — this may take a minute …")
    api.upload_file(
        path_or_fileobj = args.ckpt,
        path_in_repo    = os.path.basename(args.ckpt),
        repo_id         = model_id,
        repo_type       = "model",
        commit_message  = "Add model weights",
    )
    print(f"  Done → https://huggingface.co/{model_id}")

    if args.skip_space:
        return

    # ── Space ────────────────────────────────────────────────────────────
    print(f"\n── Space: {space_id} ──")
    create_repo(space_id, repo_type="space", space_sdk="docker",
                private=args.private, exist_ok=True)
    for fname, repo_name in [("README.md", "README.md"), ("Dockerfile", "Dockerfile")]:
        local = os.path.join(space_dir, fname)
        assert os.path.exists(local), f"Missing: {local}"
        print(f"  Uploading {fname} …")
        api.upload_file(
            path_or_fileobj = local,
            path_in_repo    = repo_name,
            repo_id         = space_id,
            repo_type       = "space",
            commit_message  = f"Add {fname}",
        )
    print(f"  Done → https://huggingface.co/spaces/{space_id}")
    print(f"\n  The Space will now build (~5 min). Watch progress at the link above.")
    print(f"  It clones the GitHub repo and downloads the checkpoint from {model_id}.")


if __name__ == "__main__":
    main()
