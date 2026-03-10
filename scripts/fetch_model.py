"""Download a Hugging Face model snapshot to a local directory."""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model to a local directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_id",
        type=str,
        help="Hugging Face model ID, e.g. Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Parent directory for model snapshots",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision, branch, or tag (default: main)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "HuggingFace access token for gated models. "
            "Can also be set via the HF_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--ignore-patterns",
        type=str,
        nargs="*",
        default=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        help="Glob patterns to skip during download (speeds up fetch for PT-only use)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Replicate HF Hub's local naming convention: org/model -> org--model
    local_name = args.model_id.replace("/", "--")
    dest = Path(args.save_dir) / local_name
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Model     : {args.model_id}")
    print(f"Revision  : {args.revision or 'main'}")
    print(f"Save to   : {dest.resolve()}")
    if args.ignore_patterns:
        print(f"Skipping  : {', '.join(args.ignore_patterns)}")
    print()

    try:
        path = snapshot_download(
            repo_id=args.model_id,
            local_dir=str(dest),
            revision=args.revision,
            token=args.token,
            ignore_patterns=args.ignore_patterns,
        )
        print(f"\n✓ Model saved to: {path}")
        print(f"\n  Run it with:")
        print(f"    python scripts/run_model.py --model-dir {path}")
    except Exception as exc:
        print(f"\n✗ Download failed: {exc}")
        if "401" in str(exc) or "gated" in str(exc).lower():
            print(
                "\nThis model is gated. Pass --token <HF_TOKEN> "
                "or set the HF_TOKEN environment variable."
            )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
