"""Fetch a Hugging Face model for local inference.

By default, the model is downloaded to the standard HF cache
(~/.cache/huggingface/hub). Pass --local-dir to save it to an explicit path
instead — useful if you want models alongside this repo or on a specific drive.

Loading after download
----------------------
  Cache download (default):
    python scripts/run_model.py --model-id Qwen/Qwen2.5-0.5B-Instruct

  Local dir download:
    python scripts/run_model.py --model-dir ./models/Qwen--Qwen2.5-0.5B-Instruct

About the HF cache layout
--------------------------
  ~/.cache/huggingface/hub/
    models--Qwen--Qwen2.5-0.5B-Instruct/
      snapshots/
        <hash>/      ← load from HERE (this is what this script prints)
      blobs/         ← raw content-addressed storage, do NOT load from here
      refs/

  Always load from the snapshots/<hash>/ directory, not from blobs/.
  The easiest approach is to just pass the original model ID — transformers
  will find the right snapshot automatically.
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a Hugging Face model for local inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Download to HF cache (load later by model ID):\n"
            "  python scripts/fetch_hf_model.py Qwen/Qwen2.5-0.5B-Instruct\n\n"
            "  # Download to an explicit local directory:\n"
            "  python scripts/fetch_hf_model.py Qwen/Qwen2.5-0.5B-Instruct \\\n"
            "      --local-dir ./models/Qwen--Qwen2.5-0.5B-Instruct\n\n"
            "  # Fetch a specific revision:\n"
            "  python scripts/fetch_hf_model.py Qwen/Qwen2.5-0.5B-Instruct \\\n"
            "      --revision abc1234\n\n"
            "  # Gated model (Llama, Gemma, etc.):\n"
            "  python scripts/fetch_hf_model.py meta-llama/Llama-3.2-1B-Instruct \\\n"
            "      --token hf_...\n"
        ),
    )
    parser.add_argument(
        "model_id",
        help="HF model ID, e.g. Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        metavar="PATH",
        help=(
            "Save model files to this directory instead of the HF cache. "
            "If omitted, files go to ~/.cache/huggingface/hub/."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Branch, tag, or commit SHA (default: main).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "HF access token for gated models (Llama, Gemma, etc.). "
            "Can also be set via the HF_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        metavar="PATTERN",
        help=(
            "Only download files matching these glob patterns. "
            "Example: --allow-patterns '*.safetensors' '*.json' 'tokenizer*'"
        ),
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        metavar="PATTERN",
        help=(
            "Skip files matching these glob patterns. "
            "Default skips non-PyTorch weight formats to save disk space."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    use_local_dir = args.local_dir is not None
    if use_local_dir:
        dest = Path(args.local_dir)
        dest.mkdir(parents=True, exist_ok=True)
        dest_str = str(dest.resolve())
    else:
        dest_str = None

    # Print plan before downloading
    print(f"Model     : {args.model_id}")
    print(f"Revision  : {args.revision or 'main'}")
    if use_local_dir:
        print(f"Save to   : {dest_str}  (local directory)")
    else:
        cache_root = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
        print(f"Save to   : HF cache  ({cache_root}/hub/)")
    if args.allow_patterns:
        print(f"Allow     : {', '.join(args.allow_patterns)}")
    if args.ignore_patterns:
        print(f"Ignore    : {', '.join(args.ignore_patterns)}")
    print()

    try:
        path = snapshot_download(
            repo_id=args.model_id,
            local_dir=dest_str,
            revision=args.revision,
            token=args.token,
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.ignore_patterns,
        )
    except Exception as exc:
        _handle_download_error(exc, args.model_id)
        return 1

    print(f"\n✓ Download complete.")
    print(f"  Files at: {path}")
    print()

    if use_local_dir:
        print("  Load with:")
        print(f"    python scripts/run_model.py --model-dir {path}")
        print()
        print("  Tip: Load from the directory root shown above, not from the blobs/")
        print("       subdirectory. The blobs/ folder is internal HF cache storage.")
    else:
        print("  Load with:")
        print(f"    python scripts/run_model.py --model-id {args.model_id}")
        print()
        print("  Or load directly from the snapshot directory:")
        print(f"    python scripts/run_model.py --model-dir {path}")
        print()
        print("  Tip: The HF cache organises files as snapshots/<hash>/. Always load")
        print("       from that snapshot directory (shown above), not from blobs/.")
        print("       Using the model ID is usually simplest — transformers resolves")
        print("       the right snapshot automatically.")

    return 0


def _handle_download_error(exc: Exception, model_id: str) -> None:
    exc_str = str(exc)
    print(f"\n✗ Download failed: {exc}")

    if "401" in exc_str or "gated" in exc_str.lower() or "access" in exc_str.lower():
        print(
            "\n  This model requires an access token.\n"
            "  1. Accept the model license at https://huggingface.co/\n"
            "  2. Create a token at https://huggingface.co/settings/tokens\n"
            "  3. Pass it: --token hf_...  or  export HF_TOKEN=hf_..."
        )
    elif "404" in exc_str or "not found" in exc_str.lower():
        print(
            f"\n  Model '{model_id}' was not found on the HF Hub.\n"
            "  Check the model ID spelling at https://huggingface.co/"
        )
    elif "connection" in exc_str.lower() or "timeout" in exc_str.lower():
        print(
            "\n  Network error. Check your internet connection.\n"
            "  If you are behind a proxy, set HTTPS_PROXY in your environment."
        )
    else:
        print(
            "\n  If this is an auth issue, try:\n"
            "    huggingface-cli login\n"
            "  then re-run this script."
        )


if __name__ == "__main__":
    sys.exit(main())
