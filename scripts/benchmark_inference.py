"""Lightweight inference throughput benchmark for consumer NVIDIA GPUs.

Measures how fast a model generates tokens on your hardware. Runs a configurable
number of warmup passes (discarded) followed by measured passes, then reports
average throughput and peak GPU memory.

What is measured
----------------
- Model load time (wall clock, includes tokenizer)
- Per-run: generation time, tokens generated, tokens/sec
- Average / min / max throughput across measured runs
- Peak GPU memory allocated during generation
- CUDA device name, dtype, and device info

What is NOT measured
--------------------
- True time-to-first-token (TTFT) — use run_model.py with streaming for that
- Prompt encoding time (only generation time is reported)
- Network or disk latency (fetch the model first with fetch_hf_model.py)
"""

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight local inference throughput benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/benchmark_inference.py \\\n"
            "      --model-id Qwen/Qwen2.5-0.5B-Instruct \\\n"
            "      --prompt 'Explain entropy in one paragraph.' \\\n"
            "      --max-new-tokens 128 --warmup 1 --runs 3\n\n"
            "  python scripts/benchmark_inference.py \\\n"
            "      --model-dir ./models/Qwen--Qwen2.5-0.5B-Instruct \\\n"
            "      --greedy --runs 5\n"
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--model-id",
        default=None,
        metavar="ID",
        help="HF model ID (downloads/uses cache automatically).",
    )
    source.add_argument(
        "--model-dir",
        default=None,
        metavar="PATH",
        help="Path to a local model snapshot directory.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the concept of entropy in one paragraph.",
        help="Prompt used for every benchmark run.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Number of tokens to generate per run.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (ignored when --greedy is set).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability (ignored when --greedy is set).",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (deterministic; slightly faster).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs before timing begins (lets the GPU reach steady state).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measured runs to average.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Weight dtype. 'auto' lets transformers choose (usually bfloat16 on Ampere+).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run on. 'auto' uses CUDA if available.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Never make network requests; model must already be cached.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_str: str, use_cuda: bool):
    if dtype_str == "auto":
        return "auto" if use_cuda else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]


def fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"


def main() -> int:
    args = parse_args()

    if not args.model_id and not args.model_dir:
        print("Error: Provide --model-id or --model-dir.")
        print("  Example: python scripts/benchmark_inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct")
        return 1
    if args.warmup < 0:
        print("Error: --warmup must be 0 or greater.")
        return 1
    if args.runs < 1:
        print("Error: --runs must be 1 or greater.")
        return 1

    model_source = args.model_dir or args.model_id

    # ------------------------------------------------------------------ #
    # Device selection
    # ------------------------------------------------------------------ #
    if args.device == "cpu":
        use_cuda = False
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("Error: --device cuda requested but no CUDA device was found.")
            print("  Run python scripts/check_gpu.py for diagnostics.")
            return 1
        use_cuda = True
    else:
        use_cuda = torch.cuda.is_available()

    device_map = "auto" if use_cuda else None
    torch_dtype = resolve_dtype(args.dtype, use_cuda)

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    bar = "─" * 52
    print(bar)
    print("  Inference Benchmark")
    print(bar)
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device    : cuda  ({gpu_name})")
        print(f"  Dtype     : {args.dtype}")
    else:
        print("  Device    : cpu")
        print("  Note      : CPU throughput is much slower than GPU")
    print(f"  Model     : {model_source}")
    prompt_preview = args.prompt[:60] + ("…" if len(args.prompt) > 60 else "")
    print(f"  Prompt    : {prompt_preview}")
    print(f"  Max tokens: {args.max_new_tokens}")
    print(f"  Sampling  : {'greedy' if args.greedy else f'temp={args.temperature}, top_p={args.top_p}'}")
    print(f"  Warmup    : {args.warmup}    Measured : {args.runs}")
    print()

    # ------------------------------------------------------------------ #
    # Load model and tokenizer
    # ------------------------------------------------------------------ #
    print("Loading model…")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    t_load_start = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=args.local_only,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            local_files_only=args.local_only,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=False,
        )
    except Exception as exc:
        _handle_load_error(exc, model_source)
        return 1

    if not use_cuda:
        model = model.to("cpu")

    load_time = time.perf_counter() - t_load_start
    if use_cuda:
        model_vram = torch.cuda.memory_allocated()
        print(f"  Load time : {load_time:.2f} s   Model VRAM: {fmt_gb(model_vram)}")
    else:
        print(f"  Load time : {load_time:.2f} s")
    print()

    # ------------------------------------------------------------------ #
    # Build prompt
    # ------------------------------------------------------------------ #
    messages = [{"role": "user", "content": args.prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for models without a chat template
        text = f"User: {args.prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt")
    prompt_token_count = inputs["input_ids"].shape[-1]
    target = model.device if use_cuda else "cpu"
    inputs = {k: v.to(target) for k, v in inputs.items()}

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        pad_token_id=tokenizer.eos_token_id,
    )
    if not args.greedy:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    # ------------------------------------------------------------------ #
    # Warmup
    # ------------------------------------------------------------------ #
    if args.warmup > 0:
        print(f"Warmup ({args.warmup} run(s), not timed)…")
        for _ in range(args.warmup):
            with torch.no_grad():
                model.generate(**generate_kwargs)
        if use_cuda:
            torch.cuda.synchronize()
        print("  Done.")
        print()

    # ------------------------------------------------------------------ #
    # Measured runs
    # ------------------------------------------------------------------ #
    print(f"Measuring ({args.runs} run(s))…")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    run_times: list[float] = []
    run_tokens: list[int] = []

    for i in range(args.runs):
        if use_cuda:
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            out = model.generate(**generate_kwargs)

        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        n_new = out.shape[-1] - prompt_token_count
        tok_per_sec = n_new / elapsed if elapsed > 0 else 0.0
        run_times.append(elapsed)
        run_tokens.append(n_new)
        print(f"  Run {i + 1}: {elapsed:.2f}s  {n_new} tokens  {tok_per_sec:.1f} tok/s")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    avg_time = sum(run_times) / len(run_times)
    avg_tokens = sum(run_tokens) / len(run_tokens)
    avg_tps = avg_tokens / avg_time if avg_time > 0 else 0.0
    tps_list = [t / s for t, s in zip(run_tokens, run_times)]
    min_tps = min(tps_list)
    max_tps = max(tps_list)

    print()
    print(bar)
    print("  Summary")
    print(bar)
    print(f"  Model load time   : {load_time:.2f} s")
    print(f"  Prompt tokens     : {prompt_token_count}")
    print(f"  Avg tokens/run    : {avg_tokens:.0f}")
    print(f"  Avg gen time      : {avg_time:.2f} s")
    print(f"  Avg throughput    : {avg_tps:.1f} tok/s")
    print(f"  Min / Max         : {min_tps:.1f} / {max_tps:.1f} tok/s")

    if use_cuda:
        peak_mem = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"  Peak GPU memory   : {fmt_gb(peak_mem)}")
        print(f"  Reserved (cache)  : {fmt_gb(reserved)}")

    print()
    print("  Throughput is output tokens only (prompt encoding excluded).")
    if not use_cuda:
        print("  Running on CPU — throughput will be much lower than on GPU.")

    return 0


def _handle_load_error(exc: Exception, model_source: str) -> None:
    exc_str = str(exc)
    print(f"\n✗ Failed to load model: {exc}")

    if "no such file" in exc_str.lower() or "not a directory" in exc_str.lower():
        print(
            f"\n  '{model_source}' does not exist or is not a valid model directory.\n"
            "  To download the model:\n"
            f"    python scripts/fetch_hf_model.py {model_source}"
        )
    elif "local_files_only" in exc_str or "offline" in exc_str.lower():
        print(
            "\n  Model is not in the local cache. Either:\n"
            "    - Remove --local-only to allow the cache to be checked/updated\n"
            f"    - Or fetch it first:  python scripts/fetch_hf_model.py {model_source}"
        )
    elif "out of memory" in exc_str.lower() or "oom" in exc_str.lower():
        print(
            "\n  Out of GPU memory. Options:\n"
            "    - Use a smaller model variant (e.g. 0.5B or 1B)\n"
            "    - Pass --dtype float16 to reduce memory\n"
            "    - Pass --device cpu (slow, but no VRAM limit)"
        )
    else:
        print(
            "\n  Troubleshooting:\n"
            "    - Verify the model ID or path is correct\n"
            "    - Run: python scripts/check_gpu.py\n"
            "    - Ensure the conda environment is active:  conda activate hf-llm-bench"
        )


if __name__ == "__main__":
    sys.exit(main())
