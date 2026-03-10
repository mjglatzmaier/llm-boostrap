"""Run local inference with a Hugging Face causal LM and report benchmark metrics.

Metrics reported
----------------
- Time to first token (TTFT)
- Total generation time
- Prompt token count
- Output token count
- Throughput (tokens / second)
- GPU memory: pre-generate, post-generate, peak allocated, reserved cache
- GPU live stats (via pynvml): utilization, temperature, power draw
"""

import argparse
import sys
import time
from threading import Thread
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    import pynvml
    _PYNVML_OK = True
except ImportError:
    _PYNVML_OK = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local inference with a Hugging Face causal LM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="HF model ID (e.g. Qwen/Qwen2.5-0.5B-Instruct). Downloads if not cached.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to a local model snapshot directory (from fetch_model.py).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a haiku about local inference.",
        help="User prompt.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a concise and helpful assistant.",
        help="System message.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (ignored when do_sample=False).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling probability.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Require local files only; never make network requests.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (disables CUDA even if available).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_model_source(model_id: Optional[str], model_dir: Optional[str]) -> str:
    if model_dir:
        return model_dir
    if model_id:
        return model_id
    raise ValueError("Provide --model-id or --model-dir.")


def fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"


def collect_gpu_stats(device_index: int = 0) -> dict:
    """Return live GPU stats from pynvml, or an empty dict if unavailable."""
    if not _PYNVML_OK:
        return {}
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000       # mW → W
        power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "temp_c": temp,
            "power_w": power_w,
            "power_limit_w": power_limit_w,
            "vram_used": mem_info.used,
            "vram_total": mem_info.total,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    try:
        model_source = resolve_model_source(args.model_id, args.model_dir)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    use_cuda = torch.cuda.is_available() and not args.cpu
    device_map = "auto" if use_cuda else None
    torch_dtype = "auto" if use_cuda else torch.float32
    device_label = f"cuda:{torch.cuda.current_device()}" if use_cuda else "cpu"

    print(f"Model  : {model_source}")
    print(f"Device : {device_label}")
    if use_cuda:
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Load model + tokenizer
    # ------------------------------------------------------------------
    print("\nLoading tokenizer and model...")
    t_load_start = time.perf_counter()

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        mem_before_load = torch.cuda.memory_allocated()

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
        print(f"\nFailed to load model: {exc}")
        return 1

    if not use_cuda:
        model = model.to("cpu")

    t_load_end = time.perf_counter()
    load_time = t_load_end - t_load_start
    print(f"Load time : {load_time:.2f} s")
    if use_cuda:
        model_mem = torch.cuda.memory_allocated() - mem_before_load
        print(f"Model VRAM: {fmt_gb(model_mem)}")

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for models without a chat template
        text = f"System: {args.system}\nUser: {args.prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt")
    prompt_tokens = inputs["input_ids"].shape[-1]

    if use_cuda:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------
    streamer = TextIteratorStreamer(
        tokenizer, skip_special_tokens=True, skip_prompt=True
    )

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    if not args.greedy:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        mem_pre_gen = torch.cuda.memory_allocated()
        torch.cuda.synchronize()

    print("\n=== OUTPUT ===\n")
    t_gen_start = time.perf_counter()

    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    ttft: Optional[float] = None
    output_text = ""
    for chunk in streamer:
        if ttft is None:
            ttft = time.perf_counter() - t_gen_start
        print(chunk, end="", flush=True)
        output_text += chunk

    thread.join()

    if use_cuda:
        torch.cuda.synchronize()

    t_gen_end = time.perf_counter()
    gen_time = t_gen_end - t_gen_start

    # Count output tokens by re-encoding (no special tokens)
    n_output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
    throughput = n_output_tokens / gen_time if gen_time > 0 else 0.0

    # ------------------------------------------------------------------
    # Benchmark summary
    # ------------------------------------------------------------------
    print("\n\n=== BENCHMARK ===")
    if ttft is not None:
        print(f"  Time to first token : {ttft:.3f} s")
    print(f"  Generation time     : {gen_time:.2f} s")
    print(f"  Prompt tokens       : {prompt_tokens}")
    print(f"  Tokens generated    : {n_output_tokens}")
    print(f"  Throughput          : {throughput:.1f} tok/s")

    if use_cuda:
        mem_post_gen = torch.cuda.memory_allocated()
        mem_peak = torch.cuda.max_memory_allocated()
        mem_reserved = torch.cuda.memory_reserved()

        print("\n=== GPU MEMORY ===")
        print(f"  Pre-generate        : {fmt_gb(mem_pre_gen)}")
        print(f"  Post-generate       : {fmt_gb(mem_post_gen)}")
        print(f"  Peak allocated      : {fmt_gb(mem_peak)}")
        print(f"  Reserved (cache)    : {fmt_gb(mem_reserved)}")

        stats = collect_gpu_stats(torch.cuda.current_device())
        if stats:
            print("\n=== GPU STATUS ===")
            print(f"  GPU utilization     : {stats['gpu_util']} %")
            print(f"  Memory utilization  : {stats['mem_util']} %")
            print(f"  Temperature         : {stats['temp_c']} °C")
            print(
                f"  Power draw          : "
                f"{stats['power_w']:.0f} W / {stats['power_limit_w']:.0f} W"
            )
            print(
                f"  VRAM used           : "
                f"{fmt_gb(stats['vram_used'])} / {fmt_gb(stats['vram_total'])}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
