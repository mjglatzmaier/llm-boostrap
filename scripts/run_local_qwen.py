import argparse
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local inference with a Hugging Face causal LM."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model id, e.g. Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Local model snapshot directory path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a haiku about local inference.",
        help="User prompt to run",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a concise and helpful assistant.",
        help="System prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution",
    )
    return parser.parse_args()


def resolve_model_source(model_id: Optional[str], model_dir: Optional[str]) -> str:
    if model_dir:
        return model_dir
    if model_id:
        return model_id
    raise ValueError("You must provide either --model-id or --model-dir.")


def main() -> int:
    args = parse_args()

    try:
        model_source = resolve_model_source(args.model_id, args.model_dir)
    except ValueError as exc:
        print(f"Argument error: {exc}")
        return 1

    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        device_map = "auto"
        torch_dtype = "auto"
        target_device = "cuda"
    else:
        device_map = None
        torch_dtype = torch.float32
        target_device = "cpu"

    print(f"Loading model from: {model_source}")
    print(f"Target device: {target_device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=True,
            trust_remote_code=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            local_files_only=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=False,
        )
    except Exception as exc:
        print("Failed to load model/tokenizer.")
        print(f"Exception: {exc}")
        return 1

    if not use_cuda:
        model = model.to("cpu")

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for tokenizers/models without chat templates
        text = f"System: {args.system}\nUser: {args.prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt")

    if use_cuda:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n=== GENERATED TEXT ===\n")
    print(decoded)
    print("\n======================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())