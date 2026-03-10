# llm-bootstrap

A small, practical starter repo for running **local LLM inference** and doing
**lightweight benchmarking** on consumer NVIDIA GPUs. It gives you a safe,
isolated environment, a couple of focused scripts, and a clear on-ramp — then
gets out of your way.

Tested on Python 3.12.9 · RTX 4090 · Ubuntu · CUDA 12.x

---

## Why this repo exists

Getting a Hugging Face model running locally for the first time involves a
surprising number of small tripping points: conda vs pip, the wrong PyTorch
wheel, confusion about the HF cache layout, OOM errors, and so on. This repo
handles the boring setup work so you can focus on actually running models.

It is **not** a framework, not a training repo, and not a production inference
stack. It is a clean starting point.

---

## Non-goals

This repo intentionally does **not** include:

- Fine-tuning or training pipelines
- Model serving (HTTP APIs, gRPC, etc.)
- Distributed or multi-GPU inference
- Quantization tooling (GPTQ, AWQ, GGUF)
- Production reliability or SLA guarantees
- Framework abstractions or plugin systems

If you need any of those, see the [graduation suggestions](#when-to-graduate)
at the bottom of this README.

---

## Assumptions

| Item | Value |
|------|-------|
| Python | 3.12.9 |
| Primary test GPU | RTX 4090 (24 GB VRAM) |
| OS | Linux (WSL on Windows should work; macOS CPU-only) |
| Package manager | Conda (isolated env) + pip for PyTorch |
| Model format | Hugging Face Transformers (safetensors / PyTorch weights) |
| Model storage | HF cache (`~/.cache/huggingface/hub`) or an explicit local dir |

CUDA 11.8 through 12.6 is supported via `setup.sh` auto-detection.

---

## Repo structure

```
llm-bootstrap/
├── .gitignore
├── LICENSE
├── Makefile                       # Optional shortcuts (make gpu-check, etc.)
├── README.md
├── environment.yaml               # Conda env: Python 3.12, non-torch deps
├── requirements.txt               # Pip deps installed after PyTorch
├── setup.sh                       # One-command setup with CUDA auto-detection
└── scripts/
    ├── check_gpu.py               # GPU diagnostics: device props + live stats
    ├── fetch_hf_model.py          # Download any HF model (cache or local dir)
    ├── run_model.py               # Single inference pass + TTFT / throughput
    └── benchmark_inference.py     # Multi-run throughput benchmark
```

---

## Quickstart

```bash
# 1. Clone and set up the environment
git clone https://github.com/you/llm-bootstrap.git
cd llm-bootstrap
bash setup.sh
conda activate hf-llm-bench

# 2. Verify your GPU is visible
python scripts/check_gpu.py

# 3. Fetch a small model
python scripts/fetch_hf_model.py Qwen/Qwen2.5-0.5B-Instruct

# 4. Run inference
python scripts/run_model.py --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --prompt "What is attention in a transformer?"

# 5. Benchmark it
python scripts/benchmark_inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct
```

---

## Environment setup

### Why Conda?

Conda creates a fully isolated Python environment, which prevents conflicts
with system Python or other projects. PyTorch has strict CUDA version
dependencies that are easiest to manage in a clean env.

### Why is PyTorch installed separately?

The PyTorch CUDA wheel must match your system's CUDA version. `setup.sh`
detects this automatically. Installing PyTorch via conda-forge or a plain
`pip install torch` often gets the CPU-only build, which is a common source
of confusion.

### Automatic setup

```bash
bash setup.sh
conda activate hf-llm-bench
```

`setup.sh` will:
1. Detect your CUDA version via `nvidia-smi`
2. Create the `hf-llm-bench` conda environment (Python 3.12.9)
3. Install PyTorch with the matching CUDA wheel
4. Install all remaining dependencies from `requirements.txt`

> **Different CUDA version?** `setup.sh` auto-selects from `cu118`, `cu121`,
> `cu124`, `cu126`. To override, edit `detect_cuda_tag` in `setup.sh` or see
> the [PyTorch install page](https://pytorch.org/get-started/locally/).

### Manual setup

```bash
conda env create -f environment.yaml
conda activate hf-llm-bench

# Replace cu124 with your CUDA version tag
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

### Verify GPU

```bash
python scripts/check_gpu.py
```

Expected output (RTX 4090 example):

```
Python     : 3.12.9
Executable : /home/user/miniconda3/envs/hf-llm-bench/bin/python
PyTorch    : 2.x.x+cu124
CUDA avail : True
GPU count  : 1

  [0] NVIDIA GeForce RTX 4090
      Compute capability : 8.9
      Total VRAM         : 24.0 GB
      Multi-processors   : 128

NVIDIA driver : 555.xx

  [0] Live stats:
      GPU utilization    : 0 %
      Temperature        : 42 °C
      Power draw         : 18 W / 450 W
      VRAM used          : 0.5 GB / 24.0 GB
```

If `CUDA avail : False`, see [Troubleshooting](#troubleshooting).

---

## Fetch a model

`scripts/fetch_hf_model.py` downloads any causal LM from the HF Hub.

### Download to HF cache (recommended default)

```bash
python scripts/fetch_hf_model.py Qwen/Qwen2.5-0.5B-Instruct
```

The model goes to `~/.cache/huggingface/hub/`. After downloading, load it by
passing the same model ID to `run_model.py` or `benchmark_inference.py`.

### Download to an explicit local directory

```bash
python scripts/fetch_hf_model.py Qwen/Qwen2.5-0.5B-Instruct \
    --local-dir ./models/Qwen--Qwen2.5-0.5B-Instruct
```

Use `--model-dir ./models/Qwen--Qwen2.5-0.5B-Instruct` when loading.

### Fetch a specific revision

```bash
python scripts/fetch_hf_model.py Qwen/Qwen2.5-0.5B-Instruct \
    --revision abc1234
```

### Gated models (Llama, Gemma, etc.)

```bash
# via flag
python scripts/fetch_hf_model.py meta-llama/Llama-3.2-1B-Instruct \
    --token hf_...

# or via environment variable
export HF_TOKEN=hf_...
python scripts/fetch_hf_model.py meta-llama/Llama-3.2-1B-Instruct
```

You must also accept the model's license on [huggingface.co](https://huggingface.co)
before the token will work.

### Fetch flags

| Flag | Default | Description |
|------|---------|-------------|
| `model_id` | *(required)* | HF model ID |
| `--local-dir` | *(HF cache)* | Save to this path instead of the cache |
| `--revision` | `main` | Branch, tag, or commit SHA |
| `--token` | `$HF_TOKEN` | Access token for gated models |
| `--allow-patterns` | *(all)* | Only download files matching these patterns |
| `--ignore-patterns` | PT/flax/TF extras | Skip files matching these patterns |

---

## Run inference

`scripts/run_model.py` runs a single inference pass and streams output,
then reports TTFT, throughput, and GPU memory.

```bash
# Load by model ID (from HF cache)
python scripts/run_model.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --prompt "Explain the attention mechanism in two sentences."

# Load from a local directory
python scripts/run_model.py \
    --model-dir ./models/Qwen--Qwen2.5-0.5B-Instruct \
    --prompt "Write a haiku about GPUs."

# Greedy decoding (deterministic), explicit dtype
python scripts/run_model.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --greedy --dtype bfloat16

# Force CPU (no CUDA required)
python scripts/run_model.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --device cpu
```

### Run flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | — | HF model ID (uses HF cache) |
| `--model-dir` | — | Path to local snapshot directory |
| `--prompt` | haiku prompt | User message |
| `--system` | helpful assistant | System message |
| `--max-new-tokens` | `256` | Max tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` | `0.9` | Nucleus sampling probability |
| `--greedy` | off | Greedy decoding (disables sampling) |
| `--dtype` | `auto` | `auto` / `float16` / `bfloat16` / `float32` |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--local-only` | off | Never make network requests |
| `--trust-remote-code` | off | Allow custom model code (use carefully) |

### Example output

```
Model  : Qwen/Qwen2.5-0.5B-Instruct
Device : cuda:0
GPU    : NVIDIA GeForce RTX 4090
Dtype  : auto

Loading tokenizer and model...
Load time : 1.43 s
Model VRAM: 0.97 GB

=== OUTPUT ===

Circuits hum and glow,
Vectors dance through silent space—
Thought made fast by light.

=== BENCHMARK ===
  Time to first token : 0.312 s
  Generation time     : 2.841 s
  Prompt tokens       : 38
  Tokens generated    : 21
  Throughput          : 7.4 tok/s

=== GPU MEMORY ===
  Pre-generate        : 1.24 GB
  Post-generate       : 1.27 GB
  Peak allocated      : 1.31 GB
  Reserved (cache)    : 2.00 GB
```

---

## Run benchmark

`scripts/benchmark_inference.py` runs multiple passes and averages throughput.
Use this for a more stable tokens/sec measurement than a single run.

```bash
# Basic benchmark (1 warmup, 3 measured runs)
python scripts/benchmark_inference.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct

# More runs, explicit prompt, greedy decoding
python scripts/benchmark_inference.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --prompt "Describe how transformers work." \
    --max-new-tokens 256 \
    --warmup 2 --runs 5 \
    --greedy

# From a local directory
python scripts/benchmark_inference.py \
    --model-dir ./models/Qwen--Qwen2.5-0.5B-Instruct \
    --runs 5
```

### Benchmark flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | — | HF model ID |
| `--model-dir` | — | Path to local snapshot directory |
| `--prompt` | entropy prompt | Prompt used for all runs |
| `--max-new-tokens` | `128` | Tokens to generate per run |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` | `0.9` | Nucleus sampling probability |
| `--greedy` | off | Greedy decoding |
| `--warmup` | `1` | Discarded warmup runs |
| `--runs` | `3` | Measured runs to average |
| `--dtype` | `auto` | Weight dtype |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--local-only` | off | Never make network requests |

### What is measured

- **Model load time** — wall clock from import to ready
- **Per-run generation time** — from `generate()` call to last token
- **Tokens/sec** — output tokens only; prompt encoding is excluded
- **Peak GPU memory** — `torch.cuda.max_memory_allocated()` over the runs

**TTFT** (time to first token) is not measured here. Use `run_model.py` for
that — it streams tokens and captures the first-chunk timestamp.

---

## Makefile shortcuts

```bash
make help        # list all targets
make gpu-check   # check GPU visibility
make fetch       # fetch Qwen/Qwen2.5-0.5B-Instruct (default)
make run         # run inference with default model/prompt
make bench       # run benchmark with default model

# Override the model or prompt
make fetch MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
make run   MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3 PROMPT="What is entropy?"
```

---

## Model compatibility

Any causal LM on the HF Hub with `AutoModelForCausalLM` support works. Examples:

```bash
python scripts/fetch_hf_model.py Qwen/Qwen2.5-7B-Instruct
python scripts/fetch_hf_model.py mistralai/Mistral-7B-Instruct-v0.3
python scripts/fetch_hf_model.py microsoft/Phi-3-mini-4k-instruct
python scripts/fetch_hf_model.py meta-llama/Llama-3.2-3B-Instruct --token hf_...
python scripts/fetch_hf_model.py google/gemma-3-4b-it --token hf_...
```

---

## HF cache layout

When you fetch without `--local-dir`, the model goes into:

```
~/.cache/huggingface/hub/
  models--Qwen--Qwen2.5-0.5B-Instruct/
    snapshots/
      <hash>/      ← this is the loadable model directory
        config.json
        tokenizer.json
        model.safetensors
        ...
    blobs/         ← raw content-addressed storage; do NOT load from here
    refs/
```

**Always load from `snapshots/<hash>/`**, not from `blobs/`. The easiest
approach is to pass the model ID (`--model-id`) and let transformers resolve
the snapshot automatically. If you want an explicit path, the fetch script
prints the correct snapshot directory after download.

---

## Troubleshooting

**`CUDA avail : False` after setup**
- You likely have the CPU-only PyTorch wheel. Reinstall with the correct index URL:
  ```bash
  pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu124
  ```
  Replace `cu124` with your CUDA tag. Run `nvidia-smi` to check your CUDA version.

**`nvidia-smi` not found / no GPU detected**
- Your GPU driver may not be installed, or you are in a VM/container without
  GPU passthrough. `check_gpu.py` will report this clearly.

**Model not found locally**
- If using `--local-only`, the model must already be in the HF cache.
  Fetch it first: `python scripts/fetch_hf_model.py <model_id>`

**Confused about `blobs/` vs `snapshots/`**
- `blobs/` is internal HF cache storage (content-addressed). Always load from
  the `snapshots/<hash>/` directory. The fetch script tells you the exact path.

**`EnvironmentError` or missing `config.json`**
- You may be pointing at a directory that isn't a complete model snapshot.
  Re-fetch the model, or check the path.

**Out of GPU memory (OOM)**
- Use a smaller model (0.5B or 1B variants are good starting points)
- Try `--dtype float16` to reduce VRAM usage
- Pass `--device cpu` as a last resort (much slower)

**Wrong Python or wrong environment**
- Run `which python` and `python --version`. If it's not the conda env,
  activate it: `conda activate hf-llm-bench`
- `check_gpu.py` prints the active Python executable path to help verify.

**Tokenizer warning about chat template**
- Some models don't have a built-in chat template. The scripts fall back to a
  plain `User: ... \nAssistant:` format automatically.

**Slow first run**
- Model weights load from disk every run. SSDs help significantly. Use
  `fetch_hf_model.py --local-dir` to keep weights on fast storage.

---

## When to graduate

This repo is a starting point. Once you have the basics working, you may want:

| Need | Tool |
|------|------|
| Fast GPU inference (batching, paged KV cache) | [vLLM](https://github.com/vllm-project/vllm) |
| Simple local chat / API server | [Ollama](https://ollama.com) |
| Quantized models (GGUF, 4-bit) | [llama.cpp](https://github.com/ggerganov/llama.cpp) |
| More benchmarking tooling | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| Fine-tuning / LoRA | [Unsloth](https://github.com/unslothai/unsloth), [TRL](https://github.com/huggingface/trl) |
| Production serving | [TGI](https://github.com/huggingface/text-generation-inference) |

