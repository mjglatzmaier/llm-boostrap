# hf-llm-bench

A lightweight, extensible framework for downloading and benchmarking Hugging Face
language models on local hardware.

---

## Goals

- **Simple setup** — one script creates the full environment on any CUDA-capable machine
- **Model agnostic** — fetch and run any causal LM from the HF Hub
- **Real benchmarks** — time to first token, throughput, GPU memory, power draw, and temperature
- **Extensible** — small, readable scripts you can adapt for your own experiments

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Linux (or WSL on Windows) | macOS works for CPU-only |
| NVIDIA GPU + CUDA 11.8 – 12.6 | CPU fallback available |
| [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda | Used to create the isolated env |
| Internet access (first run) | To download models and packages |

---

## Setup

### Automatic (recommended)

```bash
bash setup.sh
conda activate hf-llm-bench
```

`setup.sh` will:
1. Detect your CUDA version via `nvidia-smi`
2. Create the `hf-llm-bench` conda environment (Python 3.12)
3. Install PyTorch with the matching CUDA wheel
4. Install all remaining dependencies

> **Different CUDA version?** The script auto-selects from `cu118`, `cu121`, `cu124`,
> `cu126`. If you need a different tag, edit the `detect_cuda_tag` function in
> `setup.sh` or see the [PyTorch install page](https://pytorch.org/get-started/locally/).

### Manual

```bash
conda env create -f environment.yaml
conda activate hf-llm-bench

# Replace cu124 with your CUDA version (cu118 / cu121 / cu124 / cu126)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

### Verify GPU

```bash
python scripts/check_gpu.py
```

Expected output (NVIDIA GPU):
```
Python     : 3.12.x
PyTorch    : 2.x.x+cuXXX
CUDA avail : True
GPU count  : 1

  [0] NVIDIA GeForce RTX 4090
      Compute capability : 8.9
      Total VRAM         : 24.0 GB
      Multi-processors   : 128

NVIDIA driver : 555.xx.xx

  [0] Live stats:
      GPU utilization    : 0 %
      Temperature        : 42 °C
      Power draw         : 18 W / 450 W
      VRAM used          : 0.5 GB / 24.0 GB
```

---

## Usage

### 1. Fetch a model

Downloads a model snapshot from the HF Hub into `./models/`.

```bash
python scripts/fetch_model.py Qwen/Qwen2.5-0.5B-Instruct
```

The model is saved to `models/Qwen--Qwen2.5-0.5B-Instruct/`.

**Gated models** (e.g. Llama, Gemma) require an HF access token:

```bash
# via flag
python scripts/fetch_model.py meta-llama/Llama-3.2-1B-Instruct --token hf_...

# or via environment variable
export HF_TOKEN=hf_...
python scripts/fetch_model.py meta-llama/Llama-3.2-1B-Instruct
```

| Flag | Default | Description |
|------|---------|-------------|
| `model_id` | *(required)* | HF model ID |
| `--save-dir` | `models` | Parent directory for snapshots |
| `--revision` | `main` | Branch, tag, or commit SHA |
| `--token` | `$HF_TOKEN` | HF access token for gated models |
| `--ignore-patterns` | PT/flax/TF extras | File patterns to skip |

### 2. Run inference + benchmark

```bash
# Load by HF model ID (downloads if not cached locally)
python scripts/run_model.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain the attention mechanism in two sentences."

# Load from a local snapshot
python scripts/run_model.py \
  --model-dir models/Qwen--Qwen2.5-0.5B-Instruct \
  --prompt "Write a haiku about GPUs." \
  --max-new-tokens 64
```

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | — | HF model ID |
| `--model-dir` | — | Path to local snapshot |
| `--prompt` | haiku prompt | User message |
| `--system` | helpful assistant | System message |
| `--max-new-tokens` | `256` | Max tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` | `0.9` | Nucleus sampling probability |
| `--greedy` | off | Use greedy decoding instead of sampling |
| `--local-only` | off | Never make network requests |
| `--cpu` | off | Force CPU (no CUDA) |

**Example benchmark output:**

```
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

=== GPU STATUS ===
  GPU utilization     : 8 %
  Memory utilization  : 12 %
  Temperature         : 65 °C
  Power draw          : 87 W / 450 W
  VRAM used           : 1.27 GB / 24.00 GB
```

---

## Project Structure

```
hf-setup/
├── environment.yaml          # Conda environment (Python 3.12, non-torch deps)
├── requirements.txt          # Pip dependencies (installed by setup.sh)
├── setup.sh                  # One-command setup with CUDA auto-detection
└── scripts/
    ├── check_gpu.py          # GPU diagnostics (device properties + live pynvml stats)
    ├── fetch_model.py        # Download any HF model to ./models/
    └── run_model.py          # Inference runner with full benchmark output
```

---

## Model Compatibility

Any causal LM on the HF Hub with `AutoModelForCausalLM` support works. Examples:

```bash
# Qwen 2.5 (0.5B – 72B)
python scripts/fetch_model.py Qwen/Qwen2.5-7B-Instruct

# Mistral
python scripts/fetch_model.py mistralai/Mistral-7B-Instruct-v0.3

# Phi-3
python scripts/fetch_model.py microsoft/Phi-3-mini-4k-instruct

# Llama 3.2 (gated — requires HF token)
python scripts/fetch_model.py meta-llama/Llama-3.2-3B-Instruct --token hf_...

# Gemma 3 (gated)
python scripts/fetch_model.py google/gemma-3-4b-it --token hf_...
```

---

## Tips

- **VRAM too low?** Use a smaller model variant (e.g. 0.5B or 1B) or pass `--cpu`.
- **Slow on first run?** The model loads into VRAM on every run; TTFT includes that
  overhead if you're using `--model-id` without a local snapshot. Fetch first with
  `fetch_model.py` for accurate timing.
- **Reproducible outputs?** Pass `--greedy` to disable sampling.
