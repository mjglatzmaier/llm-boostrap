#!/usr/bin/env bash
# One-command setup for hf-llm-bench.
# Detects CUDA version, creates the conda environment, and installs PyTorch
# with the matching CUDA wheel.
set -euo pipefail

ENV_NAME="hf-llm-bench"

# ---------------------------------------------------------------------------
# Make conda available in this non-interactive shell.
# `bash setup.sh` does not source ~/.bashrc, so conda is not on PATH by
# default. We locate the installation and source its shell hook explicitly.
# ---------------------------------------------------------------------------
_init_conda() {
    # 1. Already on PATH — nothing to do.
    if command -v conda &>/dev/null; then
        return 0
    fi

    # 2. Try common install locations.
    local candidates=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "$HOME/miniconda"
        "$HOME/anaconda"
        "/opt/conda"
        "/usr/local/conda"
    )
    for base in "${candidates[@]}"; do
        if [ -f "${base}/etc/profile.d/conda.sh" ]; then
            # shellcheck source=/dev/null
            source "${base}/etc/profile.d/conda.sh"
            return 0
        fi
    done

    # 3. Allow the caller to pass CONDA_ROOT explicitly.
    if [ -n "${CONDA_ROOT:-}" ] && [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_ROOT}/etc/profile.d/conda.sh"
        return 0
    fi

    echo "Error: conda not found on PATH and no known install location was detected."
    echo ""
    echo "  Option A — run from a shell where conda is already active:"
    echo "    conda activate base && bash setup.sh"
    echo ""
    echo "  Option B — pass your conda root explicitly:"
    echo "    CONDA_ROOT=/path/to/miniconda3 bash setup.sh"
    exit 1
}

_init_conda

# ---------------------------------------------------------------------------
# Detect the appropriate PyTorch CUDA wheel tag from nvidia-smi
# ---------------------------------------------------------------------------
detect_cuda_tag() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "cpu"
        return
    fi

    local cuda_ver major minor
    cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)
    if [ -z "$cuda_ver" ]; then
        echo "cpu"
        return
    fi

    major=$(echo "$cuda_ver" | cut -d. -f1)
    minor=$(echo "$cuda_ver" | cut -d. -f2)

    # Map driver-reported CUDA version to the nearest PyTorch wheel tag
    if   [ "$major" -ge 12 ] && [ "$minor" -ge 6 ]; then echo "cu126"
    elif [ "$major" -ge 12 ] && [ "$minor" -ge 4 ]; then echo "cu124"
    elif [ "$major" -ge 12 ] && [ "$minor" -ge 1 ]; then echo "cu121"
    elif [ "$major" -ge 11 ] && [ "$minor" -ge 8 ]; then echo "cu118"
    else echo "cpu"
    fi
}

CUDA_TAG=$(detect_cuda_tag)
echo "Detected CUDA tag : ${CUDA_TAG}"

# ---------------------------------------------------------------------------
# Create or update the conda environment
# ---------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "Updating existing '${ENV_NAME}' environment..."
    conda env update -n "${ENV_NAME}" -f environment.yaml --prune
else
    echo "Creating '${ENV_NAME}' environment..."
    conda env create -n "${ENV_NAME}" -f environment.yaml
fi

# ---------------------------------------------------------------------------
# Install PyTorch with the appropriate CUDA/CPU wheel
# ---------------------------------------------------------------------------
if [ "$CUDA_TAG" = "cpu" ]; then
    echo "Installing PyTorch (CPU only)..."
    conda run -n "${ENV_NAME}" pip install torch torchvision torchaudio
else
    echo "Installing PyTorch with ${CUDA_TAG} support..."
    conda run -n "${ENV_NAME}" pip install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

# ---------------------------------------------------------------------------
# Install remaining Python requirements
# ---------------------------------------------------------------------------
echo "Installing requirements.txt..."
conda run -n "${ENV_NAME}" pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "  Activate    : conda activate ${ENV_NAME}"
echo "  Check GPU   : python scripts/check_gpu.py"
echo "  Fetch model : python scripts/fetch_hf_model.py Qwen/Qwen2.5-0.5B-Instruct"
echo "  Run model   : python scripts/run_model.py --model-id Qwen/Qwen2.5-0.5B-Instruct"
echo "  Benchmark   : python scripts/benchmark_inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct"
