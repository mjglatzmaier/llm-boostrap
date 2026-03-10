.PHONY: help gpu-check fetch run bench

MODEL_ID ?= Qwen/Qwen2.5-0.5B-Instruct
PROMPT   ?= Write a haiku about local inference.

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?##"}; {printf "  %-12s %s\n", $$1, $$2}'

gpu-check:  ## Check GPU, CUDA, and PyTorch availability
	python scripts/check_gpu.py

fetch:  ## Download a model to the HF cache (set MODEL_ID=... to override)
	python scripts/fetch_hf_model.py $(MODEL_ID)

run:  ## Run a single inference pass (set MODEL_ID=... and PROMPT=... to override)
	python scripts/run_model.py --model-id "$(MODEL_ID)" --prompt "$(PROMPT)"

bench:  ## Run the throughput benchmark (set MODEL_ID=... to override)
	python scripts/benchmark_inference.py --model-id "$(MODEL_ID)"
