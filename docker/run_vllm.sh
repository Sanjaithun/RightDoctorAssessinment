#!/usr/bin/env bash
# Launch vLLM on the NVIDIA DGX Spark (GB10, 128 GB unified memory, ARM64).
# Usage: HF_TOKEN=hf_xxx ./docker/run_vllm.sh
set -euo pipefail

# Use NVIDIA's NGC vLLM image built for the Spark (ARM64 + Blackwell).
# Check https://catalog.ngc.nvidia.com for the latest tag and set it here.
IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:latest}"
MODEL="${MODEL:-RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic}"   # ~70 GB, pre-quantized

docker run -d --name vllm-llama33-70b \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --restart unless-stopped \
  -p 8000:8000 \
  -e HF_TOKEN="${HF_TOKEN:?set HF_TOKEN (Llama 3.3 is a gated model)}" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --served-model-name llama-3.3-70b \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.80 \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 4096 \
    --enable-prefix-caching \
    --swap-space 0
