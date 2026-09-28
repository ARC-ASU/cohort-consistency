#!/usr/bin/env bash
# Serve the retriever / judge model through an OpenAI-compatible vLLM endpoint.
# The retriever (retrieve() calls inside programs) and the frozen judge share this endpoint.
# Paper: Qwen2.5-7B-Instruct for the 7B policy, Qwen2.5-3B-Instruct for the 3B policy.
#
#   CUDA_VISIBLE_DEVICES=0 bash training/scripts/serve_llm.sh Qwen/Qwen2.5-7B-Instruct
#
# Then export for training / evaluation:
#   export COHORT_LLM_BASE_URL=http://0.0.0.0:5500/v1
#   export COHORT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
set -euo pipefail
MODEL=${1:?usage: serve_llm.sh <model> [port]}
PORT=${2:-5500}
vllm serve "$MODEL" --port "$PORT" --api-key "${COHORT_LLM_API_KEY:-PROGRAM}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.3}"
