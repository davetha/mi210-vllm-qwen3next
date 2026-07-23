#!/usr/bin/env bash
# Launch Qwen3-Next-80B on vLLM (AMD-prebuilt CDNA image) on a single MI210 (gfx90a).
#
# Usage: launch-vllm023.sh [/path/to/model] [port] [hip_device]
set -euo pipefail

MODEL="${1:-/models/qwen3-next-80b-a3b-int4}"
PORT="${2:-8000}"
HIP_DEV="${3:-0}"
IMG="rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0"

# host video/render gids — adjust to your system (e.g. `getent group video render`)
VIDEO_GID="$(getent group video  | cut -d: -f3)"
RENDER_GID="$(getent group render | cut -d: -f3)"

docker rm -f vllm >/dev/null 2>&1 || true
docker run -d --name vllm --network host --restart unless-stopped \
  --device /dev/kfd --device /dev/dri \
  --group-add "${VIDEO_GID}" --group-add "${RENDER_GID}" \
  --security-opt seccomp=unconfined --shm-size 16g \
  -e HIP_VISIBLE_DEVICES="${HIP_DEV}" \
  -e VLLM_ROCM_USE_AITER=0 \
  -v "$(dirname "${MODEL}")":"$(dirname "${MODEL}")" \
  --entrypoint vllm "${IMG}" \
  serve "${MODEL}" --served-model-name qwen3-next \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --port "${PORT}"

echo "launched vllm on port ${PORT}, HIP device ${HIP_DEV}"
