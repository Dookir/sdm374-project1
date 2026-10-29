#!/usr/bin/env bash
set -euo pipefail
TASK_BAYES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_BAYES_ROOT"
source server/activate.sh
export CUDA_VISIBLE_DEVICES="${SDM374_GPU_UUID:-GPU-e9b4fef6-be25-1e26-6b9a-753c8d74ef66}"
if [[ "$CUDA_VISIBLE_DEVICES" != GPU-* || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo 'Exactly one full GPU UUID is required' >&2; exit 1
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_MPS_PIPE_DIRECTORY="${SDM374_MPS_PIPE:-/tmp/sdm374-bayes-mps-run-20261029}"
export CUDA_MPS_LOG_DIRECTORY="${SDM374_MPS_LOG:-/data/zengxiong/wuxinkai/bayes-mps-run-20261029-logs}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
chmod 700 "$CUDA_MPS_PIPE_DIRECTORY"
if [[ -f "$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid" ]]; then
  echo 'Private MPS controller already exists; refusing duplicate launch' >&2; exit 1
fi
nvidia-cuda-mps-control -d
cleanup() { printf 'quit\n' | nvidia-cuda-mps-control || true; }
trap cleanup EXIT
python -u -m bayes_accel.run "$@"
