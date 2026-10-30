#!/usr/bin/env bash
set -euo pipefail
TASK_FINAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_FINAL_ROOT"
source server/activate.sh
# choose_gpu checks all cards but never initializes CUDA on them.
if ! TASK_FINAL_UUID=$(CUDA_VISIBLE_DEVICES= python -m final_mlp256.run --choose-gpu); then
  echo 'PAUSED: no idle GPU; no training launched'; exit 3
fi
export CUDA_VISIBLE_DEVICES="$TASK_FINAL_UUID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_MPS_PIPE_DIRECTORY=/tmp/sdm374-final256-mps-20261030
export CUDA_MPS_LOG_DIRECTORY=/data/zengxiong/wuxinkai/final256-mps-20261030-logs
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
chmod 700 "$CUDA_MPS_PIPE_DIRECTORY"
if [[ -f "$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid" ]]; then echo 'Private MPS already running; refusing duplicate launch'; exit 1; fi
nvidia-cuda-mps-control -d
cleanup() { printf 'quit\n' | nvidia-cuda-mps-control || true; }
trap cleanup EXIT
python -u -m final_mlp256.run "$@"
