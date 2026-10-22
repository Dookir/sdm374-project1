#!/usr/bin/env bash
# Source this file; no global shell or Python configuration is changed.
TASK_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_WORK_DIR="$(dirname "$TASK_PROJECT_DIR")"
source "$TASK_WORK_DIR/.venv/bin/activate"
export PATH="$TASK_WORK_DIR/.tools:$PATH"
export UV_PYTHON_INSTALL_DIR="$TASK_WORK_DIR/.python"
export UV_PYTHON_BIN_DIR="$TASK_WORK_DIR/.tools"
export UV_CACHE_DIR="$TASK_WORK_DIR/.cache/uv"
export UV_SYSTEM_CERTS=true
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MPLBACKEND=Agg
cd "$TASK_PROJECT_DIR"
