"""Remote environment smoke check; no project training or official evaluation."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import gpytorch
import optuna
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Lab6"))
import baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "server-environment.json")
    parser.add_argument("--catboost-gpu", action="store_true")
    args = parser.parse_args()
    assert sys.version_info >= (3, 12)
    assert torch.cuda.is_available(), "CUDA is unavailable"
    torch.set_num_threads(8)
    torch.manual_seed(20261022)
    torch.cuda.manual_seed_all(20261022)
    device = torch.device("cuda:0")
    x_np = np.random.default_rng(20261022).normal(size=(64, 21)).astype(np.float32)
    x = torch.from_numpy(x_np).to(device)
    y = x.sum(dim=1, keepdim=True) / 21
    model = torch.nn.Sequential(torch.nn.Linear(21, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)).to(device)
    original = next(model.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        optimizer.step()
    torch.cuda.synchronize()
    assert not torch.equal(original, next(model.parameters()).detach())
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=21)).to(device)
    covariance = kernel(x).to_dense() + torch.eye(len(x), device=device) * 0.001
    cholesky = torch.linalg.cholesky(covariance)
    gp_loss = cholesky.diag().log().sum()
    gp_loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in kernel.parameters())
    torch.cuda.synchronize()
    ids, X, labels = baseline.load_table(ROOT / "Lab6/dataset_same8/train.csv")
    assert X.shape == (2048, 21)
    weights = json.loads((ROOT / "Lab6/weights.json").read_text())
    predicted = baseline.predict(weights, ROOT / "Lab6/model.npz")
    assert abs(predicted - 75.89570106874649) < 1e-8
    cpu_model = CatBoostRegressor(iterations=5, depth=3, verbose=False, allow_writing_files=False, thread_count=4)
    cpu_model.fit(X[:64], labels[:64])
    assert np.isfinite(cpu_model.predict(X[64:68])).all()
    gpu_check = None
    if args.catboost_gpu:
        gpu_model = CatBoostRegressor(iterations=5, depth=3, task_type="GPU", devices="0", gpu_ram_part=0.02,
                                    verbose=False, allow_writing_files=False, thread_count=4)
        gpu_model.fit(X[:64], labels[:64])
        gpu_check = bool(np.isfinite(gpu_model.predict(X[64:68])).all())
        assert gpu_check
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib", "torch", "gpytorch", "optuna", "ipykernel"]
    result = {"status": "passed", "purpose": "installation smoke test, not a model-quality experiment",
              "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in packages},
              "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu_name": torch.cuda.get_device_name(0), "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
              "neural_forward_backward_and_optimizer": True, "numpy_torch_interop": True,
              "gpytorch_cuda_covariance_and_cholesky": True,
              "catboost_cpu": True, "catboost_gpu": gpu_check, "training_rows": len(labels),
              "teacher_baseline_prediction": float(predicted),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
