"""Build and smoke-test network structures on one training fold; not a benchmark."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from .models import CONFIGS, make_surrogate, save_surrogate, load_surrogate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Lab6"))
import baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "network-architecture-check.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(8)
    ids, X, y = baseline.load_table(ROOT / "Lab6/dataset_same8/train.csv")
    dev, _ = train_test_split(np.arange(len(y)), test_size=.2, random_state=20261022)
    dev = np.sort(dev)
    fitting, _ = next(KFold(5, shuffle=True, random_state=2026).split(dev))
    fitting = dev[fitting]
    # Refit the quadratic on this fitting fold, never read the full-data model.npz.
    reference = baseline.make_model(100.).fit(X[fitting], y[fitting])
    quadratic = baseline.quadratic_form(reference)
    batch = torch.tensor(X[fitting[:128]], dtype=torch.float32, device=device)
    targets = torch.tensor(y[fitting[:128]], dtype=torch.float32, device=device)
    rows = []
    for name in CONFIGS:
        torch.manual_seed(20261022)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(20261022)
        model = make_surrogate(name, X[fitting], y[fitting], quadratic=quadratic).to(device)
        if name == "quadratic_residual":
            error = np.max(np.abs(model(batch).detach().cpu().numpy() - reference.predict(X[fitting[:128]])))
            assert error < 1e-3
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = (((model(batch) - targets) / model.target_scale) ** 2).mean()
            assert bool(torch.isfinite(loss))
            loss.backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            optimizer.step()
        assert any(not torch.equal(before[k], v) for k, v in model.named_parameters())
        model.eval()
        probe = batch[:2].detach().clone().requires_grad_(True)
        g = torch.autograd.grad(model(probe).sum(), probe)[0]
        assert g.shape == probe.shape and bool(torch.isfinite(g).all()) and bool(g.abs().sum() > 0)
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "network.pt"
            save_surrogate(model, file)
            loaded = load_surrogate(file, device)
            assert torch.equal(model(batch).detach(), loaded(batch).detach())
        row = {"name": name, "config": asdict(model.config), "trainable_parameters": model.parameter_count(),
               "device": str(device), "output_shape": list(model(batch).shape), "input_gradients_finite": True,
               "optimizer_updates_parameters": True, "checkpoint_roundtrip_exact": True}
        rows.append(row)
        print(name, row["trainable_parameters"], "parameters; gradient/optimizer/checkpoint checks passed", flush=True)
    report = {"purpose": "architecture and implementation smoke tests, NOT model selection or accuracy evaluation",
              "fitting_rows": len(fitting), "fitting_candidate_ids": [ids[int(i)] for i in fitting],
              "quadratic_refitted_on_fitting_rows": True, "teacher_full_model_used": False,
              "data_sha256": hashlib.sha256((ROOT / "Lab6/dataset_same8/train.csv").read_bytes()).hexdigest(),
              "seed": 20261022, "networks": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
