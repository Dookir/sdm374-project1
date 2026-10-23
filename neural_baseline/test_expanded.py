import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from neural_baseline.models import MLP,save,load,CONFIGS as OLD
from neural_baseline.expanded_configs import CONFIGS
class ExpandedTests(unittest.TestCase):
    def test_all_larger_models_keep_input_gradients_and_checkpoint(self):
        rng=np.random.default_rng(2); X=rng.dirichlet(np.ones(7),size=(8,3)).reshape(8,21); y=np.arange(8.)+100
        for name,cfg in CONFIGS.items():
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                m=MLP(cfg).fit_stats(X,y).eval(); x=torch.tensor(X,dtype=torch.float32,requires_grad=True)
                p=m(x); g=torch.autograd.grad(p.sum(),x)[0]
                self.assertTrue(torch.isfinite(p).all() and torch.isfinite(g).all())
                self.assertEqual(p.shape,(8,)); self.assertGreater(float(g.abs().sum()),0)
                path=Path(d)/"m.pt"; save(m,path); restored=load(path)
                self.assertTrue(torch.equal(m(x),restored(x)))
    def test_declared_capacity_and_optimizer_range(self):
        self.assertEqual(len(CONFIGS),12)
        maximum=max(sum(p.numel() for p in MLP(c).parameters()) for c in CONFIGS.values())
        old_max=max(sum(p.numel() for p in MLP(c).parameters()) for c in OLD.values())
        self.assertGreater(maximum,14*old_max)
        self.assertEqual({c["learning_rate"] for c in CONFIGS.values()},{.0003,.001,.003})
if __name__=="__main__": unittest.main()
