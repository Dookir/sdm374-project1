import tempfile,unittest
from pathlib import Path
import numpy as np,torch
from tabular_next.models import CONFIGS,Model,fit_stats,save,load
from tabular_next.run import project
from Lab6 import baseline
class ModelTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2); torch.manual_seed(1); rng=np.random.default_rng(2)
  self.X=rng.dirichlet(np.ones(7),size=(64,3)).reshape(64,21); self.y=100+self.X[:,0]*20; self.stats=fit_stats(self.X,self.y)
 def test_all_models_gradient_heads_and_roundtrip(self):
  for name,cfg in CONFIGS.items():
   with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
    m=Model(cfg,self.stats).eval(); x=torch.tensor(self.X[:4],dtype=torch.float32,requires_grad=True)
    heads=m.normalized_heads(x); self.assertEqual(heads.shape,(4,cfg.get('k',1)))
    pred=m(x); torch.testing.assert_close(pred,m.y_mean+m.y_scale*heads.mean(1))
    grad=torch.autograd.grad(pred.sum(),x,retain_graph=True)[0]; self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(float(grad.abs().sum()),0)
    target=torch.tensor((self.y[:4]-self.stats['y_mean'])/self.stats['y_scale'],dtype=torch.float32)[:,None]
    loss=(heads-target).square().mean(); loss.backward(); self.assertTrue(any(p.grad is not None and bool((p.grad!=0).any()) for p in m.parameters()))
    save(m,Path(tmp)/'m.pt'); r=load(Path(tmp)/'m.pt'); torch.testing.assert_close(m(x),r(x),atol=0,rtol=0)
 def test_input_gradient_numerical(self):
  for name in ['res_periodic128','res_ple128','tabm_raw128_k8','tabm_ple128_k16','tabm_periodic256_k16']:
   with self.subTest(name=name):
    m=Model(CONFIGS[name],self.stats).double().eval(); x=torch.tensor(self.X[1:2],dtype=torch.float64,requires_grad=True)
    self.assertTrue(torch.autograd.gradcheck(m,(x,),eps=1e-6,atol=1e-4,rtol=1e-3))
 def test_projection(self):
  x=np.random.default_rng(1).normal(size=(50,21)); np.testing.assert_allclose(project(torch.tensor(x)).numpy(),baseline.project_weights(x),atol=1e-12,rtol=0)
if __name__=='__main__':unittest.main()
