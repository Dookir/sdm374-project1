import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from Lab6 import baseline
from neural_baseline.models import CONFIGS,MLP,save,load
from neural_baseline.run import project
class NeuralTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(1); self.X=rng.dirichlet(np.ones(7),size=(32,3)).reshape(32,21); self.y=np.arange(32.)+100
    def test_forward_input_gradient_and_roundtrip(self):
        for name,config in CONFIGS.items():
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                torch.manual_seed(1); m=MLP(config).fit_stats(self.X,self.y).eval()
                x=torch.tensor(self.X,dtype=torch.float32,requires_grad=True)
                p=m(x); self.assertEqual(p.shape,(32,)); self.assertTrue(torch.isfinite(p).all())
                g=torch.autograd.grad(p.sum(),x)[0]; self.assertTrue(torch.isfinite(g).all()); self.assertGreater(float(g.abs().sum()),0)
                save(m,Path(d)/"m.pt"); r=load(Path(d)/"m.pt")
                self.assertTrue(torch.equal(m(x),r(x)))
                torch.testing.assert_close(m.x_mean,torch.tensor(self.X.mean(0),dtype=torch.float32))
    def test_projection_matches_original(self):
        rng=np.random.default_rng(2026)
        for x in (rng.normal(size=(99,21)),np.zeros((1,21)),np.full((1,21),-5.)):
            actual=project(torch.tensor(x,dtype=torch.float64)).numpy()
            np.testing.assert_allclose(actual,baseline.project_weights(x),atol=1e-12,rtol=0)
            np.testing.assert_allclose(actual.reshape(-1,3,7).sum(-1),1,atol=1e-12,rtol=0)
            self.assertTrue((actual>=0).all())
    def test_tanh_input_gradient_numerical(self):
        m=MLP(CONFIGS["mlp128_tanh"]).fit_stats(self.X,self.y).double().eval()
        x=torch.tensor(self.X[:1],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(m,(x,),eps=1e-6,atol=1e-4,rtol=1e-3))
if __name__=="__main__": unittest.main()
