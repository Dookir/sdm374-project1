import unittest
import numpy as np,torch
from weight_search.run import mixed_candidates,gradient_refine,coordinate_refine
class Bowl(torch.nn.Module):
 def forward(self,w):return ((w-1/7)**2).sum(1)+10
class SearchTests(unittest.TestCase):
 def test_candidates_and_optimization_preserve_constraints_and_incumbent(self):
  torch.set_num_threads(2); rng=np.random.default_rng(4); X=rng.dirichlet(np.ones(7),size=(64,3)).reshape(64,21); y=np.arange(64.)
  points=mixed_candidates(rng,X,y,X[0],X[:3]); self.assertGreater(len(points),65000); self.assertTrue((points>=0).all()); np.testing.assert_allclose(points.reshape(-1,3,7).sum(2),1,atol=1e-12)
  model=Bowl(); seeds=torch.tensor(X[:8],dtype=torch.float32); before=float(model(seeds).min())
  for adam in [False,True]:
   best,scores=gradient_refine(model,seeds,50,adam); self.assertLess(float(scores.min()),before); self.assertGreaterEqual(float(best.min()),0); torch.testing.assert_close(best.reshape(-1,3,7).sum(2),torch.ones((len(best),3)))
  best,value,_=coordinate_refine(model,seeds[0],rounds=2); self.assertLessEqual(value,float(model(seeds[:1])))
if __name__=='__main__':unittest.main()
