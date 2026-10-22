"""Numerical regression check for the deterministic simplex projector."""
import unittest
import numpy as np
import torch
from Lab6.baseline import project_weights
from neural.score_candidates import project

class ProjectionTests(unittest.TestCase):
    def test_projection_matches_numpy_reference(self):
        rng=np.random.default_rng(2026)
        for dtype,tol in ((torch.float64,1e-12),(torch.float32,2e-6)):
            for x in (rng.normal(size=(33,21)),np.zeros((1,21)),np.ones((2,21)),np.full((2,21),-5.)):
                with self.subTest(dtype=dtype,rows=len(x)):
                    actual=project(torch.tensor(x,dtype=dtype)).numpy()
                    np.testing.assert_allclose(actual,project_weights(x),atol=tol,rtol=0)
                    np.testing.assert_allclose(actual.reshape(-1,3,7).sum(-1),1,atol=tol,rtol=0)
                    self.assertTrue((actual>=0).all())
if __name__=="__main__": unittest.main()
