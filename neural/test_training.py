import unittest
from types import SimpleNamespace
import numpy as np
from neural.train import make_splits,fit_network
from neural.training_metrics import metrics,oof_metrics

class TrainingTests(unittest.TestCase):
    def test_splits(self):
        splits=make_splits(2048)
        self.assertEqual(sorted(np.concatenate([s[1] for s in splits]).tolist()),list(range(2048)))
        for train,test,fit,stop in splits:
            self.assertFalse(set(train)&set(test)); self.assertFalse(set(fit)&set(stop))
            self.assertEqual(set(train),set(fit)|set(stop))
    def test_true_decision_metric(self):
        m=metrics(np.array([10.,20.,30.,40.]),np.array([5.,4.,2.,1.]))
        self.assertEqual(m["top2_true"],30.); self.assertEqual(m["regret_at2"],20.)
        self.assertEqual(m["selected_prediction_bias"],-33.5)
    def test_fold_candidate_pools(self):
        m=oof_metrics(np.array([10.,20.,30.,100.,110.,120.]),np.array([3.,2.,1.,3.,2.,1.]),np.array([0,0,0,1,1,1]))
        self.assertEqual(m["top2_true"],65.); self.assertEqual(m["regret_at2"],10.)
    def test_outer_labels_not_used(self):
        rng=np.random.default_rng(1); X=rng.dirichlet(np.ones(7),size=(30,3)).reshape(30,21); y=np.arange(30.)+100
        a=SimpleNamespace(device="cpu",lr=.001,weight_decay=.0001,max_epochs=3,eval_every=1,min_delta=.001,patience=3,batch_size=8)
        m,h=fit_network("mlp_small",X,y,np.arange(20),np.arange(20,25),None,1,a)
        y2=y.copy(); y2[25:]=1e9
        n,j=fit_network("mlp_small",X,y2,np.arange(20),np.arange(20,25),None,1,a)
        self.assertEqual(h["best_epoch"],j["best_epoch"])
        for k,v in m.state_dict().items(): self.assertTrue(v.equal(n.state_dict()[k]))
        _,r=fit_network("mlp_small",X,y,np.arange(25),None,None,1,a,fixed_epochs=2)
        self.assertEqual(r["epochs_run"],2); self.assertEqual(r["history"],[])
if __name__=="__main__": unittest.main()
