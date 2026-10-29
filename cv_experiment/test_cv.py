import tempfile,unittest
from pathlib import Path
import numpy as np,torch
from cv_experiment.common import CONFIGS,TRAIN,SEARCH,make_splits,validate_splits,summarize,sha,dump
from cv_experiment.training import train_fold,save_atomic,predict_rows
from cv_experiment.run import verify_manifest,load_completed_training,export_predictions
from cv_experiment.optimize import refine
from neural_baseline.models import load
class CVTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(5)
  self.X=rng.dirichlet(np.ones(7),size=(128,3)).reshape(128,21);self.y=120+self.X[:,0]*20-self.X[:,5]*10;self.config=dict(hidden=[8,4],activation='relu',dropout=0.,weight_decay=.001)
 def test_partition_coverage_and_shared_seeds(self):
  folds=make_splits(2048);validate_splits(folds,2048);self.assertEqual([len(f['test']) for f in folds],[205]*8+[204]*2);self.assertEqual(folds,make_splits(2048))
  self.assertEqual(len(CONFIGS),3);self.assertEqual(TRAIN['epochs'],600)
 def test_training_is_deterministic_and_heldout_perturbation_is_irrelevant(self):
  train=np.arange(96);test=np.arange(96,128)
  first,_=train_fold(self.X[train],self.y[train],self.config,12,99,'cpu',epochs=3,batch_size=16)
  alteredX=self.X.copy();alteredY=self.y.copy();alteredX[test]*=1000;alteredY[test]+=100000
  second,_=train_fold(alteredX[train],alteredY[train],self.config,12,99,'cpu',epochs=3,batch_size=16)
  for k,v in first.state_dict().items():torch.testing.assert_close(v,second.state_dict()[k],atol=0,rtol=0)
  torch.testing.assert_close(first.x_mean,torch.tensor(self.X[train].mean(0),dtype=torch.float32))
  torch.testing.assert_close(first.y_mean,torch.tensor(self.y[train].mean(),dtype=torch.float32))
 def test_checkpoint_resume_stage_and_corruption_detection(self):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);m,_=train_fold(self.X,self.y,self.config,1,2,'cpu',epochs=2);save_atomic(m,folder/'model.pt');p=predict_rows(m,self.X)
   np.testing.assert_allclose(p,predict_rows(load(folder/'model.pt'),self.X),atol=0,rtol=0)
   dump(folder/'training_complete.json',dict(files={'model.pt':sha(folder/'model.pt')}));load_completed_training(folder)
   (folder/'model.pt').write_bytes(b'corrupted')
   with self.assertRaises(ValueError):load_completed_training(folder)
 def test_resume_manifest_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);dump(folder/'splits.json',[]);dump(folder/'manifest.json',dict(version=1,protected_files={},splits_sha256=sha(folder/'splits.json')))
   self.assertIsNotNone(verify_manifest(folder,{'version':1},True))
   with self.assertRaises(ValueError):verify_manifest(folder,{'version':2},True)
   with self.assertRaises(ValueError):verify_manifest(folder,{'version':1},False)
 def test_weighted_train_and_oof_aggregation(self):
  rows=[dict(model='relu_64_32',fold=1,n_train=9,n_test=1,train_abs_error_sum=9.,test_abs_error_sum=2.,test_mae=2.,search_after=50.),dict(model='relu_64_32',fold=2,n_train=8,n_test=2,train_abs_error_sum=24.,test_abs_error_sum=8.,test_mae=4.,search_after=40.)]
  row=summarize(rows)[0];self.assertAlmostEqual(row['training_mae'],33/17);self.assertAlmostEqual(row['test_mae_oof'],10/3);self.assertAlmostEqual(row['test_fold_mae_std'],np.sqrt(2));self.assertEqual(row['best_fold'],2)
 def test_prediction_exports_recompute_train_and_heldout_mae(self):
  import pandas as pd
  model,_=train_fold(self.X[:96],self.y[:96],self.config,3,5,'cpu',epochs=2)
  fold=dict(train=list(range(96)),test=list(range(96,128)))
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);metrics=export_predictions(folder,model,[str(i) for i in range(128)],self.X,self.y,fold)
   for split in ['train','test']:
    frame=pd.read_csv(folder/(split+'_predictions.csv'),float_precision='round_trip')
    np.testing.assert_array_equal(frame.row_index,fold[split]);self.assertAlmostEqual(metrics[split+'_mae'],float(np.abs(frame.y_pred-frame.y_true).mean()),places=10)
 def test_search_preserves_checkpoint_and_legal_nonregressing_weights(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'model.pt';model,_=train_fold(self.X,self.y,self.config,2,3,'cpu',epochs=2);save_atomic(model,path);digest=sha(path)
   settings=dict(SEARCH,seed_count=4,pgd_iterations=3,adam_iterations=3,coordinate_rounds=1,slsqp_starts=1,slsqp_iterations=2)
   weights,result=refine(path,self.X,self.y,44,'cpu',settings=settings)
   self.assertEqual(sha(path),digest);self.assertEqual(weights.shape,(3,7));self.assertTrue((weights>=0).all());np.testing.assert_allclose(weights.sum(1),1,atol=1e-12)
   self.assertLessEqual(result['after'],result['before']);self.assertGreater(result['stages'][0]['candidates'],65000)
if __name__=='__main__':unittest.main()
