import tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd,torch
from earlystop_experiment.common import TRAIN,CONFIGS,SEARCH,BASELINE,make_splits,validate_splits,summarize,sha,dump,read
from cv_experiment.common import make_splits as outer_splits
from earlystop_experiment.training import EarlyStopper,train_fold
from earlystop_experiment.run import verify_manifest,load_completed_training,export_predictions
from cv_experiment.training import save_atomic,predict_rows
from cv_experiment.optimize import refine
from neural_baseline.models import load
class EarlyStoppingTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(6)
  self.X=rng.dirichlet(np.ones(7),size=(128,3)).reshape(128,21);self.y=120+self.X[:,0]*20-self.X[:,5]*10;self.config=dict(hidden=[8,4],activation='relu',dropout=0.,weight_decay=.001)
 def fit(self,X=None,y=None,valX=None,valY=None):
  return train_fold(self.X[:96] if X is None else X,self.y[:96] if y is None else y,self.X[96:112] if valX is None else valX,self.y[96:112] if valY is None else valY,self.config,11,42,'cpu',epochs=8,patience=3,batch_size=16)
 def test_nested_splits_disjoint_and_outer_tests_identical(self):
  folds=make_splits(2048);validate_splits(folds,2048);old=outer_splits(2048)
  for f,o in zip(folds,old):
   self.assertEqual(f['test'],o['test']);self.assertEqual(f['outer_train'],o['train']);self.assertEqual(len(f['validation']),185)
  self.assertEqual(sum(len(f['train']) for f in folds),16582);self.assertEqual(TRAIN['patience'],30);self.assertEqual(TRAIN['epochs'],600)
 def test_patience_and_best_state_restored_not_last_state(self):
  model=torch.nn.Linear(1,1,bias=False);stopper=EarlyStopper(patience=3);flags=[]
  for epoch,value in enumerate([5.,4.,4.1,4.2,4.3],1):
   with torch.no_grad():model.weight.fill_(epoch)
   flags.append(stopper.update(epoch,value,model))
  self.assertEqual(flags,[False,False,False,False,True]);self.assertEqual(stopper.best_epoch,2);stopper.restore(model);self.assertEqual(float(model.weight.detach()),2.)
 def test_same_seed_reproducible_and_outer_test_cannot_affect_training(self):
  first,h1,s1=self.fit();X=self.X.copy();y=self.y.copy();X[112:]*=1000;y[112:]+=10000
  second,h2,s2=self.fit(X=X[:96],y=y[:96],valX=X[96:112],valY=y[96:112]);self.assertEqual(h1,h2);self.assertEqual(s1,s2)
  for key,v in first.state_dict().items():torch.testing.assert_close(v,second.state_dict()[key],atol=0,rtol=0)
 def test_validation_never_fits_normalization_statistics(self):
  first,_,_=self.fit();second,_,_=self.fit(valX=self.X[96:112]*500,valY=self.y[96:112]+10000)
  for key in ['x_mean','x_scale','y_mean','y_scale']:torch.testing.assert_close(getattr(first,key),getattr(second,key),atol=0,rtol=0)
  torch.testing.assert_close(first.x_mean,torch.tensor(self.X[:96].mean(0),dtype=torch.float32))
 def test_returned_model_matches_lowest_validation_epoch(self):
  model,history,info=self.fit();self.assertEqual(info['selected_epoch'],min(history,key=lambda h:h['validation_mae'])['epoch'])
  value=float(np.abs(predict_rows(model,self.X[96:112]).astype(float)-self.y[96:112]).mean());self.assertAlmostEqual(value,info['validation_mae_best'],places=4)
 def test_exports_and_checkpoint_stage_integrity(self):
  model,_,_=self.fit();fold=dict(train=list(range(96)),validation=list(range(96,112)),test=list(range(112,128)))
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);save_atomic(model,folder/'model.pt');metrics=export_predictions(folder,model,[str(i) for i in range(128)],self.X,self.y,fold)
   np.testing.assert_allclose(predict_rows(model,self.X),predict_rows(load(folder/'model.pt'),self.X),atol=0,rtol=0)
   for split in ['train','validation','test']:
    frame=pd.read_csv(folder/(split+'_predictions.csv'),float_precision='round_trip');self.assertAlmostEqual(metrics[split+'_mae'],float(np.abs(frame.y_pred-frame.y_true).mean()),places=10)
   dump(folder/'training_complete.json',dict(files={'model.pt':sha(folder/'model.pt')}));load_completed_training(folder);(folder/'model.pt').write_bytes(b'corrupt')
   with self.assertRaises(ValueError):load_completed_training(folder)
 def test_resume_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);dump(folder/'splits.json',[]);dump(folder/'manifest.json',dict(version=1,protected_files={},splits_sha256=sha(folder/'splits.json')))
   self.assertIsNotNone(verify_manifest(folder,{'version':1},True))
   with self.assertRaises(ValueError):verify_manifest(folder,{'version':2},True)
 def test_search_uses_fit_rows_and_preserves_saved_checkpoint(self):
  model,_,_=self.fit()
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'model.pt';save_atomic(model,path);digest=sha(path);settings=dict(SEARCH,seed_count=4,pgd_iterations=3,adam_iterations=3,coordinate_rounds=1,slsqp_starts=1,slsqp_iterations=2)
   weights,info=refine(path,self.X[:96],self.y[:96],99,'cpu',settings=settings)
   self.assertEqual(digest,sha(path));self.assertTrue((weights>=0).all());np.testing.assert_allclose(weights.sum(1),1,atol=1e-12);self.assertLessEqual(info['after'],info['before'])
   anchor=np.asarray(info['reference_weights']).ravel();self.assertTrue(np.any(np.all(self.X[:96]==anchor,axis=1)))
 def test_summary_separates_fit_validation_test_and_reference(self):
  rows=[dict(model='relu_64_32',fold=1,n_train=9,n_validation=2,n_test=1,train_abs_error_sum=9.,validation_abs_error_sum=4.,test_abs_error_sum=2.,test_mae=2.,search_after=50.,selected_epoch=4,epochs_run=34),dict(model='relu_64_32',fold=2,n_train=8,n_validation=1,n_test=2,train_abs_error_sum=24.,validation_abs_error_sum=5.,test_abs_error_sum=8.,test_mae=4.,search_after=40.,selected_epoch=8,epochs_run=38)]
  row=summarize(rows)[0];self.assertAlmostEqual(row['training_mae'],33/17);self.assertAlmostEqual(row['validation_mae'],3.);self.assertAlmostEqual(row['test_mae_oof'],10/3);self.assertEqual(row['selected_epoch_mean'],6.)
if __name__=='__main__':unittest.main()
