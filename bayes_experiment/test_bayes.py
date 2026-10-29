import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from bayes_experiment.common import CONFIGS,TRAIN,make_splits,inner_splits,read
from bayes_experiment.training import fit,predict_rows,save_atomic
from neural_baseline.models import load
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);r=np.random.default_rng(19);cls.X=r.dirichlet(np.ones(7),size=(90,3)).reshape(90,21);cls.y=100+cls.X[:,0]*20+cls.X[:,8]*10
 def test_nested_splits(self):
  for outer in make_splits(2048):
   parts=inner_splits(outer);seen=[]
   for part in parts:
    self.assertFalse(set(part['train'])&set(part['validation']));self.assertEqual(set(part['train'])|set(part['validation']),set(outer['train']));seen+=part['validation']
   self.assertEqual(sorted(seen),outer['train'])
 def test_dropout_decay_learning_rate_effective(self):
  base=dict(CONFIGS['relu_64_32'],learning_rate=.001,weight_decay=.01,dropout=.1)
  model,h,meta=fit(self.X[:60],self.y[:60],base,1,2,'cpu',epochs=4)
  self.assertTrue(any(isinstance(m,torch.nn.Dropout) and m.p==.1 for m in model.modules()))
  self.assertEqual(meta['actual_optimizer'],dict(learning_rate=.001,weight_decay=.01));self.assertFalse(model.training)
  changed=dict(base,learning_rate=.002,weight_decay=.03)
  other,_,_=fit(self.X[:60],self.y[:60],changed,1,2,'cpu',epochs=4)
  self.assertFalse(np.array_equal(predict_rows(model,self.X),predict_rows(other,self.X)))
 def test_determinism_checkpoint_train_stats(self):
  cfg=dict(CONFIGS['relu_64_32'],learning_rate=.001,dropout=.1)
  first,history,meta=fit(self.X[:60],self.y[:60],cfg,1,2,'cpu',epochs=8,validation=(self.X[60:80],self.y[60:80]),patience=3)
  second,history2,_=fit(self.X[:60],self.y[:60],cfg,1,2,'cpu',epochs=8,validation=(self.X[60:80],self.y[60:80]),patience=3)
  self.assertEqual(history,history2);np.testing.assert_array_equal(predict_rows(first,self.X),predict_rows(second,self.X))
  np.testing.assert_array_equal(first.x_mean.numpy(),self.X[:60].mean(0).astype(np.float32))
  self.assertEqual(meta['selected_epoch'],1+int(np.argmin([r['validation_mae'] for r in history])))
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)/'model.pt';save_atomic(first,p);np.testing.assert_array_equal(predict_rows(first,self.X),predict_rows(load(p),self.X))
 def test_tuning_resume(self):
  from bayes_experiment import tuning
  outer=make_splits(90)[0];original=tuning.fit
  def short(*args,**kwargs):return original(*args,**dict(kwargs,epochs=3,patience=2))
  with tempfile.TemporaryDirectory() as directory,patch.dict(TRAIN,trials=2),patch.object(tuning,'fit',side_effect=short):
   folder=Path(directory);result=tuning.tune(folder,CONFIGS['relu_64_32'],outer,self.X,self.y,'cpu',lambda *a:None)
   self.assertEqual(result['trials'],2);self.assertEqual(result['refit_epochs'],int(np.median(result['selected_epochs'])))
   self.assertEqual(tuning.tune(folder,CONFIGS['relu_64_32'],outer,self.X,self.y,'cpu',lambda *a:None),result)
   # Simulate a crash after trial artifacts were committed but before tell().
   import optuna
   study=optuna.load_study(study_name='tuning',storage='sqlite:///'+str(folder/'study.db'))
   self.assertEqual(len(study.trials),2)
 def test_interrupted_trial_restarts_only_unfinished_inner(self):
  from bayes_experiment import tuning
  outer=make_splits(90)[0];original=tuning.fit;calls=[]
  def short(*args,**kwargs):
   calls.append(1)
   if len(calls)==2:raise RuntimeError('intentional interruption')
   return original(*args,**dict(kwargs,epochs=3,patience=2))
  with tempfile.TemporaryDirectory() as directory,patch.dict(TRAIN,trials=1),patch.object(tuning,'fit',side_effect=short):
   folder=Path(directory)
   with self.assertRaisesRegex(RuntimeError,'intentional'):tuning.tune(folder,CONFIGS['relu_64_32'],outer,self.X,self.y,'cpu',lambda *a:None)
   digest=(folder/'trial00/inner01/complete.json').read_bytes()
   result=tuning.tune(folder,CONFIGS['relu_64_32'],outer,self.X,self.y,'cpu',lambda *a:None)
   self.assertEqual(len(calls),4);self.assertEqual((folder/'trial00/inner01/complete.json').read_bytes(),digest);self.assertEqual(result['best_trial'],0)
 def test_outer_test_labels_cannot_affect_tuning(self):
  from bayes_experiment import tuning
  outer=make_splits(90)[0];original=tuning.fit;changed=self.y.copy();changed[outer['test']]+=100000
  def short(*args,**kwargs):return original(*args,**dict(kwargs,epochs=3,patience=2))
  with tempfile.TemporaryDirectory() as directory,patch.dict(TRAIN,trials=2),patch.object(tuning,'fit',side_effect=short):
   a=tuning.tune(Path(directory)/'a',CONFIGS['relu_64_32'],outer,self.X,self.y,'cpu',lambda *a:None)
   b=tuning.tune(Path(directory)/'b',CONFIGS['relu_64_32'],outer,self.X,changed,'cpu',lambda *a:None)
   self.assertEqual(a,b)
 def test_complete_internal_audit(self):
  from bayes_experiment import tuning
  from bayes_experiment.audit_tuning import audit_tuning
  outer=make_splits(90)[0]
  with tempfile.TemporaryDirectory() as directory,patch.dict(TRAIN,trials=1):
   folder=Path(directory);result=tuning.tune(folder,CONFIGS['relu_64_32'],outer,self.X,self.y,'cpu',lambda *a:None)
   self.assertEqual(audit_tuning(folder,outer,CONFIGS['relu_64_32'],self.X,self.y),result)
 def test_assignment_file_interruption_reuses_pending_trial(self):
  from bayes_experiment import tuning
  import optuna
  outer=make_splits(90)[0];original=tuning.fit
  def short(*args,**kwargs):return original(*args,**dict(kwargs,epochs=3,patience=2))
  with tempfile.TemporaryDirectory() as directory,patch.dict(TRAIN,trials=1),patch.object(tuning,'fit',side_effect=short):
   folder=Path(directory);storage='sqlite:///'+str(folder/'study.db')
   study=optuna.create_study(storage=storage,study_name='tuning');study.enqueue_trial(dict(learning_rate=.001,weight_decay=.001,dropout=0.));pending=study.ask(fixed_distributions=tuning.space())
   result=tuning.tune(folder,CONFIGS['relu_64_32'],outer,self.X,self.y,'cpu',lambda *a:None)
   self.assertEqual(result['best_trial'],pending.number);self.assertEqual(len(optuna.load_study(study_name='tuning',storage=storage).trials),1)
if __name__=='__main__':unittest.main()
