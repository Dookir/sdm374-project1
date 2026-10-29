import tempfile,unittest
from pathlib import Path
import numpy as np,torch
from bayes_experiment.training import fit as original_fit
from bayes_accel.training import fit,TrainingPaused,predict_rows
from bayes_experiment.common import CONFIGS
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(4);torch.use_deterministic_algorithms(True);r=np.random.default_rng(92);cls.X=r.dirichlet(np.ones(7),size=(180,3)).reshape(180,21);cls.y=100+30*cls.X[:,0]
 def test_matches_original_with_dropout(self):
  for base in CONFIGS.values():
   cfg=dict(base,learning_rate=.002,dropout=.2,weight_decay=.01)
   args=(self.X[:130],self.y[:130],cfg,1,2,'cpu');kw=dict(epochs=40,validation=(self.X[130:],self.y[130:]),patience=10)
   first,h1,m1=original_fit(*args,**kw);second,h2,m2=fit(*args,**kw)
   self.assertEqual(h1,h2)
   for key in first.state_dict():torch.testing.assert_close(first.state_dict()[key],second.state_dict()[key],atol=0,rtol=0)
   self.assertEqual(m1['selected_epoch'],m2['selected_epoch'])
 def test_exact_epoch_resume(self):
  cfg=dict(CONFIGS['relu_64_32'],learning_rate=.002,dropout=.2)
  args=(self.X[:130],self.y[:130],cfg,1,2,'cpu');kw=dict(epochs=35,validation=(self.X[130:],self.y[130:]),patience=600)
  expected,history,_=fit(*args,**kw)
  with tempfile.TemporaryDirectory() as directory:
   checkpoint=Path(directory)/'resume.pt';pause=Path(directory)/'pause.request';pause.touch()
   with self.assertRaises(TrainingPaused):fit(*args,**kw,resume_path=checkpoint,pause_path=pause)
   self.assertTrue(checkpoint.exists());pause.unlink()
   actual,history2,_=fit(*args,**kw,resume_path=checkpoint,pause_path=pause)
   self.assertEqual(history,history2);np.testing.assert_array_equal(predict_rows(expected,self.X),predict_rows(actual,self.X));self.assertFalse(checkpoint.exists())
if __name__=='__main__':unittest.main()
