import unittest,tempfile
from pathlib import Path
import numpy as np,torch
from final_mlp256.run import idle_gpu_rows,definition,export_final,CONFIG
from bayes_experiment.common import inner_splits,read
from neural_baseline.models import MLP,save
class Tests(unittest.TestCase):
 def test_only_truly_idle_cards(self):
  self.assertEqual(idle_gpu_rows('GPU-busy, 20000, 0\nGPU-compute, 0, 100\nGPU-idle, 0, 0'),['GPU-idle'])
  self.assertEqual(idle_gpu_rows('GPU-busy, 100, 0'),[])
 def test_full_data_selection_partitions(self):
  outer=definition(2048);self.assertEqual(outer['test'],[]);seen=[]
  for part in inner_splits(outer):
   self.assertFalse(set(part['train'])&set(part['validation']));self.assertEqual(set(part['train'])|set(part['validation']),set(range(2048)));seen+=part['validation']
  self.assertEqual(sorted(seen),list(range(2048)))
 def test_export_never_labels_training_as_test(self):
  rng=np.random.default_rng(2);X=rng.normal(size=(30,21));y=rng.normal(size=30);torch.set_num_threads(1)
  selected=dict(parameters=dict(learning_rate=.001,weight_decay=.001,dropout=.1),inner_validation_mae=5.,refit_epochs=20)
  model=MLP(dict(CONFIG,**selected['parameters'])).fit_stats(X,y).eval()
  with tempfile.TemporaryDirectory() as directory:
   out=Path(directory);save(model,out/'model.pt')
   for p in ['history.json','fit_metadata.json','selection.json']:(out/p).write_text('{}')
   record=export_final(out,model,[str(i) for i in range(30)],X,y,selected)
   self.assertIsNone(record['independent_test_mae']);self.assertFalse(record['reference_cv_is_this_models_test_score']);self.assertEqual(record['n_training_rows'],30)
if __name__=='__main__':unittest.main()
