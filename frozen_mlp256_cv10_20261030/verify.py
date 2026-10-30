import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd,torch
from model import load
from helpers import predict_rows
from frozen_data import load_table
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
 manifest=json.loads((ROOT/'manifest.json').read_text())
 for path,digest in manifest['files'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Frozen file changed: '+path)
 recipes=json.loads((ROOT/'recipes.json').read_text());ids,X,y=load_table(ROOT/'train.csv');torch.set_num_threads(4)
 coverage=np.zeros(len(y),dtype=int);errors=[];max_difference=0.;results=[]
 for recipe in recipes:
  folder=ROOT/'models'/f"fold{recipe['fold']:02d}";m=load(folder/'model.pt');assert m.config==recipe['config']
  a=np.asarray(recipe['train_indices']);b=np.asarray(recipe['test_indices']);assert not np.intersect1d(a,b).size;coverage[b]+=1
  assert [ids[i] for i in a]==recipe['train_candidate_ids'] and [ids[i] for i in b]==recipe['test_candidate_ids']
  tx=torch.tensor(X[a],dtype=torch.float64);ty=torch.tensor(y[a],dtype=torch.float64)
  stats=dict(x_mean=tx.mean(0),x_scale=tx.std(0,unbiased=False).clamp_min(1e-6),y_mean=ty.mean(),y_scale=ty.std(unbiased=False).clamp_min(1e-6))
  for key,value in stats.items():torch.testing.assert_close(getattr(m,key),value.float(),rtol=0,atol=0)
  for part,index in [('train',a),('test',b)]:
   frame=pd.read_csv(folder/f'{part}_predictions.csv',float_precision='round_trip',dtype={'candidate_id':str});pred=predict_rows(m,X[index]).astype(np.float64)
   np.testing.assert_array_equal(frame.row_index,index);np.testing.assert_array_equal(frame.candidate_id,np.asarray(ids)[index]);np.testing.assert_allclose(frame.y_true,y[index],rtol=0,atol=1e-12)
   difference=float(np.max(np.abs(pred-frame.y_pred.to_numpy())));max_difference=max(max_difference,difference)
   # This is a reference-inference portability tolerance, not a promise for cross-hardware retraining.
   np.testing.assert_allclose(pred,frame.y_pred.to_numpy(),rtol=0,atol=1e-3)
   stored_mae=float(np.abs(frame.y_pred.to_numpy()-y[index]).mean());assert abs(stored_mae-recipe['reference_'+part+'_mae'])<1e-9
   if part=='test':errors.extend(np.abs(pred-y[index]));results.append(dict(fold=recipe['fold'],test_mae=float(np.abs(pred-y[index]).mean())))
 assert np.array_equal(coverage,np.ones(len(y),dtype=int))
 oof=float(np.mean(errors));assert abs(oof-manifest['reference_oof_test_mae'])<1e-3
 report=dict(passed=True,frozen_file_hashes=True,models=10,no_retraining_performed=True,unique_test_rows=2048,oof_test_mae=oof,reference_oof_test_mae=manifest['reference_oof_test_mae'],max_prediction_difference=max_difference,reference_inference_atol=1e-3,folds=results)
 print(json.dumps(report,ensure_ascii=False,indent=2));return report
if __name__=='__main__':verify()
