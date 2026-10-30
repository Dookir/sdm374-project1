import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def audit(out):
 import numpy as np,pandas as pd,torch
 from final_mlp256.run import definition,CONFIG,expected
 from bayes_experiment.common import SEED,data,read,dump,sha,inner_splits,validate_files
 from bayes_experiment.audit_tuning import audit_tuning
 from neural_baseline.models import load
 from cv_experiment.training import predict_rows
 torch.set_num_threads(4);ids,X,y=data();manifest=read(out/'manifest.json');record=read(out/'complete.json');validate_files(out,record)
 for k,v in expected(ids).items():assert manifest[k]==v,k
 for path,digest in (manifest['source_sha256']|manifest['protected_files']).items():assert sha(ROOT/path)==digest,path
 selected=audit_tuning(out/'tuning'/'relu_256_128'/'fold01',definition(len(y)),CONFIG,X,y)
 assert selected==read(out/'selection.json') and read(out/'internal_splits.json')==inner_splits(definition(len(y)))
 assert len(list((out/'tuning').glob('*/fold*/trial*/inner*/complete.json')))==90
 model=load(out/'model.pt');assert model.config==dict(CONFIG,**selected['parameters'])==record['selected_config']
 tx=torch.tensor(X,dtype=torch.float64);ty=torch.tensor(y,dtype=torch.float64);stats=dict(x_mean=tx.mean(0),x_scale=tx.std(0,unbiased=False).clamp_min(1e-6),y_mean=ty.mean(),y_scale=ty.std(unbiased=False).clamp_min(1e-6))
 for key,value in stats.items():torch.testing.assert_close(getattr(model,key),value.float(),atol=0,rtol=0)
 history=read(out/'history.json');assert len(history)==record['epochs']==selected['refit_epochs']==int(np.median(selected['selected_epochs']))
 import math
 for k,h in enumerate(history,1):assert h['epoch']==k and abs(h['learning_rate']-model.config['learning_rate']*(.01+.99*(1+math.cos(math.pi*k/600))/2))<1e-12
 frame=pd.read_csv(out/'training_predictions.csv',float_precision='round_trip',dtype={'candidate_id':str});np.testing.assert_array_equal(frame.candidate_id,ids);np.testing.assert_array_equal(frame.row_index,np.arange(len(y)))
 pred=predict_rows(model,X).astype(np.float64);np.testing.assert_allclose(pred,frame.y_pred,atol=1e-4,rtol=0);np.testing.assert_allclose(frame.y_true,y,atol=1e-12,rtol=0)
 assert abs(float(np.abs(frame.y_pred.to_numpy()-y).mean())-record['training_mae'])<1e-9
 assert record['independent_test_mae'] is None and not record['reference_cv_is_this_models_test_score'] and not record['ramping_optimization_performed']
 launches=read(out/'launches.json');assert launches and all(l['cuda_device_count']==1 and l['gpu_uuid'].startswith('GPU-') for l in launches)
 for p in (out/'tuning').glob('*/fold*/trial*/inner*/complete.json'):assert read(p)['gpu_uuid']==manifest['selected_gpu_uuid']
 result=dict(passed=True,models=1,full_training_rows=len(y),internal_trainings=90,configuration_trials=30,full_data_normalization=True,internal_validation_only_selection=True,epochs_selected_before_refit=True,reference_prediction_reproduced=True,training_mae=record['training_mae'],independent_test_mae=None,protected_old_files_unchanged=True,one_gpu=True)
 dump(out/'audit.json',result);print(json.dumps(result,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='final_mlp256/results/full_20261030');p.add_argument('--audit',action='store_true');a=p.parse_args();out=ROOT/a.out
 for name in ['failure.json','complete.json','status.json']:
  if (out/name).exists():print((out/name).read_text());break
 else:print(json.dumps(dict(state='not_started')))
 if a.audit:audit(out)
if __name__=='__main__':main()
