# One snapshot by default; a complete CPU audit only with --audit.
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parents[1]
def audit(out):
 import numpy as np,pandas as pd,torch
 from neural_baseline.models import load
 from cv_experiment.common import SEED,CONFIGS,TRAIN,SEARCH,sha,read,dump,data,make_splits,validate_splits,validate_files,summarize
 from cv_experiment.training import predict_rows
 torch.set_num_threads(4);ids,X,y=data();manifest=read(out/'manifest.json');splits=read(out/'splits.json');complete=read(out/'complete.json');records=read(out/'fold_metrics.json')
 assert manifest['n_rows']==len(y)==2048 and manifest['seed']==SEED
 assert manifest['training']==TRAIN and manifest['architectures']==CONFIGS and manifest['search']==SEARCH
 assert manifest['candidate_ids']==ids and manifest['data_sha256']==sha(ROOT/'Lab6/dataset_same8/train.csv')
 assert splits==make_splits(len(y)) and sha(out/'splits.json')==manifest['splits_sha256'];validate_splits(splits,len(y))
 for path,digest in (manifest['protected_files']|manifest['source_sha256']).items():assert sha(ROOT/path)==digest,path
 expected={(name,fold['fold']) for name in CONFIGS for fold in splits};assert len(records)==30 and {(r['model'],r['fold']) for r in records}==expected
 coverage={name:np.zeros(len(y),dtype=int) for name in CONFIGS}
 for row in records:
  name=row['model'];fold=splits[row['fold']-1];folder=out/'models'/name/f"fold{row['fold']:02d}";training=read(folder/'training_complete.json');search=read(folder/'search_complete.json');validate_files(folder,training);validate_files(folder,search)
  assert training['epochs']==TRAIN['epochs'] and training['init_seed']==fold['init_seed'] and training['shuffle_seed']==fold['shuffle_seed']
  assert search['search_seed']==fold['search_seed']
  model=load(folder/'model.pt');assert model.config==CONFIGS[name]
  tx=torch.tensor(X[fold['train']],dtype=torch.float64);ty=torch.tensor(y[fold['train']],dtype=torch.float64)
  stats=dict(x_mean=tx.mean(0),x_scale=tx.std(0,unbiased=False).clamp_min(1e-6),y_mean=ty.mean(),y_scale=ty.std(unbiased=False).clamp_min(1e-6))
  for key,value in stats.items():torch.testing.assert_close(getattr(model,key),value.float(),atol=0,rtol=0)
  for split in ['train','test']:
   idx=np.asarray(fold[split]);frame=pd.read_csv(folder/(split+'_predictions.csv'),dtype={'candidate_id':str},float_precision='round_trip')
   np.testing.assert_array_equal(frame.row_index,idx);np.testing.assert_array_equal(frame.candidate_id,np.asarray(ids)[idx]);np.testing.assert_allclose(frame.y_true,y[idx],atol=1e-12,rtol=0)
   predicted=predict_rows(model,X[idx]);np.testing.assert_allclose(frame.y_pred,predicted,atol=1e-4,rtol=0)
   errors=np.abs(frame.y_pred.to_numpy()-y[idx]);assert len(idx)==row['n_'+split]
   assert abs(float(errors.mean())-row[split+'_mae'])<1e-9
   assert abs(float(errors.sum())-row[split+'_abs_error_sum'])<1e-7
   assert abs(training[split+'_mae']-row[split+'_mae'])<1e-12
   if split=='test':coverage[name][idx]+=1
  details=read(folder/'search_details.json');anchor=np.asarray(details['reference_weights']).ravel();assert np.any(np.all(X[fold['train']]==anchor,axis=1))
  before=float(predict_rows(model,anchor[None,:])[0]);assert abs(before-row['search_before'])<1e-5
  weights=np.asarray(read(folder/'weights.json'));assert weights.shape==(3,7) and np.isfinite(weights).all() and (weights>=0).all() and np.allclose(weights.sum(1),1,atol=1e-12,rtol=0)
  after=float(predict_rows(model,weights.reshape(1,21))[0]);assert abs(after-row['search_after'])<1e-5
  assert row['search_after']<=row['search_before']+1e-6
  assert search['after']==row['search_after'] and search['before']==row['search_before']
 for counts in coverage.values():np.testing.assert_array_equal(counts,np.ones(len(y),dtype=int))
 summary=summarize(records);assert read(out/'summary.json')==summary==complete['summary']
 oof=pd.read_csv(out/'oof_predictions.csv',dtype={'candidate_id':str},float_precision='round_trip');assert len(oof)==6144
 for name in CONFIGS:
  group=oof[oof.model==name].sort_values('row_index');assert len(group)==2048
  np.testing.assert_array_equal(group.row_index,np.arange(len(y)));np.testing.assert_array_equal(group.candidate_id,ids);np.testing.assert_allclose(group.y_true,y,atol=1e-12,rtol=0)
  for fold in splits:
   part=group[group.fold==fold['fold']];np.testing.assert_array_equal(part.row_index,fold['test'])
   stored=pd.read_csv(out/'models'/name/f"fold{fold['fold']:02d}"/'test_predictions.csv',float_precision='round_trip');np.testing.assert_allclose(part.y_pred,stored.y_pred,atol=0,rtol=0)
  s=next(r for r in summary if r['model']==name);assert s['folds']==10 and s['n_train_predictions']==18432 and s['n_test_predictions']==2048
  assert abs(float(np.abs(group.y_pred.to_numpy()-y).mean())-s['test_mae_oof'])<1e-9
  best=min((r for r in records if r['model']==name),key=lambda r:r['search_after']);folder=out/'best'/name
  assert sha(folder/'model.pt')==sha(ROOT/best['checkpoint']) and sha(folder/'weights.json')==sha(ROOT/best['weights'])
  assert read(folder/'selection.json')['fold']==best['fold']
 launches=read(out/'launches.json');assert launches and all(r['cuda_device_count']==1 and r['gpu_uuid'].startswith('GPU-') for r in launches)
 result=dict(passed=True,models_verified=30,folds=10,test_coverage_per_model=2048,disjoint_partitions=True,preprocessing_training_rows_only=True,train_and_test_mae_recomputed=True,oof_summary_recomputed=True,predictions_reproduced=True,all_weights_legal=True,search_not_worse=True,originals_and_code_unchanged=True)
 dump(out/'audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--audit',action='store_true');a=p.parse_args();out=(ROOT/a.out).resolve()
 for name in ['failure.json','complete.json','status.json']:
  if (out/name).exists():print((out/name).read_text());break
 else:print(json.dumps(dict(state='not_started',out=str(out))))
 if a.audit:audit(out)
if __name__=='__main__':main()
