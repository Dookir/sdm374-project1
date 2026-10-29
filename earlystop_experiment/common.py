# Original hyperparameters plus nested validation for early stopping only.
from cv_experiment.common import ROOT,SEED,CONFIGS,SEARCH,sha,dump,read,data,validate_files
from cv_experiment.common import TRAIN as BASE_TRAIN,make_splits as outer_splits,summarize as base_summary
from sklearn.model_selection import train_test_split
import numpy as np
TRAIN=dict(BASE_TRAIN,validation_fraction=.1,patience=30,min_delta=0.,validation_metric='MAE in original ramping units')
BASELINE=ROOT/'cv_experiment/results/cv10_20261029'
def make_splits(n,seed=SEED):
 result=[]
 for outer in outer_splits(n,seed):
  split_seed=seed+300000+outer['fold']-1
  fit,validation=train_test_split(outer['train'],test_size=TRAIN['validation_fraction'],shuffle=True,random_state=split_seed)
  result.append(dict(**{k:v for k,v in outer.items() if k!='train'},outer_train=outer['train'],train=sorted(fit),validation=sorted(validation),inner_split_seed=split_seed))
 return result
def validate_splits(splits,n):
 seen=np.zeros(n,dtype=int)
 for fold in splits:
  fit,val,test=[np.asarray(fold[k]) for k in ['train','validation','test']]
  assert len(np.unique(fit))==len(fit) and len(np.unique(val))==len(val) and len(np.unique(test))==len(test)
  assert not np.intersect1d(fit,val).size and not np.intersect1d(fit,test).size and not np.intersect1d(val,test).size
  np.testing.assert_array_equal(np.sort(np.r_[fit,val]),fold['outer_train']);np.testing.assert_array_equal(np.sort(np.r_[fit,val,test]),np.arange(n));seen[test]+=1
 np.testing.assert_array_equal(seen,np.ones(n,dtype=int))
def summarize(records):
 rows=base_summary(records)
 baseline={r['model']:r for r in read(BASELINE/'summary.json')}
 for row in rows:
  group=[r for r in records if r['model']==row['model']]
  row.update(validation_mae=sum(r['validation_abs_error_sum'] for r in group)/sum(r['n_validation'] for r in group),n_validation_predictions=sum(r['n_validation'] for r in group),selected_epoch_mean=float(np.mean([r['selected_epoch'] for r in group])),epochs_run_mean=float(np.mean([r['epochs_run'] for r in group])),stopped_early_folds=sum(r['epochs_run']<TRAIN['epochs'] for r in group),baseline_test_mae=baseline[row['model']]['test_mae_oof'],test_mae_improvement=baseline[row['model']]['test_mae_oof']-row['test_mae_oof'])
 return rows
