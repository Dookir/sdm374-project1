import math
import numpy as np,pandas as pd,torch,optuna
from bayes_experiment.common import inner_splits,read,sha,validate_files,TRAIN

def audit_tuning(folder,outer,base_config,X,y):
 splits=inner_splits(outer);assert read(folder/'splits.json')==splits
 outertrain=set(outer['train']);test=set(outer['test']);seen=[]
 for split in splits:
  a=set(split['train']);b=set(split['validation']);assert not a&b and a|b==outertrain and not (a|b)&test;seen.extend(b)
 assert sorted(seen)==sorted(outertrain)
 study=optuna.load_study(study_name='tuning',storage='sqlite:///'+str((folder/'study.db').resolve()))
 records=[]
 for slot in range(TRAIN['trials']):
  directory=folder/f'trial{slot:02d}';record=read(directory/'complete.json');validate_files(directory,record);params=record['parameters'];assignment=read(directory/'parameters.json')
  assert assignment['slot']==slot and assignment['parameters']==params and assignment['trial_number']==record['trial_number']
  assert .0001<=params['learning_rate']<=.003 and .0001<=params['weight_decay']<=.03 and params['dropout'] in [0.,.05,.1,.2]
  if slot==0:assert params==dict(learning_rate=.001,weight_decay=.001,dropout=0.)
  total=0.;n=0;epochs=[]
  for split in splits:
   part=directory/f"inner{split['inner_fold']:02d}";info=read(part/'complete.json');validate_files(part,info);a=np.asarray(split['train']);b=np.asarray(split['validation'])
   assert info['config']==dict(base_config,**params) and info['init_seed']==split['init_seed'] and info['shuffle_seed']==split['shuffle_seed']
   assert info['actual_optimizer']=={k:params[k] for k in ['learning_rate','weight_decay']} and info['dropout']==params['dropout'] and info['scheduler_t_max']==600
   tx=torch.tensor(X[a],dtype=torch.float64);ty=torch.tensor(y[a],dtype=torch.float64)
   expected=dict(x_mean=tx.mean(0),x_scale=tx.std(0,unbiased=False).clamp_min(1e-6),y_mean=ty.mean(),y_scale=ty.std(unbiased=False).clamp_min(1e-6));stats=read(part/'normalization.json')
   for key,value in expected.items():np.testing.assert_array_equal(np.asarray(stats[key]),value.float().numpy())
   history=read(part/'history.json');assert len(history)==info['epochs_run']<=600
   best=float('inf');chosen=0;bad=0
   for k,h in enumerate(history,1):
    assert h['epoch']==k and math.isfinite(h['validation_mae'])
    expected_lr=params['learning_rate']*(.01+.99*(1+math.cos(math.pi*k/600))/2);assert abs(h['learning_rate']-expected_lr)<1e-12
    if h['validation_mae']<best:best=h['validation_mae'];chosen=k;bad=0
    else:bad+=1
    if bad>=30:assert k==len(history)
   assert len(history)==600 or bad>=30
   assert info['selected_epoch']==chosen and info['validation_mae_best']==best
   frame=pd.read_csv(part/'validation_predictions.csv',float_precision='round_trip');np.testing.assert_array_equal(frame.row_index,b);np.testing.assert_allclose(frame.y_true,y[b],atol=1e-12,rtol=0)
   err=np.abs(frame.y_pred.to_numpy()-y[b]);assert np.isfinite(frame.y_pred).all() and abs(float(err.sum())-info['validation_abs_error_sum'])<1e-7
   assert info['n_validation']==len(b) and abs(float(err.mean())-best)<1e-3
   total+=float(err.sum());n+=len(b);epochs.append(chosen)
  assert abs(total/n-record['objective'])<1e-9 and record['selected_epochs']==epochs
  trial=study.trials[record['trial_number']];assert trial.state==optuna.trial.TrialState.COMPLETE and trial.params==params and abs(trial.value-record['objective'])<1e-12
  records.append(record)
 best=min(records,key=lambda r:(r['objective'],r['slot']));selection=read(folder/'complete.json');validate_files(folder,selection)
 assert read(folder/'trials.json')==records
 assert selection['best_trial']==best['trial_number'] and selection['best_slot']==best['slot'] and selection['parameters']==best['parameters'] and selection['inner_validation_mae']==best['objective']
 assert selection['selected_epochs']==best['selected_epochs'] and selection['refit_epochs']==int(np.median(best['selected_epochs'])) and selection['trials']==TRAIN['trials']
 return selection
