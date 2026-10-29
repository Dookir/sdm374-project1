import gc,pickle
import numpy as np,pandas as pd,torch,optuna
from optuna.distributions import FloatDistribution,CategoricalDistribution
from optuna.trial import TrialState
from bayes_experiment.common import TRAIN,SEED,inner_splits,read,dump,sha,validate_files
from bayes_accel.training import fit,predict_rows
from bayes_accel.workers import get_pool,train_inner

def space():return dict(learning_rate=FloatDistribution(.0001,.003,log=True),weight_decay=FloatDistribution(.0001,.03,log=True),dropout=CategoricalDistribution([0.,.05,.1,.2]))
def tune(folder,config,fold,X,y,device,status):
 folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():
  result=read(folder/'complete.json');validate_files(folder,result);return result
 splits=inner_splits(fold);dump(folder/'splits.json',splits)
 storage='sqlite:///'+str((folder/'study.db').resolve())
 # One seeded sampler per trial makes interrupted ask/recovery independent of opaque RNG state.
 seed=SEED+fold['fold']*1000
 study=optuna.create_study(storage=storage,study_name='tuning',direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=10,multivariate=True),load_if_exists=True)
 if not study.trials:study.enqueue_trial(dict(learning_rate=.001,weight_decay=.001,dropout=0.))
 for slot in range(TRAIN['trials']):
  marker=folder/f'trial{slot:02d}'/'complete.json';trialdir=marker.parent;trialdir.mkdir(exist_ok=True)
  plan=trialdir/'parameters.json'
  if marker.exists():
   record=read(marker);validate_files(trialdir,record)
   trial=study.trials[record['trial_number']]
   if trial.state==TrialState.RUNNING:study.tell(trial.number,record['objective'])
   elif trial.state!=TrialState.COMPLETE:raise ValueError('Trial state incompatible with completed artifacts')
   continue
  if plan.exists():
   assignment=read(plan);frozen=study.trials[assignment['trial_number']]
   if frozen.params!=assignment['parameters'] or frozen.state not in [TrialState.RUNNING,TrialState.COMPLETE]:raise ValueError('Pending trial changed')
   number=frozen.number;params=assignment['parameters']
  else:
   # Recover an ask interrupted before its assignment file was committed.
   pending=study.get_trials(deepcopy=False,states=(TrialState.RUNNING,))
   reusable=[t for t in pending if set(t.params)==set(space())]
   for old in pending:
    if old not in reusable:study.tell(old.number,state=TrialState.FAIL)
   if len(reusable)>1:raise ValueError('Multiple pending parameter assignments')
   study.sampler=optuna.samplers.TPESampler(seed=seed+slot,n_startup_trials=10,multivariate=True)
   if reusable:number=reusable[0].number;params=reusable[0].params
   else:
    if slot==0 and not study.get_trials(deepcopy=False,states=(TrialState.WAITING,)):study.enqueue_trial(dict(learning_rate=.001,weight_decay=.001,dropout=0.))
    trial=study.ask(fixed_distributions=space());number=trial.number;params=trial.params
   dump(plan,dict(slot=slot,trial_number=number,parameters=params,sampler_seed=seed+slot))
   tmp=folder/'sampler.pkl.tmp';tmp.write_bytes(pickle.dumps(study.sampler));tmp.replace(folder/'sampler.pkl')
  candidate=dict(config,**params);parts={};futures={}
  out=folder.parents[2]
  status(slot,0,'tuning_parallel_3',None)
  for inner in splits:
   part=trialdir/f"inner{inner['inner_fold']:02d}";part.mkdir(exist_ok=True)
   if (part/'complete.json').exists():
    info=read(part/'complete.json');validate_files(part,info);parts[inner['inner_fold']]=info;continue
   if (out/'pause.request').exists():
    from bayes_accel.training import TrainingPaused
    raise TrainingPaused('Pause requested before next inner fit')
   job=dict(part=str(part),inner=inner,X=X,y=y,candidate=candidate,pause_path=str(out/'pause.request'))
   futures[get_pool().submit(train_inner,job)]=inner['inner_fold']
  from concurrent.futures import as_completed
  for future in as_completed(futures):parts[futures[future]]=future.result()
  parts=[parts[k] for k in range(1,4)]
  objective=sum(p['validation_abs_error_sum'] for p in parts)/sum(p['n_validation'] for p in parts)
  record=dict(slot=slot,trial_number=number,parameters=params,objective=objective,selected_epochs=[p['selected_epoch'] for p in parts],files={'parameters.json':sha(plan),**{f'inner{k:02d}/complete.json':sha(trialdir/f'inner{k:02d}'/'complete.json') for k in range(1,4)}})
  dump(marker,record)
  if study.trials[number].state==TrialState.RUNNING:study.tell(number,objective)
  print(f'TUNING {folder.parent.name}/{folder.name} trial={slot+1}/30 MAE={objective:.6f}',flush=True)
 records=[read(folder/f'trial{k:02d}'/'complete.json') for k in range(TRAIN['trials'])]
 best=min(records,key=lambda r:(r['objective'],r['slot']))
 result=dict(best_trial=best['trial_number'],best_slot=best['slot'],parameters=best['parameters'],inner_validation_mae=best['objective'],selected_epochs=best['selected_epochs'],refit_epochs=int(np.median(best['selected_epochs'])),trials=TRAIN['trials'],files={'splits.json':sha(folder/'splits.json'),**{f'trial{k:02d}/complete.json':sha(folder/f'trial{k:02d}'/'complete.json') for k in range(TRAIN['trials'])}})
 dump(folder/'trials.json',records);dump(folder/'complete.json',result);return result
