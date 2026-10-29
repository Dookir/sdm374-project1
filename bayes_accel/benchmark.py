import os,time,tempfile,argparse
from pathlib import Path
import numpy as np,torch
from bayes_experiment.common import ROOT,CONFIGS,data,make_splits,inner_splits,dump
from bayes_experiment.training import fit as original_fit
from bayes_accel.training import fit as fast_fit
from bayes_accel.workers import initialize,get_pool,train_inner,close_pool

def timed_serial(fit,X,y,config,parts,epochs):
 saved=[];torch.cuda.synchronize();start=time.perf_counter()
 for inner in parts:
  a=np.asarray(inner['train']);b=np.asarray(inner['validation'])
  m,h,meta=fit(X[a],y[a],config,inner['init_seed'],inner['shuffle_seed'],'cuda',epochs=epochs,validation=(X[b],y[b]),patience=600)
  saved.append((h,{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},meta));del m
 torch.cuda.synchronize();return time.perf_counter()-start,saved

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=80);a=p.parse_args();initialize();_,X,y=data();parts=inner_splits(make_splits(len(y))[5]);report=[]
 with tempfile.TemporaryDirectory(prefix='bayes-accel-benchmark-',dir=str(ROOT.parent)) as directory:
  folder=Path(directory)
  # Start all three processes and warm their CUDA contexts before timing.
  warmconfig=dict(CONFIGS['relu_64_32'],learning_rate=.002,dropout=.2,weight_decay=.01)
  futures=[get_pool().submit(train_inner,dict(part=str(folder/f'warm{k}'),inner=s,X=X,y=y,candidate=warmconfig,epochs=2,patience=600)) for k,s in enumerate(parts)]
  for f in futures:f.result()
  for name,base in CONFIGS.items():
   config=dict(base,learning_rate=.002,dropout=.2,weight_decay=.01)
   # Warm the parent process too.
   timed_serial(original_fit,X,y,config,parts[:1],2)
   seconds_old,old=timed_serial(original_fit,X,y,config,parts,a.epochs)
   seconds_fast,fast=timed_serial(fast_fit,X,y,config,parts,a.epochs)
   for i in range(3):
    assert [h['validation_mae'] for h in old[i][0]]==[h['validation_mae'] for h in fast[i][0]],'Validation history changed'
    for key in old[i][1]:torch.testing.assert_close(old[i][1][key],fast[i][1][key],atol=0,rtol=0)
    assert old[i][2]['selected_epoch']==fast[i][2]['selected_epoch']
   begun=time.perf_counter();jobs=[get_pool().submit(train_inner,dict(part=str(folder/f'{name}-inner{k}'),inner=s,X=X,y=y,candidate=config,epochs=a.epochs,patience=600,save_model=True)) for k,s in enumerate(parts)]
   infos=[f.result() for f in jobs];seconds_parallel=time.perf_counter()-begun
   import json
   for k,info in enumerate(infos):
    history=json.loads((folder/f'{name}-inner{k}/history.json').read_text());assert [h['validation_mae'] for h in history]==[h['validation_mae'] for h in old[k][0]],'Parallel RNG or calculations changed'
    from neural_baseline.models import load
    parallel_model=load(folder/f'{name}-inner{k}/benchmark_model.pt')
    for key in old[k][1]:torch.testing.assert_close(old[k][1][key],parallel_model.state_dict()[key],atol=0,rtol=0)
    assert info['selected_epoch']==old[k][2]['selected_epoch'] and info['validation_mae_best']==old[k][2]['validation_mae_best']
   if name=='relu_64_32':
    from bayes_accel.training import TrainingPaused
    checkpoint=folder/'resume.pt';pause=folder/'pause.request';pause.touch();inner=parts[0];ia=np.asarray(inner['train']);ib=np.asarray(inner['validation'])
    args=(X[ia],y[ia],config,inner['init_seed'],inner['shuffle_seed'],'cuda');kw=dict(epochs=a.epochs,validation=(X[ib],y[ib]),patience=600,resume_path=checkpoint,pause_path=pause)
    try:fast_fit(*args,**kw);raise AssertionError('Expected requested pause')
    except TrainingPaused:pass
    pause.unlink();resumed,rh,_=fast_fit(*args,**kw)
    assert [h['validation_mae'] for h in rh]==[h['validation_mae'] for h in old[0][0]]
    for key in old[0][1]:torch.testing.assert_close(old[0][1][key],resumed.state_dict()[key].cpu(),atol=0,rtol=0)
   row=dict(model=name,epochs_per_inner=a.epochs,original_serial_seconds=seconds_old,sync_reduced_serial_seconds=seconds_fast,three_inner_parallel_seconds=seconds_parallel,speedup=seconds_old/seconds_parallel,validation_histories_bitwise_equal=True,serial_model_states_bitwise_equal=True,parallel_model_states_bitwise_equal=True,selected_epochs_equal=True,worker_pids=[i['worker_pid'] for i in infos],gpu_uuids=list(set(i['gpu_uuid'] for i in infos)))
   report.append(row);print(row,flush=True)
 close_pool();dump(Path(a.out),dict(passed=True,gpu_epoch_resume_verified=True,gpu_epoch_resume_tested_model='relu_64_32',gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],mps_pipe=os.environ.get('CUDA_MPS_PIPE_DIRECTORY'),benchmarks=report,total_speedup=sum(x['original_serial_seconds'] for x in report)/sum(x['three_inner_parallel_seconds'] for x in report)))
if __name__=='__main__':main()
