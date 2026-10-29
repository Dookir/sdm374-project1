import os,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd,torch
from bayes_experiment.common import dump,sha
from bayes_accel.training import fit,predict_rows
_POOL=None

def initialize():
 if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:raise ValueError('One GPU UUID only')
 if torch.cuda.device_count()!=1:raise ValueError('Worker sees multiple GPUs')
 torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def get_pool():
 global _POOL
 if _POOL is None:_POOL=ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),initializer=initialize)
 return _POOL

def close_pool():
 global _POOL
 if _POOL is not None:_POOL.shutdown(wait=True,cancel_futures=True);_POOL=None

def train_inner(job):
 part=Path(job['part']);part.mkdir(parents=True,exist_ok=True);inner=job['inner'];X=job['X'];y=job['y'];candidate=job['candidate'];a=np.asarray(inner['train']);b=np.asarray(inner['validation'])
 def progress(epoch,history):
  dump(part/'progress.json',dict(epoch=epoch,state='running',pid=os.getpid(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']))
  if epoch%50==0:dump(part/'history.json',history)
 model,history,meta=fit(X[a],y[a],candidate,inner['init_seed'],inner['shuffle_seed'],'cuda',epochs=job.get('epochs',600),patience=job.get('patience',30),validation=(X[b],y[b]),callback=progress,resume_path=part/'resume.pt',pause_path=job.get('pause_path'))
 model=model.cpu();prediction=predict_rows(model,X[b]).astype(np.float64);errors=np.abs(prediction-y[b])
 pd.DataFrame(dict(row_index=b,y_true=y[b],y_pred=prediction)).to_csv(part/'validation_predictions.csv',index=False)
 stats={key:getattr(model,key).detach().numpy().tolist() for key in ['x_mean','x_scale','y_mean','y_scale']};dump(part/'normalization.json',stats);dump(part/'history.json',history)
 if job.get('save_model'):
  from neural_baseline.models import save
  save(model,part/'benchmark_model.pt')
 info=dict(**meta,inner_fold=inner['inner_fold'],n_validation=len(b),validation_mae_cpu=float(errors.mean()),validation_abs_error_sum=float(errors.sum()),config=candidate,init_seed=inner['init_seed'],shuffle_seed=inner['shuffle_seed'],worker_pid=os.getpid(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],files={p:sha(part/p) for p in ['validation_predictions.csv','normalization.json','history.json']})
 if abs(info['validation_mae_cpu']-meta['validation_mae_best'])>1e-3:raise ValueError('Best checkpoint validation mismatch')
 dump(part/'complete.json',info);dump(part/'progress.json',dict(epoch=meta['epochs_run'],state='complete',pid=os.getpid(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']))
 # Keep the CUDA allocator warm across independent inner fits.
 return info
