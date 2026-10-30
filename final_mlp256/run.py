import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,fcntl,json,subprocess,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
from importlib.metadata import version
from bayes_experiment.common import ROOT,SEED,TRAIN,sha,dump,read,data,inner_splits,validate_files
from bayes_accel.tuning import tune
from bayes_accel.training import fit,save_atomic,predict_rows,TrainingPaused
from bayes_accel.workers import close_pool
from neural_baseline.models import load
CONFIG=dict(hidden=[256,128],activation='relu',dropout=0.,weight_decay=.001)

class NoIdleGPU(Exception):pass

def idle_gpu_rows(text):
 rows=[]
 for line in text.strip().splitlines():
  uuid,memory,util=[x.strip() for x in line.split(',')]
  if uuid.startswith('GPU-') and int(memory)<=64 and int(util)==0:rows.append(uuid)
 return rows

def choose_gpu():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 rows=idle_gpu_rows(text)
 if not rows:raise NoIdleGPU('No idle GPU; paused without starting training')
 return rows[0]

def definition(n):return dict(fold=1,train=list(range(n)),test=[],init_seed=SEED,shuffle_seed=SEED+100000,search_seed=SEED+200000)

def source_hashes():
 paths=list((ROOT/'final_mlp256').glob('*.py'))+[ROOT/'final_mlp256/launch.sh']+list((ROOT/'bayes_accel').glob('*.py'))+list((ROOT/'bayes_experiment').glob('*.py'))+list((ROOT/'cv_experiment').glob('*.py'))+[ROOT/p for p in ['neural_baseline/models.py','earlystop_experiment/training.py','earlystop_experiment/common.py','Lab6/baseline.py']]
 return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def expected(ids):return dict(version=1,seed=SEED,config=CONFIG,training=TRAIN,data_sha256=sha(ROOT/'Lab6/dataset_same8/train.csv'),candidate_ids=ids,source_sha256=source_hashes(),versions={n:version(n) for n in ['torch','numpy','pandas','scikit-learn','scipy','optuna']},purpose='full-data final fit after three-fold internal TPE; no independent final test')

def export_final(out,model,ids,X,y,selected):
 predictions=predict_rows(model,X).astype(np.float64);errors=np.abs(predictions-y)
 pd.DataFrame(dict(row_index=np.arange(len(y)),candidate_id=ids,y_true=y,y_pred=predictions)).to_csv(out/'training_predictions.csv',index=False)
 stats={k:getattr(model,k).detach().numpy().tolist() for k in ['x_mean','x_scale','y_mean','y_scale']};dump(out/'normalization.json',stats)
 result=dict(state='complete',model='MLP 256->128',n_training_rows=len(y),training_mae=float(errors.mean()),independent_test_mae=None,selected_inner_validation_mae=selected['inner_validation_mae'],selected_config=dict(CONFIG,**selected['parameters']),epochs=selected['refit_epochs'],init_seed=SEED,shuffle_seed=SEED+100000,reference_cv_mae=4.561498209482524,reference_cv_is_this_models_test_score=False,hyperparameter_trials=30,internal_trainings=90,ramping_optimization_performed=False,files={p:sha(out/p) for p in ['model.pt','history.json','fit_metadata.json','selection.json','training_predictions.csv','normalization.json']})
 dump(out/'complete.json',result);dump(out/'status.json',dict(state='complete',training_mae=result['training_mae'],epochs=result['epochs']))
 (out/'README.md').write_text('# 全数据最终256→128模型\n\n最终模型使用全部2048条数据。30组TPE配置、每组内部三折，仅用内部验证选择参数和训练轮数，随后固定配置重训全数据；没有独立测试集。\n\n训练MAE：'+str(result['training_mae'])+'。内部选择验证MAE：'+str(result['selected_inner_validation_mae'])+'，这是用于选配置的验证值，不是独立测试成绩。最终轮数：'+str(result['epochs'])+'。\n\n历史十折OOF MAE 4.561498仅作为先前方法的参考，不冒充本模型的测试MAE。冻结的原十折包不修改；本轮不寻优ramping、不使用老师测试机会。\n')
 return result

def run(args):
 out=(ROOT/args.out).resolve()
 if not out.is_relative_to(ROOT):raise ValueError('Output must stay inside project')
 out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);args.owns_output=True
 ids,X,y=data();ex=expected(ids);manifest=read(out/'manifest.json') if (out/'manifest.json').exists() else None
 if manifest:
  if not args.resume:raise ValueError('Use --resume for existing output')
  for k,v in ex.items():
   if manifest.get(k)!=v:raise ValueError('Resume mismatch: '+k)
  for path,digest in manifest['protected_files'].items():
   if sha(ROOT/path)!=digest:raise ValueError('Protected file changed: '+path)
  if (out/'complete.json').exists():print('ALREADY COMPLETE',flush=True);return
 elif args.resume:raise ValueError('No manifest to resume')
 uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if not uuid.startswith('GPU-') or ',' in uuid:raise ValueError('Only one full GPU UUID allowed')
 locks=ROOT.parent/'.locks';locks.mkdir(exist_ok=True);gpu_lock=(locks/('cv-'+uuid+'.lock')).open('w');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 snapshot=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 if uuid not in idle_gpu_rows(snapshot):raise NoIdleGPU('Chosen GPU is no longer idle; paused before training')
 if torch.cuda.device_count()!=1:raise ValueError('Exactly one CUDA device required')
 torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if not manifest:
  tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split(chr(0));protected={p:sha(ROOT/p) for p in tracked if p and not p.startswith('final_mlp256/') and not p.endswith('.md') and (ROOT/p).is_file()}
  manifest=dict(**ex,protected_files=protected,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),selected_gpu_uuid=uuid);dump(out/'manifest.json',manifest)
 for name in ['failure.json','pause.request']:
  if (out/name).exists():(out/name).unlink()
 started=time.monotonic()
 def status(trial,inner,stage,epoch):dump(out/'status.json',dict(state='running',stage=stage,trial=trial+1,inner_parallelism=3,epoch=epoch,gpu_uuid=uuid,elapsed_seconds_current_launch=time.monotonic()-started))
 launches=read(out/'launches.json') if (out/'launches.json').exists() else [];launches.append(dict(gpu_uuid=uuid,cuda_device_count=1,initial_memory_utilization_snapshot=snapshot.strip(),mps_pipe=os.environ.get('CUDA_MPS_PIPE_DIRECTORY'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())));dump(out/'launches.json',launches)
 fold=definition(len(y));dump(out/'internal_splits.json',inner_splits(fold));print('START full-data final MLP; one GPU '+uuid,flush=True)
 selected=tune(out/'tuning'/'relu_256_128'/'fold01',CONFIG,fold,X,y,'cuda',status);dump(out/'selection.json',selected)
 status(29,0,'full_data_refit',None)
 def progress(epoch,history):dump(out/'history.json',history);status(29,0,'full_data_refit',epoch)
 config=dict(CONFIG,**selected['parameters']);model,history,meta=fit(X,y,config,SEED,SEED+100000,'cuda',epochs=selected['refit_epochs'],callback=progress,resume_path=out/'resume.pt',pause_path=out/'pause.request')
 save_atomic(model,out/'model.pt');loaded=load(out/'model.pt','cuda')
 with torch.no_grad():torch.testing.assert_close(model(torch.tensor(X[:32],dtype=torch.float32,device='cuda')),loaded(torch.tensor(X[:32],dtype=torch.float32,device='cuda')),atol=0,rtol=0)
 del model,loaded;dump(out/'history.json',history);dump(out/'fit_metadata.json',meta);result=export_final(out,load(out/'model.pt'),ids,X,y,selected)
 for path,digest in manifest['protected_files'].items():assert sha(ROOT/path)==digest,path
 print('COMPLETE '+json.dumps(result),flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',default='final_mlp256/results/full_20261030');parser.add_argument('--resume',action='store_true');parser.add_argument('--choose-gpu',action='store_true');args=parser.parse_args();args.owns_output=False
 if args.choose_gpu:
  try:print(choose_gpu())
  except NoIdleGPU as e:print(str(e),file=__import__('sys').stderr);raise SystemExit(3)
  return
 try:run(args)
 except (NoIdleGPU,TrainingPaused) as e:
  if args.owns_output:dump((ROOT/args.out)/'status.json',dict(state='paused',reason=str(e)))
  print('PAUSED '+str(e),flush=True)
 except Exception:
  if args.owns_output and not ((ROOT/args.out)/'complete.json').exists():dump((ROOT/args.out)/'failure.json',dict(state='failed',traceback=traceback.format_exc()))
  raise
 finally:close_pool()
if __name__=='__main__':main()
