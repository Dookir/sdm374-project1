import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,fcntl,json,subprocess,sys,time,traceback
import numpy as np,pandas as pd,torch
from neural_baseline.models import load
from cv_experiment.common import ROOT,SEED,CONFIGS,TRAIN,SEARCH,sha,dump,read,data,make_splits,validate_splits,validate_files,summarize
from cv_experiment.training import train_fold,save_atomic,predict_rows
from cv_experiment.optimize import refine

def dependency_hashes():
 paths=list((ROOT/'cv_experiment').glob('*.py'))+[ROOT/p for p in ['neural_baseline/models.py','Lab6/baseline.py','weight_search/run.py','tabular_next/run.py','tabular_next/models.py']]
 return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def verify_manifest(out,expected,resume):
 path=out/'manifest.json'
 if path.exists():
  if not resume:raise ValueError('Output exists; use --resume or a fresh --out')
  manifest=read(path)
  for key,value in expected.items():
   if manifest.get(key)!=value:raise ValueError('Resume mismatch: '+key)
  for path,digest in manifest['protected_files'].items():
   if not (ROOT/path).is_file() or sha(ROOT/path)!=digest:raise ValueError('Protected file changed: '+path)
  if sha(out/'splits.json')!=manifest['splits_sha256']:raise ValueError('Splits file changed')
  return manifest
 if resume:raise ValueError('--resume requires an existing manifest')
 return None

def load_completed_training(folder):
 record=read(folder/'training_complete.json');validate_files(folder,record);return record

def export_predictions(folder,model,ids,X,y,fold):
 metrics={}
 for split in ['train','test']:
  idx=np.asarray(fold[split]);pred=predict_rows(model,X[idx]).astype(np.float64);actual=y[idx];absolute=np.abs(pred-actual)
  frame=pd.DataFrame(dict(row_index=idx,candidate_id=np.asarray(ids)[idx],y_true=actual,y_pred=pred))
  frame.to_csv(folder/(split+'_predictions.csv'),index=False)
  metrics['n_'+split]=len(idx);metrics[split+'_mae']=float(absolute.mean());metrics[split+'_abs_error_sum']=float(absolute.sum())
 return metrics

def finalize(out,records,manifest):
 summary=summarize(records);oof=[]
 for record in records:
  folder=out/'models'/record['model']/f"fold{record['fold']:02d}"
  frame=pd.read_csv(folder/'test_predictions.csv',dtype={'candidate_id':str},float_precision='round_trip');frame.insert(0,'fold',record['fold']);frame.insert(0,'model',record['model']);oof.append(frame)
 merged=pd.concat(oof,ignore_index=True).sort_values(['model','row_index']);merged.to_csv(out/'oof_predictions.csv',index=False)
 dump(out/'summary.json',summary);pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
 import shutil
 for name in CONFIGS:
  best=min((r for r in records if r['model']==name),key=lambda r:r['search_after']);folder=out/'models'/name/f"fold{best['fold']:02d}";target=out/'best'/name;target.mkdir(parents=True,exist_ok=True)
  shutil.copy2(folder/'model.pt',target/'model.pt');shutil.copy2(folder/'weights.json',target/'weights.json');dump(target/'selection.json',dict(**best,selection='lowest own predicted_ramping among ten fixed-epoch fold models'))
 text='# 10折交叉验证与权重寻优结果\n\n三种网络、统一600轮、每条样本一次留出测试。训练MAE按训练样本数加权；测试MAE为合并OOF预测MAE；标准差为10折测试MAE样本标准差。未使用测试MAE选检查点、超参数或权重。\n\n|模型|训练MAE|测试MAE（OOF）|各折测试MAE标准差|\n|---|---:|---:|---:|\n'
 for r in summary:text+=f"|{r['model']}|{r['training_mae']:.6f}|{r['test_mae_oof']:.6f}|{r['test_fold_mae_std']:.6f}|\n"
 text+='\n|模型|10折最低自身预测ramping|均值|标准差|最低分所在折|\n|---|---:|---:|---:|---:|\n'
 for r in summary:text+=f"|{r['model']}|{r['predicted_ramping_min']:.6f}|{r['predicted_ramping_mean']:.6f}|{r['predicted_ramping_std']:.6f}|{r['best_fold']}|\n"
 text+='\nMAE是预测能力诊断，自身预测ramping为候选比较成绩；与历史全数据模型分开记录。全部原模型/权重保留，未使用老师测试机会。模型和权重按各模型的最低自身预测分导出在best/；不是按最低测试MAE挑选。\n'
 (out/'README.md').write_text(text)
 for path,digest in (manifest['protected_files']|manifest['source_sha256']).items():assert sha(ROOT/path)==digest,path
 complete=dict(state='complete',completed_training=30,completed_search=30,summary=summary,source_and_original_files_unchanged=True)
 dump(out/'complete.json',complete);dump(out/'status.json',dict(state='complete',completed_training=30,completed_search=30));print('COMPLETE',flush=True)

def run(args):
 out=(ROOT/args.out).resolve()
 if not out.is_relative_to(ROOT):raise ValueError('--out must be inside this repository')
 out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);args.owns_output=True
 ids,X,y=data();splits=make_splits(len(y));validate_splits(splits,len(y))
 from importlib.metadata import version
 expected=dict(version=1,seed=SEED,architectures=CONFIGS,training=TRAIN,search=SEARCH,data_sha256=sha(ROOT/'Lab6/dataset_same8/train.csv'),candidate_ids=ids,n_rows=len(y),source_sha256=dependency_hashes(),versions={name:version(name) for name in ['torch','numpy','pandas','scikit-learn','scipy']})
 manifest=verify_manifest(out,expected,args.resume)
 if manifest is not None:
  if read(out/'splits.json')!=splits:raise ValueError('Unexpected fold indices')
  if (out/'complete.json').exists():print('Already complete; use check --audit',flush=True);return
 uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if not uuid.startswith('GPU-') or ',' in uuid:raise ValueError('Exactly one GPU UUID required')
 locks=ROOT.parent/'.locks';locks.mkdir(exist_ok=True);gpu_lock=(locks/('cv-'+uuid+'.lock')).open('w');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 mem,util=[int(v.strip()) for v in subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(',')]
 if mem>64 or util>0:raise RuntimeError('Selected GPU is occupied; no training started')
 if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required')
 torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if manifest is None:
  dump(out/'splits.json',splits)
  tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0');protected={path:sha(ROOT/path) for path in tracked if path and not path.startswith('cv_experiment/') and not path.endswith('.md') and (ROOT/path).is_file()}
  manifest=dict(**expected,splits_sha256=sha(out/'splits.json'),protected_files=protected,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());dump(out/'manifest.json',manifest)
 launches=read(out/'launches.json') if (out/'launches.json').exists() else [];launches.append(dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),gpu_uuid=uuid,cuda_device_count=1));dump(out/'launches.json',launches)
 if (out/'failure.json').exists():(out/'failure.json').unlink()
 records=[];trained=0;begun=time.monotonic()
 def status(name,fold,stage,epoch=None):dump(out/'status.json',dict(state='running',active_model=name,fold=fold,stage=stage,epoch=epoch,completed_training=trained,completed_search=len(records),expected_tasks=30,gpu_uuid=uuid,elapsed_seconds_current_launch=time.monotonic()-begun))
 print('START ten-fold CV; exactly one GPU '+uuid,flush=True)
 for fold in splits:
  ti=np.asarray(fold['train']);Xtrain=X[ti].copy();ytrain=y[ti].copy()
  for name,config in CONFIGS.items():
   folder=out/'models'/name/f"fold{fold['fold']:02d}";folder.mkdir(parents=True,exist_ok=True);status(name,fold['fold'],'training')
   if (folder/'training_complete.json').exists():training=load_completed_training(folder)
   else:
    def progress(epoch,history):
     dump(folder/'history.json',history);status(name,fold['fold'],'training',epoch)
     if epoch%100==0:print(f"TRAIN {name} fold={fold['fold']} epoch={epoch}",flush=True)
    model,history=train_fold(Xtrain,ytrain,config,fold['init_seed'],fold['shuffle_seed'],'cuda',callback=progress)
    save_atomic(model,folder/'model.pt');restored=load(folder/'model.pt','cuda')
    with torch.no_grad():torch.testing.assert_close(model(torch.tensor(Xtrain[:32],dtype=torch.float32,device='cuda')),restored(torch.tensor(Xtrain[:32],dtype=torch.float32,device='cuda')),atol=0,rtol=0)
    del model,restored;torch.cuda.empty_cache();cpu=load(folder/'model.pt');metrics=export_predictions(folder,cpu,ids,X,y,fold);del cpu
    training=dict(model=name,fold=fold['fold'],init_seed=fold['init_seed'],shuffle_seed=fold['shuffle_seed'],epochs=TRAIN['epochs'],**metrics,files={p:sha(folder/p) for p in ['model.pt','history.json','train_predictions.csv','test_predictions.csv']});dump(folder/'training_complete.json',training)
   trained+=1;status(name,fold['fold'],'search')
   if (folder/'search_complete.json').exists():search=read(folder/'search_complete.json');validate_files(folder,search)
   else:
    def stage(label):status(name,fold['fold'],label)
    weights,info=refine(folder/'model.pt',Xtrain,ytrain,fold['search_seed'],'cuda',callback=stage)
    dump(folder/'weights.json',weights.tolist());dump(folder/'search_details.json',info)
    search=dict(model=name,fold=fold['fold'],search_seed=fold['search_seed'],before=info['before'],after=info['after'],improvement=info['improvement'],negative_prediction=info['negative_prediction'],files={p:sha(folder/p) for p in ['model.pt','training_complete.json','weights.json','search_details.json']});dump(folder/'search_complete.json',search)
   record={k:v for k,v in training.items() if k!='files'};record.update(search_before=search['before'],search_after=search['after'],negative_prediction=search['negative_prediction'],checkpoint=str((folder/'model.pt').relative_to(ROOT)),weights=str((folder/'weights.json').relative_to(ROOT)))
   records.append(record);dump(out/'fold_metrics.json',records);dump(out/'summary_partial.json',summarize(records));status(name,fold['fold'],'finished')
   print(f"DONE {len(records)}/30 {name} fold={fold['fold']} train_MAE={record['train_mae']:.6f} test_MAE={record['test_mae']:.6f} own_ramping={record['search_after']:.6f}",flush=True);torch.cuda.empty_cache()
 finalize(out,records,manifest)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();a.owns_output=False
 try:run(a)
 except Exception:
  out=(ROOT/a.out).resolve()
  # A bad resume must never overwrite a valid completed experiment's markers.
  if a.owns_output and out.is_relative_to(ROOT) and out.exists() and not (out/'complete.json').exists():dump(out/'failure.json',dict(state='failed',traceback=traceback.format_exc()))
  raise
if __name__=='__main__':main()
