import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,fcntl,json,subprocess,time,traceback
import numpy as np,torch
from bayes_experiment.common import ROOT,read,dump,sha,data
from cv_experiment.common import SEARCH
from cv_experiment.optimize import refine
from cv_experiment.training import predict_rows
from neural_baseline.models import load
CHECKPOINT=ROOT/'final_mlp256/results/full_20261030/model.pt'
SEEDS=[20261030,20261031,20261032]

def audit(out):
 torch.set_num_threads(4);record=read(out/'complete.json');manifest=read(out/'manifest.json');model=load(CHECKPOINT);_,X,y=data()
 for path,digest in manifest['protected_files'].items():assert sha(ROOT/path)==digest,path
 assert sha(CHECKPOINT)==manifest['model_sha256']
 rows=[]
 for seed in SEEDS:
  folder=out/str(seed);info=read(folder/'search.json');w=np.asarray(read(folder/'weights.json'))
  assert w.shape==(3,7) and np.isfinite(w).all() and (w>=0).all() and np.allclose(w.sum(1),1,atol=1e-12,rtol=0)
  prediction=float(predict_rows(model,w.reshape(1,21))[0]);assert abs(prediction-info['after'])<1e-5 and prediction<=info['before']+1e-6;rows.append((prediction,seed))
 best=min(rows);w=np.asarray(read(out/'weights.json'));prediction=float(predict_rows(model,w.reshape(1,21))[0])
 assert record['best_seed']==best[1] and abs(record['predicted_ramping']-best[0])<1e-5 and prediction==record['predicted_ramping']
 assert record['model_sha256']==sha(CHECKPOINT) and not record['model_retrained'] and record['negative_prediction']==(prediction<0)
 report=dict(passed=True,weights_legal=True,cpu_predicted_ramping=prediction,three_searches_reproduced=True,model_and_old_files_unchanged=True)
 dump(out/'audit.json',report);print(json.dumps(report),flush=True)

def run(args):
 out=(ROOT/args.out).resolve()
 if not out.is_relative_to(ROOT):raise ValueError('Output must be inside project')
 out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);args.owns_output=True
 if (out/'complete.json').exists():raise ValueError('Completed output already exists')
 uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if not uuid.startswith('GPU-') or ',' in uuid:raise ValueError('Exactly one GPU UUID required')
 locks=ROOT.parent/'.locks';locks.mkdir(exist_ok=True);gpu_lock=(locks/('cv-'+uuid+'.lock')).open('w');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 mem,util=[int(v.strip()) for v in subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(',')]
 if mem>64 or util>0:dump(out/'status.json',dict(state='paused',reason='Selected GPU no longer idle'));print('PAUSED: no idle selected GPU',flush=True);return
 if torch.cuda.device_count()!=1:raise ValueError('Exactly one visible CUDA device required')
 torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split(chr(0));protected={p:sha(ROOT/p) for p in tracked if p and not p.startswith('final_mlp256_search/') and not p.endswith('.md') and (ROOT/p).is_file()}
 manifest=dict(model=str(CHECKPOINT.relative_to(ROOT)),model_sha256=sha(CHECKPOINT),seeds=SEEDS,search_settings=SEARCH,gpu_uuid=uuid,cuda_device_count=1,protected_files=protected,source_sha256=sha(Path(__file__)));dump(out/'manifest.json',manifest)
 _,X,y=data();results=[];started=time.monotonic()
 for i,seed in enumerate(SEEDS):
  folder=out/str(seed);folder.mkdir(exist_ok=True)
  def progress(stage):dump(out/'status.json',dict(state='running',search=i+1,total_searches=3,stage=stage,gpu_uuid=uuid,elapsed_seconds=time.monotonic()-started))
  w,info=refine(CHECKPOINT,X,y,seed,'cuda',callback=progress);dump(folder/'weights.json',w.tolist());dump(folder/'search.json',info)
  results.append(dict(seed=seed,predicted_ramping=info['after'],before=info['before'],nearest_training_distance=float(np.linalg.norm(X-w.reshape(1,21),axis=1).min())))
  print('DONE '+json.dumps(results[-1]),flush=True)
 best=min(results,key=lambda r:r['predicted_ramping']);dump(out/'weights.json',read(out/str(best['seed'])/'weights.json'))
 record=dict(state='complete',model=str(CHECKPOINT.relative_to(ROOT)),model_sha256=sha(CHECKPOINT),model_retrained=False,predicted_ramping=best['predicted_ramping'],best_seed=best['seed'],negative_prediction=best['predicted_ramping']<0,nearest_training_distance=best['nearest_training_distance'],searches=results,global_optimum_proven=False,elapsed_seconds=time.monotonic()-started)
 dump(out/'complete.json',record);audit(out);dump(out/'status.json',dict(state='complete',predicted_ramping=best['predicted_ramping']))
 (out/'README.md').write_text('# 全数据最终MLP的控制权重寻优\n\n固定final_mlp256/results/full_20261030/model.pt，不重新训练。使用三个固定种子，每组约6.5万候选、64起点、800步投影梯度、1000步投影Adam、权重转移与8起点SLSQP。仅优化原3×7合法权重上的本模型自身预测ramping，CPU复核。\n\n最低找到的predicted_ramping：**'+str(best['predicted_ramping'])+'**。这是搜索找到的最低预测值，不是全局最优证明。权重为weights.json，nearest_training_distance仅诊断，不加入评分。负预测字段：'+str(record['negative_prediction'])+'。未消耗老师测试机会，原模型与冻结包不变。\n\n复现：在仓库根目录运行 `CUDA_VISIBLE_DEVICES= python -m cv_experiment.predict --checkpoint final_mlp256/results/full_20261030/model.pt --weights final_mlp256_search/results/run_20261030/weights.json`。\n')
 print('COMPLETE '+json.dumps(record),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='final_mlp256_search/results/run_20261030');p.add_argument('--audit',action='store_true');a=p.parse_args();a.owns_output=False
 if a.audit:audit(ROOT/a.out);return
 try:run(a)
 except Exception:
  if a.owns_output:dump(ROOT/a.out/'failure.json',dict(state='failed',traceback=traceback.format_exc()))
  raise
if __name__=='__main__':main()
