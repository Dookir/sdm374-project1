# All selection uses own predicted ramping, never a fixed baseline scorer.
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,time,traceback
import numpy as np
import torch
from Lab6 import baseline
from tabular_next.models import CONFIGS,Model,fit_stats,save,load
ROOT=Path(__file__).resolve().parents[1]
def dump(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); tmp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def project(w):
 rows=w.reshape(-1,7); s=rows.sort(1,descending=True).values
 prefix=s@torch.ones((7,7),device=w.device,dtype=w.dtype).triu()-1
 rho=(s-prefix/torch.arange(1,8,device=w.device)>0).sum(1)-1
 theta=prefix.gather(1,rho[:,None]).squeeze(1)/(rho+1)
 return (rows-theta[:,None]).clamp_min(0).reshape_as(w)
def search(model,starts,iterations=500):
 model.eval()
 for p in model.parameters():p.requires_grad_(False)
 device=next(model.parameters()).device; best=float('inf'); chosen=None; history=[]
 def consider(w,label,i):
  nonlocal best,chosen
  values=model(w); idx=int(values.detach().argmin()); v=float(values[idx].detach())
  if not np.isfinite(v):raise ValueError('Nonfinite search prediction')
  if v<best:best=v; chosen=w[idx].detach().clone()
  if i%100==0:history.append(dict(method=label,iteration=i,predicted_ramping=best))
  return values
 w=torch.tensor(np.tile(starts,(3,1)),dtype=torch.float32,device=device,requires_grad=True)
 steps=torch.tensor(np.repeat([.0003,.003,.03],len(starts)),device=device,dtype=torch.float32)[:,None]
 stale=0
 for i in range(iterations+1):
  old=best; values=consider(w,'projected_gradient',i); stale=0 if old-best>1e-5 else stale+1
  if i==iterations or stale>=120:break
  g=torch.autograd.grad(values.sum(),w)[0]
  with torch.no_grad():w.copy_(project(w-steps*g))
 aw=torch.tensor(starts,dtype=torch.float32,device=device,requires_grad=True)
 opt=torch.optim.Adam([aw],lr=.01)
 for i in range(201):
  opt.zero_grad(set_to_none=True); v=consider(aw,'projected_adam',i)
  if i==200:break
  v.sum().backward(); opt.step()
  with torch.no_grad():aw.copy_(project(aw))
 weights=baseline.project_weights(chosen.cpu().double().numpy()).reshape(3,7)
 with torch.no_grad():value=float(model(torch.tensor(weights.reshape(1,21),device=device,dtype=torch.float32)))
 for p in model.parameters():p.requires_grad_(True)
 return weights,dict(predicted_ramping=value,start_count=len(starts),history=history,negative_prediction=value<0)
def run(args):
 out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest.json').exists():raise RuntimeError('Output exists; choose a fresh directory')
 import fcntl
 lock=(out/'run.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if not visible.startswith('GPU-') or ',' in visible:raise ValueError('Exactly one GPU UUID required')
 raw=subprocess.check_output(['nvidia-smi','-i',visible,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 mem,util=map(lambda s:int(s.strip()),raw.split(','))
 if mem>64 or util>0:raise RuntimeError('GPU is not idle; no training started')
 if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required')
 torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.1); torch.set_num_threads(4)
 torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 _,X,y=baseline.load_table(ROOT/'Lab6/dataset_same8/train.csv'); stats=fit_stats(X,y)
 retained=json.loads((ROOT/'neural_baseline/retained/manifest.json').read_text())
 starts=np.concatenate([baseline.search_starts(X,y,64),np.random.default_rng(20261023).dirichlet(np.full(7,.1),size=(32,3)).reshape(32,21),*[np.asarray(json.loads((ROOT/'neural_baseline/retained'/r['name']/'weights.json').read_text())).reshape(1,21) for r in retained['models']]])
 protected={str(p.relative_to(ROOT)):sha(p) for folder in [ROOT/'Lab6',ROOT/'neural_baseline/retained'] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 versions={}
 from importlib.metadata import version
 for name in ['torch','tabm','rtdl-num-embeddings']:versions[name]=version(name)
 manifest=dict(selection='lowest own predicted_ramping',target='original train.csv ramping',configs=CONFIGS,seeds=args.seeds,epochs=args.epochs,checkpoints=args.checkpoints,training='normalized per-head MSE; heads averaged only for inference/search; AdamW with cosine schedule',gpu_uuid=visible,cuda_device_count=1,protected_files=protected,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tabular_next').glob('*.py')},versions=versions,reference_scores=[dict(name=r['name'],predicted_ramping=r['predicted_ramping']) for r in retained['models']],expected_candidates=len(CONFIGS)*len(args.seeds)*len(args.checkpoints))
 dump(out/'manifest.json',manifest); begun=time.monotonic(); rows=[]
 def status(state,**kwargs):dump(out/'status.json',dict(state=state,completed_candidates=len(rows),expected_candidates=manifest['expected_candidates'],elapsed_seconds=time.monotonic()-begun,gpu_uuid=visible,**kwargs))
 status('running'); print('START '+visible+' own-prediction scoring only',flush=True)
 tx=torch.tensor(X,dtype=torch.float32,device='cuda'); ty=torch.tensor(y,dtype=torch.float32,device='cuda')
 for name,cfg in CONFIGS.items():
  for seed in args.seeds:
   torch.manual_seed(seed); np.random.seed(seed); model=Model(cfg,stats).cuda()
   trial=out/'models'/f'{name}_{seed}'; trial.mkdir(parents=True)
   opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'],fused=True)
   schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=1e-5); hist=[]
   for epoch in range(1,args.epochs+1):
    model.train()
    for idx in torch.randperm(len(tx),device='cuda').split(128):
     opt.zero_grad(set_to_none=True); pred=model.normalized_heads(tx[idx]); target=((ty[idx]-model.y_mean)/model.y_scale)[:,None]
     loss=(pred-target).square().mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
    schedule.step()
    if epoch%25==0 or epoch in args.checkpoints:
     model.eval()
     with torch.no_grad():mse=float((model(tx)-ty).square().mean())
     if not np.isfinite(mse):raise ValueError('Nonfinite training MSE')
     hist.append(dict(epoch=epoch,training_mse_diagnostic=mse)); dump(trial/'history.json',hist)
     status('running',active_config=name,seed=seed,epoch=epoch)
     print(f'TRAIN {name} seed={seed} epoch={epoch} train_MSE={mse:.6g}',flush=True)
    if epoch in args.checkpoints:
     path=trial/f'epoch{epoch}.pt'; save(model,path); restored=load(path,'cuda')
     with torch.no_grad():torch.testing.assert_close(model.eval()(tx[:32]),restored(tx[:32]),atol=0,rtol=0)
     w,info=search(restored,starts); label=f'{name}_{seed}_e{epoch}'
     dump(out/'weights'/f'{label}.json',w.tolist()); dump(trial/f'search_epoch{epoch}.json',info)
     row=dict(name=label,config=name,family=cfg['family'],seed=seed,epoch=epoch,predicted_ramping=info['predicted_ramping'],negative_prediction=info['negative_prediction'],checkpoint=str(path.relative_to(ROOT)),weights=str((out/'weights'/f'{label}.json').relative_to(ROOT)))
     rows.append(row); dump(out/'comparison.json',rows); status('running',active_config=name,seed=seed,epoch=epoch)
     print(f"SCORE {len(rows)}/{manifest['expected_candidates']} {label} own_ramping={row['predicted_ramping']:.8f}",flush=True); del restored
   del model,opt,schedule; torch.cuda.empty_cache()
 best_by_family={family:min((r for r in rows if r['family']==family),key=lambda r:r['predicted_ramping']) for family in ['resmlp','tabm']}
 exported={}
 for family,row in best_by_family.items():
  target=out/'best'/family; target.mkdir(parents=True)
  shutil.copy2(ROOT/row['checkpoint'],target/'model.pt'); shutil.copy2(ROOT/row['weights'],target/'weights.json')
  info=dict(**row,model_sha256=sha(target/'model.pt'),weights_sha256=sha(target/'weights.json')); dump(target/'selection.json',info); exported[family]=info
 for p,h in protected.items():
  if sha(ROOT/p)!=h:raise RuntimeError('Protected file changed: '+p)
 complete=dict(state='complete',completed_candidates=len(rows),training_trials=len(CONFIGS)*len(args.seeds),best_by_family=exported,best_new=min(rows,key=lambda r:r['predicted_ramping']),previous_best=retained['models'][0]['predicted_ramping'],elapsed_seconds=time.monotonic()-begun,gpu_uuid=visible,cuda_device_count=1,peak_memory_bytes=torch.cuda.max_memory_allocated(),protected_files_unchanged=True)
 dump(out/'complete.json',complete); status('complete'); print('COMPLETE '+json.dumps(best_by_family),flush=True)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--out',default='tabular_next/results/run_20261023'); p.add_argument('--epochs',type=int,default=800); p.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028]); p.add_argument('--checkpoints',type=int,nargs='+',default=[50,150,400,800]); args=p.parse_args()
 if sorted(set(args.checkpoints))!=args.checkpoints or args.checkpoints[-1]!=args.epochs:p.error('Checkpoints must be ordered and end at epochs')
 try:run(args)
 except Exception:
  dump(ROOT/args.out/'failure.json',dict(state='failed',traceback=traceback.format_exc())); raise
if __name__=='__main__':main()
