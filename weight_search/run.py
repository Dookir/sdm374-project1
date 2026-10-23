# Fixed checkpoints, expanded candidate coverage, own predicted ramping only.
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,hashlib,json,subprocess,time,traceback,fcntl
import numpy as np,torch
from scipy.optimize import minimize
from Lab6 import baseline
from neural_baseline.models import load as load_old
from tabular_next.models import load as load_new
from tabular_next.run import project,dump
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def catalog():
 records=[]; base=ROOT/'neural_baseline/retained'
 for r in json.loads((base/'manifest.json').read_text())['models']:
  records.append(dict(name=r['name'],family='mlp',loader='old',checkpoint=str((base/r['name']/'model.pt').relative_to(ROOT)),weights=str((base/r['name']/'weights.json').relative_to(ROOT)),previous_score=r['predicted_ramping']))
 rows=json.loads((ROOT/'tabular_next/results/run_20261023/comparison.json').read_text())
 for name in dict.fromkeys(r['config'] for r in rows):
  r=min((r for r in rows if r['config']==name),key=lambda r:r['predicted_ramping'])
  records.append(dict(name=name,family=r['family'],loader='new',checkpoint=r['checkpoint'],weights=r['weights'],previous_score=r['predicted_ramping'],original_candidate=r['name']))
 return records
def getmodel(r,device='cpu'):return (load_old if r['loader']=='old' else load_new)(ROOT/r['checkpoint'],device)
def evaluate(model,x,batch=512):
 with torch.no_grad():return torch.cat([model(c) for c in x.split(batch)]).detach()
def mixed_candidates(rng,X,y,incumbent,other):
 parts=[rng.dirichlet(np.full(7,a),size=(8192,3)).reshape(-1,21) for a in [.03,.2,1.,5.]]
 sparse=rng.dirichlet(np.ones(7),size=(16384,3)); mask=rng.random(sparse.shape)>.55; mask[:,:,0]|=~mask.any(-1); sparse*=mask; sparse/=sparse.sum(-1,keepdims=True); parts.append(sparse.reshape(-1,21))
 near=incumbent[None,:]+rng.normal(size=(8192,21))*rng.choice([.005,.02,.08,.2],size=(8192,1)); parts.append(baseline.project_weights(near))
 best=X[np.argsort(y)[:50]]; near=best[rng.integers(0,50,size=8192)]+rng.normal(size=(8192,21))*.08; parts.append(baseline.project_weights(near))
 vertices=[]
 for a in range(7):
  for b in range(7):
   for c in range(7):
    w=np.zeros((3,7)); w[0,a]=w[1,b]=w[2,c]=1; vertices.append(w.ravel())
 return np.concatenate([incumbent[None,:],other,X[np.argsort(y)[:50]],*parts,np.asarray(vertices)])
def choose_seeds(points,scores,incumbent,n=64):
 chosen=[incumbent]; order=torch.argsort(scores)[:2048].cpu().tolist()
 for i in order:
  candidate=points[i]
  if float(torch.linalg.vector_norm(torch.stack(chosen)-candidate,dim=1).min())>.04:chosen.append(candidate)
  if len(chosen)==n:break
 if len(chosen)<n:
  for i in order:
   chosen.append(points[i])
   if len(chosen)==n:break
 return torch.stack(chosen)
def gradient_refine(model,seeds,iterations,adam=False):
 device=seeds.device; n=len(seeds); w=seeds.repeat(3,1).clone().requires_grad_(True)
 rates=torch.tensor([.002,.01,.05] if adam else [.0001,.001,.01],device=device).repeat_interleave(n)[:,None]
 with torch.no_grad():best_scores=model(w).detach(); best_points=w.detach().clone()
 m=torch.zeros_like(w); v=torch.zeros_like(w)
 for i in range(1,iterations+1):
  pred=model(w); g=torch.autograd.grad(pred.sum(),w)[0]
  with torch.no_grad():
   better=pred<best_scores; best_scores=torch.where(better,pred,best_scores); best_points[better]=w[better]
   if adam:
    m.mul_(.9).add_(g,alpha=.1); v.mul_(.999).addcmul_(g,g,value=.001)
    delta=(m/(1-.9**i))/((v/(1-.999**i)).sqrt()+1e-8)
    factor=.05+.95*.5*(1+np.cos(np.pi*i/iterations))
   else:delta=g; factor=1.
   w.copy_(project(w-rates*factor*delta))
 with torch.no_grad():
  pred=model(w); better=pred<best_scores; best_scores=torch.where(better,pred,best_scores); best_points[better]=w[better]
 return best_points,best_scores
def coordinate_refine(model,w,rounds=30):
 best=w.clone(); bestscore=float(model(best[None,:]).detach()); count=0
 for _ in range(rounds):
  variants=[]
  for b in range(3):
   for i in range(7):
    for j in range(7):
     if i==j:continue
     for fraction in [.01,.05,.2,.5,1.]:
      z=best.clone(); transfer=z[b*7+i]*fraction; z[b*7+i]-=transfer; z[b*7+j]+=transfer; variants.append(z)
  points=torch.stack(variants); values=evaluate(model,points); index=int(values.argmin()); count+=len(points)
  if float(values[index])>=bestscore-1e-6:break
  best=points[index]; bestscore=float(values[index])
 return best,bestscore,count
def run(args):
 out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest.json').exists():raise ValueError('Choose fresh output directory')
 lock=(out/'run.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if not uuid.startswith('GPU-') or ',' in uuid:raise ValueError('Exactly one GPU UUID required')
 mem,util=[int(s.strip()) for s in subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(',')]
 if mem>64 or util>0:raise RuntimeError('GPU occupied, no search started')
 assert torch.cuda.device_count()==1; torch.cuda.set_per_process_memory_fraction(.1); torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False
 records=catalog(); protected={r[k]:sha(ROOT/r[k]) for r in records for k in ['checkpoint','weights']}
 protected.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'Lab6').rglob('*') if p.is_file() and '__pycache__' not in p.parts})
 dump(out/'manifest.json',dict(records=records,protected_files=protected,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'weight_search').glob('*.py')},selection='lowest own predicted_ramping with fixed checkpoints',seed=20261023,candidate_sampling='65536 mixed random/local + 343 exact vertices + previous candidates + 50 measured best',pgd_iterations=800,adam_iterations=1000,slsqp_starts=8,slsqp_iterations=200,gpu_uuid=uuid,cuda_device_count=1))
 _,X,y=baseline.load_table(ROOT/'Lab6/dataset_same8/train.csv'); other=np.stack([np.array(json.loads((ROOT/r['weights']).read_text())).ravel() for r in records]); begun=time.monotonic(); results=[]
 def status(active,stage):dump(out/'status.json',dict(state='running',completed_models=len(results),total_models=len(records),active_model=active,stage=stage,elapsed_seconds=time.monotonic()-begun))
 print('START fixed-checkpoint own-score search '+uuid,flush=True)
 for index,r in enumerate(records):
  status(r['name'],'sampling'); model=getmodel(r,'cuda'); model.eval()
  for p in model.parameters():p.requires_grad_(False)
  oldw=other[index]; rng=np.random.default_rng(20261023+index); sampled=mixed_candidates(rng,X,y,oldw,other)
  points=torch.tensor(sampled,dtype=torch.float32,device='cuda'); scores=evaluate(model,points)
  if not torch.isfinite(scores).all():raise ValueError('Nonfinite sampled predictions')
  incumbent=torch.tensor(oldw,dtype=torch.float32,device='cuda'); seeds=choose_seeds(points,scores,incumbent)
  stages=[dict(stage='sampling',predicted_ramping=float(scores.min()),evaluated=len(points))]; best=points[int(scores.argmin())].clone()
  status(r['name'],'projected_gradient'); pp,pv=gradient_refine(model,seeds,800); stages.append(dict(stage='projected_gradient',predicted_ramping=float(pv.min())))
  if float(pv.min())<float(model(best[None,:]).detach()):best=pp[int(pv.argmin())].clone()
  status(r['name'],'projected_adam'); ap,av=gradient_refine(model,seeds,1000,adam=True); stages.append(dict(stage='projected_adam',predicted_ramping=float(av.min())))
  if float(av.min())<float(model(best[None,:]).detach()):best=ap[int(av.argmin())].clone()
  status(r['name'],'coordinate_transfer'); best,score,coordcount=coordinate_refine(model,best); stages.append(dict(stage='coordinate_transfer',predicted_ramping=score,evaluated=coordcount))
  status(r['name'],'cpu_slsqp'); cpu=getmodel(r).double(); cpu.eval()
  for p in cpu.parameters():p.requires_grad_(False)
  matrix=np.kron(np.eye(3),np.ones((1,7)))
  def fg(v):
   tensor=torch.tensor(v.reshape(1,21),dtype=torch.float64,requires_grad=True); value=cpu(tensor); grad=torch.autograd.grad(value.sum(),tensor)[0]; return float(value.detach()),grad.detach().numpy().ravel()
  candidates=[best.detach().cpu().double().numpy()]+[p.detach().cpu().double().numpy() for p in torch.cat([pp[pv.argsort()[:4]],ap[av.argsort()[:3]]])]
  polish=[]
  for start in candidates:
   result=minimize(fg,start,jac=True,method='SLSQP',bounds=[(0.,1.)]*21,constraints=[dict(type='eq',fun=lambda v:matrix@v-1,jac=lambda v:matrix)],options=dict(ftol=1e-10,maxiter=200))
   if np.isfinite(result.x).all():
    w=baseline.project_weights(result.x).reshape(1,21); tensor=torch.tensor(w,dtype=torch.float32,device='cuda')
    with torch.no_grad():v=float(model(tensor))
    polish.append(dict(success=bool(result.success),predicted_ramping=v,message=str(result.message)))
    if v<score:best=tensor[0]; score=v
  stages.append(dict(stage='slsqp',predicted_ramping=score,runs=polish))
  weights=baseline.project_weights(best.cpu().double().numpy()).reshape(3,7)
  ref=getmodel(r)
  with torch.no_grad():
   before=float(ref(torch.tensor(oldw[None,:],dtype=torch.float32))); after=float(ref(torch.tensor(weights.reshape(1,21),dtype=torch.float32)))
  if after>=before:weights=oldw.reshape(3,7); after=before
  dump(out/'weights'/f"{r['name']}.json",weights.tolist())
  row=dict(name=r['name'],family=r['family'],checkpoint=r['checkpoint'],checkpoint_sha256=sha(ROOT/r['checkpoint']),before=before,after=after,improvement=before-after,negative_prediction=after<0,weights=str((out/'weights'/f"{r['name']}.json").relative_to(ROOT)),stages=stages)
  dump(out/'details'/f"{r['name']}.json",row); results.append(row); dump(out/'comparison.json',results)
  print(f"DONE {len(results)}/{len(records)} {r['name']}: {before:.7f} -> {after:.7f}",flush=True)
  del model,cpu,ref,points,scores,seeds,pp,pv,ap,av,best; torch.cuda.empty_cache()
 for p,h in protected.items():assert sha(ROOT/p)==h,p
 complete=dict(state='complete',completed_models=len(results),results=results,best=min(results,key=lambda r:r['after']),elapsed_seconds=time.monotonic()-begun,cuda_device_count=1,gpu_uuid=uuid,checkpoints_unchanged=True)
 dump(out/'complete.json',complete); dump(out/'status.json',dict(state='complete',completed_models=len(results))); print('COMPLETE',flush=True)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--out',default='weight_search/results/run_20261023'); a=p.parse_args()
 try:run(a)
 except Exception:dump(ROOT/a.out/'failure.json',dict(state='failed',traceback=traceback.format_exc())); raise
if __name__=='__main__':main()
