import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,fcntl,json,subprocess,time,traceback,itertools
import numpy as np,pandas as pd,torch
from scipy.optimize import differential_evolution,minimize
from scipy.spatial import cKDTree
from bayes_experiment.common import ROOT,sha,read,dump,data
from Lab6.baseline import project_weights,FEATURES
from neural_baseline.models import load
from cv_experiment.training import predict_rows
from tabular_next.run import project
CHECKPOINT=ROOT/'final_mlp256/results/full_20261030/model.pt'
PREVIOUS=ROOT/'final_mlp256_search/results/extended_20261030'
SEEDS=[20261041,20261042,20261043]

def population(X,y,seed):
 rng=np.random.default_rng(seed)
 vertices=np.zeros((343,3,7))
 for k,combo in enumerate(itertools.product(range(7),repeat=3)):
  for b,j in enumerate(combo):vertices[k,b,j]=1
 random=np.concatenate([rng.dirichlet(np.full(7,a),size=(30,3)).reshape(30,21) for a in [.03,.2,1.,5.]])
 result=np.vstack([vertices.reshape(343,21),random,X[np.argsort(y)[:48]],np.full((1,21),1/7)])
 assert result.shape==(512,21);return result

def polish(weights,cpu):
 start=project_weights(np.asarray(weights).reshape(21));before=float(predict_rows(cpu,start[None,:])[0]);model=load(CHECKPOINT).double()
 for p in model.parameters():p.requires_grad_(False)
 matrix=np.kron(np.eye(3),np.ones((1,7)))
 def fg(v):
  x=torch.tensor(v[None,:],dtype=torch.float64,requires_grad=True);z=model(x);g=torch.autograd.grad(z.sum(),x)[0];return float(z.detach()),g.detach().numpy().ravel()
 r=minimize(fg,start,jac=True,method='SLSQP',bounds=[(0.,1.)]*21,constraints=[dict(type='eq',fun=lambda v:matrix@v-1,jac=lambda v:matrix)],options=dict(maxiter=500,ftol=1e-10))
 candidate=project_weights(r.x);after=float(predict_rows(cpu,candidate[None,:])[0])
 return (candidate if after<before else start),dict(raw_cpu_prediction=before,polished_cpu_prediction=min(before,after),slsqp_success=bool(r.success),slsqp_message=str(r.message))

def diagnostics(X,y,ids,cpu,w,b):
 tree=cKDTree(X);nn=tree.query(X,k=2)[0][:,1];d,i=tree.query(w.reshape(21),k=10);p=float(predict_rows(cpu,w.reshape(1,21))[0]);bp=float(predict_rows(cpu,b.reshape(1,21))[0]);actual=120.21789791621268;ba=119.09161796723492
 f=pd.read_csv(ROOT/'bayes_accel/results/bo10_20261029/oof_predictions.csv',float_precision='round_trip');f=f[f.model=='relu_256_128'];err=np.abs(f.y_true-f.y_pred)
 return dict(weights_feature_order=FEATURES,local_download_expected_sha256='bcc1bb7ac10a3ace5f9d52ca386023f3d5c8b9251a6cc39fe973ecf085e93de4',submitted_weights_sha256=sha(PREVIOUS/'weights.json'),model_sha256=sha(CHECKPOINT),candidate_predicted_ramping=p,candidate_user_reported_evaluator_ramping=actual,candidate_error=actual-p,baseline_predicted_by_same_model=bp,baseline_user_reported_evaluator_ramping=ba,baseline_error=ba-bp,actual_candidate_minus_baseline=actual-ba,candidate_near_zero_weights=int((w<1e-7).sum()),training_max_near_zero_weights=int((X<1e-7).sum(1).max()),training_median_near_zero_weights=float(np.median((X<1e-7).sum(1))),candidate_nearest_training_distance=float(d[0]),training_nn_distance_percentile=float(np.mean(nn<=d[0])*100),nn_distance_quantiles={str(q):float(np.quantile(nn,q)) for q in [.5,.9,.95,.99,1.]},nearest_training_rows=[dict(candidate_id=ids[int(k)],distance=float(dist),actual=float(y[int(k)])) for dist,k in zip(d,i)],prior_oof_mae=float(err.mean()),prior_oof_abs_error_quantiles={str(q):float(np.quantile(err,q)) for q in [.5,.9,.95,.99,1.]},prior_oof_errors_at_least_current=int((err>=actual-p).sum()),n_oof=len(err),real_evaluations_in_this_run=0)

def run(args):
 out=(ROOT/args.out).resolve();assert out.is_relative_to(ROOT) and not out.exists();out.mkdir(parents=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 uuid=os.environ.get('CUDA_VISIBLE_DEVICES','');assert uuid.startswith('GPU-') and ',' not in uuid
 gpu_lock=(ROOT.parent/'.locks'/('cv-'+uuid+'.lock')).open('w');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 mem,util=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(',')]
 if mem>64 or util>0:dump(out/'status.json',dict(state='paused',reason='Selected GPU occupied'));return
 assert torch.cuda.device_count()==1;torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split(chr(0));protected={p:sha(ROOT/p) for p in tracked if p and not p.startswith('heuristic_check/') and not p.endswith('.md') and (ROOT/p).is_file()}
 dump(out/'manifest.json',dict(model_sha256=sha(CHECKPOINT),source_sha256=sha(Path(__file__)),gpu_uuid=uuid,protected_files=protected,seeds=SEEDS,population=512,generations=500,initial_population_contains_old_optimum=False))
 ids,X,y=data();cpu=load(CHECKPOINT);model=load(CHECKPOINT,'cuda')
 for p in model.parameters():p.requires_grad_(False)
 old=read(PREVIOUS/'complete.json');w0=np.asarray(read(PREVIOUS/'weights.json'));bw=np.asarray(read(ROOT/'Lab6/weights.json'));dump(out/'diagnostics.json',diagnostics(X,y,ids,cpu,w0,bw));rows=[];started=time.monotonic()
 def evaluate(points):
  with torch.inference_mode():return model(points)
 def status(method,seed,step):dump(out/'status.json',dict(state='running',method=method,seed=seed,step=step,gpu_uuid=uuid,elapsed_seconds=time.monotonic()-started))
 for seed in SEEDS:
  initial=population(X,y,seed)
  for method in ['differential_evolution','particle_swarm']:
   begin=time.monotonic();status(method,seed,0)
   if method=='differential_evolution':
    counts=[0]
    def objective(values):
     points=torch.tensor(np.asarray(values).T,dtype=torch.float32,device='cuda');points=project(points)
     scores=evaluate(points).detach().cpu().numpy();counts[0]+=len(points)
     return scores
    generation=[0]
    def callback(intermediate_result):
     generation[0]+=1
     if generation[0]%25==0:status(method,seed,generation[0])
    result=differential_evolution(objective,[(0.,1.)]*21,init=initial,maxiter=500,popsize=1,mutation=(.5,1.),recombination=.9,rng=np.random.default_rng(seed),updating='deferred',vectorized=True,workers=1,polish=False,tol=1e-9,atol=1e-6,callback=callback)
    raw=project_weights(result.x);evaluations=counts[0];iterations=result.nit;converged=bool(result.success)
   else:
    generator=torch.Generator(device='cuda').manual_seed(seed);position=torch.tensor(initial,dtype=torch.float32,device='cuda');velocity=torch.zeros_like(position);personal=position.clone();scores=evaluate(position);best_scores=scores.clone()
    for k in range(500):
     global_best=personal.index_select(0,best_scores.argmin().reshape(1))[0]
     r1=torch.rand(position.shape,generator=generator,device='cuda');r2=torch.rand(position.shape,generator=generator,device='cuda')
     velocity=(.729*velocity+1.49445*r1*(personal-position)+1.49445*r2*(global_best-position)).clamp(-.15,.15)
     next_position=project(position+velocity);velocity=next_position-position;position=next_position;scores=evaluate(position);mask=scores<best_scores;personal=torch.where(mask[:,None],position,personal);best_scores=torch.minimum(scores,best_scores)
     if (k+1)%25==0:status(method,seed,k+1)
    raw=personal.index_select(0,best_scores.argmin().reshape(1))[0].detach().cpu().double().numpy();evaluations=512*501;iterations=500;converged=None
   weights,detail=polish(raw,cpu);weights=weights.reshape(3,7);prediction=float(predict_rows(cpu,weights.reshape(1,21))[0]);folder=out/(method+'_'+str(seed));folder.mkdir();dump(folder/'weights.json',weights.tolist())
   row=dict(method=method,seed=seed,predicted_ramping=prediction,**detail,evaluations=evaluations,iterations=iterations,converged=converged,elapsed_seconds=time.monotonic()-begin,weights=str((folder/'weights.json').relative_to(ROOT)));rows.append(row);dump(folder/'result.json',row);dump(out/'comparison_partial.json',rows);print('DONE '+json.dumps(row),flush=True)
 # Preserve the previous best; new heuristics never overwrite the supplied candidate.
 best=min(rows,key=lambda r:r['predicted_ramping']);chosen=(ROOT/best['weights']) if best['predicted_ramping']<old['predicted_ramping'] else PREVIOUS/'weights.json';best_weights=read(chosen);dump(out/'weights.json',best_weights)
 final_pred=float(predict_rows(cpu,np.asarray(best_weights).reshape(1,21))[0]);record=dict(state='complete',previous_gradient_based_best=old['predicted_ramping'],best_new_heuristic_prediction=best['predicted_ramping'],selected_predicted_ramping=final_pred,improvement=old['predicted_ramping']-final_pred,model_unchanged=True,model_retrained=False,negative_prediction=final_pred<0,results=rows,real_evaluations_used=0,global_optimum_proven=False,elapsed_seconds=time.monotonic()-started)
 for row in rows:
  w=np.asarray(read(ROOT/row['weights']));assert w.shape==(3,7) and (w>=0).all() and np.allclose(w.sum(1),1,atol=1e-12,rtol=0);assert abs(float(predict_rows(cpu,w.reshape(1,21))[0])-row['predicted_ramping'])<1e-5
 for path,digest in protected.items():assert sha(ROOT/path)==digest,path
 dump(out/'audit.json',dict(passed=True,heuristic_candidates=6,weights_legal=True,cpu_predictions_reproduced=True,model_and_old_files_unchanged=True,previous_best_not_worse=True));dump(out/'complete.json',record);dump(out/'status.json',dict(state='complete',selected_predicted_ramping=final_pred));pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False);print('COMPLETE '+json.dumps(record),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='heuristic_check/results/check_20261030');a=p.parse_args()
 try:run(a)
 except Exception:
  out=ROOT/a.out
  if out.exists():dump(out/'failure.json',dict(state='failed',traceback=traceback.format_exc()))
  raise
if __name__=='__main__':main()
