# Only fold-training arrays and the fold checkpoint are accepted as search inputs.
import numpy as np,torch
from scipy.optimize import minimize
from Lab6 import baseline
from neural_baseline.models import load
from weight_search.run import mixed_candidates,choose_seeds,gradient_refine,coordinate_refine,evaluate
from cv_experiment.common import SEARCH
from cv_experiment.training import predict_rows

def refine(checkpoint,X_train,y_train,seed,device,callback=None,settings=None):
 cfg=SEARCH if settings is None else settings
 ref=load(checkpoint);pred=predict_rows(ref,X_train);incumbent=np.array(X_train[int(pred.argmin())],copy=True)
 before=float(predict_rows(ref,incumbent[None,:])[0]);rng=np.random.default_rng(seed)
 sampled=mixed_candidates(rng,X_train,y_train,incumbent,X_train)
 model=load(checkpoint,device)
 for p in model.parameters():p.requires_grad_(False)
 def stage(name):
  if callback:callback(name)
 stage('sampling');points=torch.tensor(sampled,dtype=torch.float32,device=device);scores=evaluate(model,points)
 if not torch.isfinite(scores).all():raise ValueError('Nonfinite candidate score')
 seeds=choose_seeds(points,scores,torch.tensor(incumbent,dtype=torch.float32,device=device),cfg['seed_count'])
 best=points[int(scores.argmin())].clone();score=float(scores.min());stages=[dict(stage='sampling',predicted_ramping=score,candidates=len(sampled))]
 stage('projected_gradient');pp,pv=gradient_refine(model,seeds,cfg['pgd_iterations']);v=float(pv.min());stages.append(dict(stage='projected_gradient',predicted_ramping=v))
 if v<score:best=pp[int(pv.argmin())].clone();score=v
 stage('projected_adam');ap,av=gradient_refine(model,seeds,cfg['adam_iterations'],adam=True);v=float(av.min());stages.append(dict(stage='projected_adam',predicted_ramping=v))
 if v<score:best=ap[int(av.argmin())].clone();score=v
 stage('coordinate_transfer');best,score,count=coordinate_refine(model,best,cfg['coordinate_rounds']);stages.append(dict(stage='coordinate_transfer',predicted_ramping=score,candidates=count))
 stage('cpu_slsqp');cpu=load(checkpoint).double()
 for p in cpu.parameters():p.requires_grad_(False)
 matrix=np.kron(np.eye(3),np.ones((1,7)))
 def fg(v):
  tensor=torch.tensor(v.reshape(1,21),dtype=torch.float64,requires_grad=True);value=cpu(tensor);gradient=torch.autograd.grad(value.sum(),tensor)[0]
  return float(value.detach()),gradient.detach().numpy().ravel()
 starts=[best.detach().cpu().double().numpy()]+[p.detach().cpu().double().numpy() for p in torch.cat([pp[pv.argsort()[:4]],ap[av.argsort()[:3]]])]
 polished=[]
 for start in starts[:cfg['slsqp_starts']]:
  result=minimize(fg,start,jac=True,method='SLSQP',bounds=[(0.,1.)]*21,constraints=[dict(type='eq',fun=lambda v:matrix@v-1,jac=lambda v:matrix)],options=dict(ftol=1e-10,maxiter=cfg['slsqp_iterations']))
  if np.isfinite(result.x).all():
   weights=baseline.project_weights(result.x);tensor=torch.tensor(weights.reshape(1,21),dtype=torch.float32,device=device)
   with torch.no_grad():v=float(model(tensor))
   polished.append(dict(success=bool(result.success),predicted_ramping=v,message=str(result.message)))
   if v<score:best=tensor[0].clone();score=v
 stages.append(dict(stage='cpu_slsqp',predicted_ramping=score,runs=polished))
 weights=baseline.project_weights(best.detach().cpu().double().numpy()).reshape(3,7)
 after=float(predict_rows(ref,weights.reshape(1,21))[0])
 if after>=before:weights=incumbent.reshape(3,7);after=before
 return weights,dict(before=before,after=after,improvement=before-after,negative_prediction=after<0,reference='lowest own prediction among fold-training rows',reference_weights=incumbent.reshape(3,7).tolist(),stages=stages)
