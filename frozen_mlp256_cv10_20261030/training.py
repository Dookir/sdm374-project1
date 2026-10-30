import os,math
from pathlib import Path
import numpy as np,torch
from model import MLP
class EarlyStopper:
 def __init__(self,patience=30,min_delta=0.):
  self.patience=patience;self.min_delta=min_delta;self.best_value=float('inf');self.best_epoch=0;self.bad_epochs=0;self.state=None
 def update(self,epoch,value,model):
  if not math.isfinite(value):raise ValueError('Nonfinite validation MAE')
  if value<self.best_value-self.min_delta:
   self.best_value=value;self.best_epoch=epoch;self.bad_epochs=0;self.state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  else:self.bad_epochs+=1
  return self.bad_epochs>=self.patience
 def restore(self,model):
  if self.state is None:raise ValueError('No validated checkpoint')
  model.load_state_dict(self.state);return model.eval()
from helpers import save_atomic,predict_rows
import json
TRAIN=json.loads(Path(__file__).with_name('training_constants.json').read_text())
class TrainingPaused(Exception):pass
class GPUStopper(EarlyStopper):
 def update(self,epoch,value,model):
  if not math.isfinite(value):raise ValueError('Nonfinite validation MAE')
  if value<self.best_value:
   self.best_value=value;self.best_epoch=epoch;self.bad_epochs=0
   self.state={k:v.detach().clone() for k,v in model.state_dict().items()}
  else:self.bad_epochs+=1
  return self.bad_epochs>=self.patience

def fit(X,y,config,init_seed,shuffle_seed,device,epochs=600,validation=None,callback=None,patience=30,resume_path=None,pause_path=None):
 torch.manual_seed(init_seed);np.random.seed(init_seed)
 model=MLP(config).fit_stats(np.array(X,copy=True),np.array(y,copy=True)).to(device)
 x=torch.tensor(X,dtype=torch.float32,device=device);target=torch.tensor(y,dtype=torch.float32,device=device)
 generator=torch.Generator(device=device).manual_seed(shuffle_seed)
 optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'],fused=str(device).startswith('cuda'))
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=600,eta_min=config['learning_rate']*.01)
 stopper=GPUStopper(patience,0.) if validation is not None else None
 if validation is not None:
  vx=torch.tensor(validation[0],dtype=torch.float32,device=device);vy=torch.tensor(validation[1],dtype=torch.float64,device=device)
 history=[];start=1
 if resume_path and Path(resume_path).exists():
  checkpoint=torch.load(resume_path,map_location=device,weights_only=False)
  assert checkpoint['config']==config and checkpoint['init_seed']==init_seed and checkpoint['shuffle_seed']==shuffle_seed and checkpoint['epochs']==epochs
  model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
  history=checkpoint['history'];start=checkpoint['epoch']+1
  if stopper is not None:
   for k,v in checkpoint['stopper'].items():setattr(stopper,k,v)
  torch.set_rng_state(checkpoint['cpu_rng'].cpu());np.random.set_state(checkpoint['numpy_rng']);generator.set_state(checkpoint['shuffle_rng'].cpu())
  if str(device).startswith('cuda'):torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu())
 for epoch in range(start,epochs+1):
  model.train();total=torch.zeros((),dtype=torch.float64,device=device)
  for idx in torch.randperm(len(x),device=device,generator=generator).split(TRAIN['batch_size']):
   optimizer.zero_grad(set_to_none=True);loss=(((model(x[idx])-target[idx])/model.y_scale)**2).mean()
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total.add_(loss.detach().double()*len(idx))
  scheduler.step();model.eval()
  if stopper is not None:
   with torch.no_grad():val=(model(vx).double()-vy).abs().mean()
   # One GPU->CPU transfer per epoch, rather than two per minibatch.
   values=torch.stack([total,val]).detach().cpu().tolist();total_value,value=values
  else:total_value=float(total)
  if not math.isfinite(total_value):raise ValueError('Nonfinite loss detected at epoch boundary')
  item=dict(epoch=epoch,learning_rate=float(scheduler.get_last_lr()[0]),normalized_training_mse=total_value/len(x));stop=False
  if stopper is not None:item['validation_mae']=value;stop=stopper.update(epoch,value,model)
  history.append(item)
  if callback and (epoch%10==0 or stop or epoch==epochs):callback(epoch,history)
  if stop or epoch==epochs:break
  if pause_path and epoch%10==0 and Path(pause_path).exists():
   checkpoint=dict(config=config,init_seed=init_seed,shuffle_seed=shuffle_seed,epochs=epochs,epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,cpu_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),shuffle_rng=generator.get_state(),cuda_rng=torch.cuda.get_rng_state() if str(device).startswith('cuda') else None,stopper={k:getattr(stopper,k) for k in ['best_value','best_epoch','bad_epochs','state']} if stopper else None)
   if resume_path is None:raise ValueError('Pause requires resume_path')
   temp=Path(str(resume_path)+'.tmp');torch.save(checkpoint,temp);temp.replace(resume_path);raise TrainingPaused('Saved exact epoch checkpoint: '+str(resume_path))
 if stopper is not None:stopper.restore(model)
 if resume_path and Path(resume_path).exists():Path(resume_path).unlink()
 metadata=dict(epochs_run=epoch,selected_epoch=stopper.best_epoch if stopper else epoch,validation_mae_best=stopper.best_value if stopper else None,actual_optimizer=dict(learning_rate=config['learning_rate'],weight_decay=config['weight_decay']),dropout=config['dropout'],scheduler_t_max=600,execution_backend='sync-reduced; independent process RNG')
 return model.eval(),history,metadata
