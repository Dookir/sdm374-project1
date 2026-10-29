# This function accepts training rows only; held-out data cannot affect fitting.
from pathlib import Path
import os
import numpy as np,torch
from neural_baseline.models import MLP,save,load
from cv_experiment.common import TRAIN

def train_fold(X_train,y_train,config,init_seed,shuffle_seed,device,epochs=TRAIN['epochs'],batch_size=TRAIN['batch_size'],callback=None):
 torch.manual_seed(init_seed);np.random.seed(init_seed)
 model=MLP(config).fit_stats(np.array(X_train,copy=True),np.array(y_train,copy=True)).to(device)
 x=torch.tensor(np.array(X_train,copy=True),dtype=torch.float32,device=device);y=torch.tensor(np.array(y_train,copy=True),dtype=torch.float32,device=device)
 generator=torch.Generator(device=device);generator.manual_seed(shuffle_seed)
 optimizer=torch.optim.AdamW(model.parameters(),lr=TRAIN['learning_rate'],weight_decay=TRAIN['weight_decay'],fused=str(device).startswith('cuda'))
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=TRAIN['eta_min']);history=[]
 for epoch in range(1,epochs+1):
  model.train()
  for idx in torch.randperm(len(x),device=device,generator=generator).split(batch_size):
   optimizer.zero_grad(set_to_none=True);loss=(((model(x[idx])-y[idx])/model.y_scale)**2).mean()
   if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),TRAIN['gradient_clip']);optimizer.step()
  scheduler.step()
  if epoch%25==0 or epoch==epochs:
   model.eval()
   with torch.no_grad():mse=float(((model(x)-y)**2).mean())
   history.append(dict(epoch=epoch,training_mse_diagnostic=mse,learning_rate=float(scheduler.get_last_lr()[0])))
   if callback:callback(epoch,history)
 return model.eval(),history

def save_atomic(model,path):
 path=Path(path);tmp=path.with_suffix('.pt.tmp');save(model,tmp);os.replace(tmp,path)

def predict_rows(model,X,batch_size=256):
 model.eval()
 with torch.no_grad():return np.concatenate([model(torch.tensor(part,dtype=torch.float32)).numpy() for part in np.array_split(X,max(1,(len(X)+batch_size-1)//batch_size))])
