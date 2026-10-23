# Numerical-embedding ResMLP and the official TabM implementation.
from pathlib import Path
import torch
from torch import nn
import rtdl_num_embeddings as embeddings
from tabm import TabM
CONFIGS={
 'res_periodic128':dict(family='resmlp',embedding='periodic',d_embedding=8,width=128,blocks=2,dropout=0.,lr=.001,wd=.001),
 'res_ple128':dict(family='resmlp',embedding='ple',d_embedding=8,width=128,blocks=2,dropout=.05,lr=.001,wd=.001),
 'res_periodic256':dict(family='resmlp',embedding='periodic',d_embedding=16,width=256,blocks=3,dropout=.05,lr=.0003,wd=.01),
 'tabm_raw128_k8':dict(family='tabm',embedding='none',d_embedding=8,width=128,blocks=2,dropout=.1,k=8,lr=.001,wd=.001),
 'tabm_ple128_k16':dict(family='tabm',embedding='ple',d_embedding=8,width=128,blocks=2,dropout=.1,k=16,lr=.001,wd=.001),
 'tabm_periodic256_k16':dict(family='tabm',embedding='periodic',d_embedding=16,width=256,blocks=2,dropout=.1,k=16,lr=.0003,wd=.01),
}
class ResidualBlock(nn.Module):
 def __init__(self,width,dropout):
  super().__init__(); self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.SiLU(),nn.Dropout(dropout),nn.Linear(width*2,width))
 def forward(self,x):return x+self.net(x)
class Model(nn.Module):
 def __init__(self,config,stats):
  super().__init__(); self.config=dict(config); self.stats=stats
  for key in ('x_mean','x_scale','y_mean','y_scale'):
   self.register_buffer(key,torch.tensor(stats[key],dtype=torch.float32))
  if config['embedding']=='periodic':
   emb=embeddings.PeriodicEmbeddings(21,d_embedding=config['d_embedding'],n_frequencies=8,frequency_init_scale=.1,lite=False)
  elif config['embedding']=='ple':
   emb=embeddings.PiecewiseLinearEmbeddings([torch.tensor(b) for b in stats['bins']],config['d_embedding'],activation=True,version='B')
  else:emb=None
  if config['family']=='tabm':
   self.core=TabM.make(n_num_features=21,d_out=1,num_embeddings=emb,n_blocks=config['blocks'],d_block=config['width'],dropout=config['dropout'],k=config['k'],arch_type='tabm')
  else:
   self.embedding=emb
   self.core=nn.Sequential(nn.Linear(21*config['d_embedding'],config['width']),*[ResidualBlock(config['width'],config['dropout']) for _ in range(config['blocks'])],nn.LayerNorm(config['width']),nn.SiLU(),nn.Linear(config['width'],1))
 def normalized_heads(self,w):
  x=(w-self.x_mean)/self.x_scale
  if self.config['family']=='tabm':return self.core(x).squeeze(-1)
  return self.core(self.embedding(x).flatten(1))
 def forward(self,w):return self.y_mean+self.y_scale*self.normalized_heads(w).mean(1)
def fit_stats(X,y):
 x=torch.tensor(X,dtype=torch.float64); y=torch.tensor(y,dtype=torch.float64)
 mean=x.mean(0); scale=x.std(0,unbiased=False).clamp_min(1e-6)
 bins=embeddings.compute_bins(((x-mean)/scale).float(),n_bins=16)
 return dict(x_mean=mean.tolist(),x_scale=scale.tolist(),y_mean=float(y.mean()),y_scale=float(y.std(unbiased=False).clamp_min(1e-6)),bins=[b.tolist() for b in bins])
def save(model,path):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 torch.save(dict(version=1,config=model.config,stats=model.stats,state={k:v.detach().cpu() for k,v in model.state_dict().items()}),path)
def load(path,device='cpu'):
 obj=torch.load(path,map_location='cpu',weights_only=True)
 if obj['version']!=1:raise ValueError('Unsupported checkpoint')
 m=Model(obj['config'],obj['stats']); m.load_state_dict(obj['state']); return m.to(device).eval()
