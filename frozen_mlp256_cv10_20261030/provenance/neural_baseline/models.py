"""Raw-21-input regression MLPs; checkpoints include frozen preprocessing."""
from pathlib import Path
import torch
from torch import nn
CONFIGS={
 "mlp64_relu":dict(hidden=[64,32],activation="relu",dropout=0.,weight_decay=.001),
 "mlp128_relu":dict(hidden=[128,64],activation="relu",dropout=0.,weight_decay=.001),
 "mlp128_tanh":dict(hidden=[128,64],activation="tanh",dropout=0.,weight_decay=.001),
 "mlp256_deep":dict(hidden=[256,128,64],activation="relu",dropout=.1,weight_decay=.01),
}
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__(); self.config=dict(config); dims=[21,*config["hidden"],1]; layers=[]
        for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
            linear=nn.Linear(a,b); nn.init.xavier_uniform_(linear.weight); nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i<len(dims)-2:
                layers.append(nn.ReLU() if config["activation"]=="relu" else nn.Tanh())
                if config["dropout"]: layers.append(nn.Dropout(config["dropout"]))
        self.net=nn.Sequential(*layers)
        self.register_buffer("x_mean",torch.zeros(21)); self.register_buffer("x_scale",torch.ones(21))
        self.register_buffer("y_mean",torch.zeros(())); self.register_buffer("y_scale",torch.ones(()))
    def forward(self,w):
        if w.ndim!=2 or w.shape[1]!=21: raise ValueError("Expected (batch,21) weights")
        return self.y_mean+self.y_scale*self.net((w-self.x_mean)/self.x_scale).squeeze(-1)
    def fit_stats(self,X,y):
        X=torch.as_tensor(X,dtype=torch.float64); y=torch.as_tensor(y,dtype=torch.float64)
        with torch.no_grad():
            self.x_mean.copy_(X.mean(0)); self.x_scale.copy_(X.std(0,unbiased=False).clamp_min(1e-6))
            self.y_mean.copy_(y.mean()); self.y_scale.copy_(y.std(unbiased=False).clamp_min(1e-6))
        return self

def save(model,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(version=1,config=model.config,state={k:v.detach().cpu() for k,v in model.state_dict().items()}),path)

def load(path,device="cpu"):
    data=torch.load(path,map_location="cpu",weights_only=True)
    if data["version"]!=1: raise ValueError("Unsupported checkpoint")
    model=MLP(data["config"]); model.load_state_dict(data["state"])
    return model.to(device).eval()
