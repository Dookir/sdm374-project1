from pathlib import Path
import os,numpy as np,torch
from model import save

def save_atomic(model,path):
 path=Path(path);tmp=path.with_suffix('.pt.tmp');save(model,tmp);os.replace(tmp,path)

def predict_rows(model,X,batch_size=256):
 model.eval()
 with torch.no_grad():return np.concatenate([model(torch.tensor(part,dtype=torch.float32)).numpy() for part in np.array_split(X,max(1,(len(X)+batch_size-1)//batch_size))])
