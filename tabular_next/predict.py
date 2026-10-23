from pathlib import Path
import argparse,json
import numpy as np,torch
from tabular_next.models import load
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--weights',type=Path,required=True); a=p.parse_args()
 w=np.asarray(json.loads(a.weights.read_text()),dtype=float)
 if w.shape!=(3,7) or not np.isfinite(w).all() or (w<0).any() or not np.allclose(w.sum(1),1,atol=1e-12,rtol=0):raise ValueError('Expected legal 3x7 weights')
 torch.set_num_threads(2); m=load(a.checkpoint)
 with torch.no_grad():v=float(m(torch.tensor(w.reshape(1,21),dtype=torch.float32)))
 print(json.dumps(dict(predicted_ramping=v)))
if __name__=='__main__':main()
