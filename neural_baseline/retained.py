# Predict each retained network with its own checkpoint, on CPU.
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import torch
from neural_baseline.models import load
BASE=Path(__file__).resolve().parent/'retained'
def main():
    p=argparse.ArgumentParser(description='Predict using the five retained networks')
    p.add_argument('--name'); p.add_argument('--weights',type=Path); p.add_argument('--verify',action='store_true')
    a=p.parse_args(); torch.set_num_threads(2)
    manifest=json.loads((BASE/'manifest.json').read_text()); found=[]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for item in manifest['models']:
        if a.name and a.name!=item['name']:continue
        folder=BASE/item['name']; path=a.weights or folder/'weights.json'
        w=np.asarray(json.loads(path.read_text()),dtype=float)
        if w.shape!=(3,7) or not np.isfinite(w).all() or (w<0).any() or not np.allclose(w.sum(1),1,atol=1e-12,rtol=0):raise ValueError('Expected finite nonnegative 3x7 rows summing to one')
        if a.verify:
            if sha(folder/'weights.json')!=item['weights_sha256']:raise ValueError('Weight hash mismatch')
            for m in item['models']:
                if sha(folder/m['file'])!=m['sha256']:raise ValueError('Checkpoint hash mismatch')
        with torch.no_grad():value=float(np.mean([float(load(folder/m['file'])(torch.tensor(w.reshape(1,21),dtype=torch.float32))) for m in item['models']]))
        if a.verify and not a.weights and abs(value-item['predicted_ramping'])>1e-5:raise ValueError('Prediction mismatch')
        found.append(dict(name=item['name'],predicted_ramping=value,weights=str(path)))
    if not found:p.error('Unknown network name')
    if a.verify:
        root=BASE.parents[1]
        for path,expected in manifest['teacher_files_sha256'].items():
            if sha(root/path)!=expected:raise ValueError('Original teacher file changed: '+path)
    print(json.dumps(dict(verified=a.verify,models=found),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
