# Read-only single snapshot, or final CPU audit; never poll.
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--out',default='tabular_next/results/run_20261023'); p.add_argument('--audit',action='store_true'); a=p.parse_args(); out=ROOT/a.out
 for name in ['failure.json','complete.json','status.json']:
  if (out/name).exists():print((out/name).read_text()); break
 if not a.audit:return
 import torch
 from tabular_next.models import load
 torch.set_num_threads(2); manifest=json.loads((out/'manifest.json').read_text()); rows=json.loads((out/'comparison.json').read_text()); complete=json.loads((out/'complete.json').read_text())
 assert len(rows)==manifest['expected_candidates']==72
 for row in rows:
  w=np.asarray(json.loads((ROOT/row['weights']).read_text())); assert w.shape==(3,7) and np.isfinite(w).all() and (w>=0).all() and np.allclose(w.sum(1),1,atol=1e-12,rtol=0)
 for family,row in complete['best_by_family'].items():
  folder=out/'best'/family
  for name,key in [('model.pt','model_sha256'),('weights.json','weights_sha256')]:assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==row[key]
  model=load(folder/'model.pt'); w=torch.tensor(json.loads((folder/'weights.json').read_text()),dtype=torch.float32).reshape(1,21)
  with torch.no_grad():value=float(model(w))
  assert abs(value-row['predicted_ramping'])<.002,(value,row['predicted_ramping'])
 for path,digest in (manifest['protected_files']|manifest['source_sha256']).items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
 report=dict(passed=True,legal_candidates=len(rows),best_models_reloaded_on_cpu=True,source_and_protected_hashes_match=True,cuda_device_count=complete['cuda_device_count'])
 (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
if __name__=='__main__':main()
