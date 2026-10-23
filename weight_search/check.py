from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--audit',action='store_true'); a=p.parse_args(); out=ROOT/'weight_search/results/run_20261023'
 for filename in ['failure.json','complete.json','status.json']:
  if (out/filename).exists():
   obj=json.loads((out/filename).read_text())
   if filename=='complete.json':print(json.dumps(dict(state='complete',completed_models=obj['completed_models'],comparison=[dict(name=r['name'],before=r['before'],after=r['after'],improvement=r['improvement'],negative_prediction=r['negative_prediction']) for r in obj['results']]),indent=2))
   else:print(json.dumps(obj,indent=2))
   break
 if not a.audit:return
 import numpy as np,torch
 from weight_search.run import getmodel
 torch.set_num_threads(2); manifest=json.loads((out/'manifest.json').read_text()); complete=json.loads((out/'complete.json').read_text()); lookup={r['name']:r for r in manifest['records']}
 assert complete['completed_models']==len(lookup)==11
 for path,digest in (manifest['protected_files']|manifest['source_sha256']).items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
 for row in complete['results']:
  w=np.array(json.loads((ROOT/row['weights']).read_text())); assert w.shape==(3,7) and np.isfinite(w).all() and (w>=0).all() and np.allclose(w.sum(1),1,atol=1e-12,rtol=0)
  model=getmodel(lookup[row['name']])
  with torch.no_grad():value=float(model(torch.tensor(w.reshape(1,21),dtype=torch.float32)))
  assert abs(value-row['after'])<1e-5; assert row['after']<=row['before']+1e-6
 report=dict(passed=True,models_verified=len(lookup),all_weights_legal=True,predictions_reproduced=True,no_score_regressions=True,protected_files_and_sources_unchanged=True)
 (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
if __name__=='__main__':main()
