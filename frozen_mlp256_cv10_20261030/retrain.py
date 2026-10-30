# Explicitly invoked only; no parameter search, early stopping or test-driven selection.
import argparse,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=range(1,11));parser.add_argument('--all',action='store_true');parser.add_argument('--device',choices=['cuda','cpu'],required=True);parser.add_argument('--gpu',default='0');parser.add_argument('--out',required=True);args=parser.parse_args()
 if (args.fold is None)==(not args.all):parser.error('Choose exactly one of --fold N or --all')
 if ',' in args.gpu:parser.error('One GPU only')
 if os.environ.get('PYTHONHASHSEED')!='20261029':
  os.environ['PYTHONHASHSEED']='20261029';os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
 os.environ['CUDA_VISIBLE_DEVICES']=args.gpu if args.device=='cuda' else '';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 # Disable unrelated inherited MPS endpoints on the teacher computer.
 os.environ.pop('CUDA_MPS_PIPE_DIRECTORY',None);os.environ.pop('CUDA_MPS_LOG_DIRECTORY',None)
 import json,random,platform
 import numpy as np,torch
 from frozen_data import load_table
 from training import fit
 from helpers import save_atomic,predict_rows
 from model import load
 import hashlib
 def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
 manifest=json.loads((ROOT/'manifest.json').read_text())
 for path,digest in manifest['files'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Frozen file changed: '+path)
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 if args.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):raise ValueError('Exactly one available CUDA GPU required')
 target=Path(args.out).resolve()
 if target==ROOT or ROOT.is_relative_to(target):raise ValueError('Output must not overwrite the frozen package')
 target.mkdir(parents=True,exist_ok=True);ids,X,y=load_table(ROOT/'train.csv');recipes=json.loads((ROOT/'recipes.json').read_text());reports=[]
 for recipe in recipes:
  if args.fold is not None and recipe['fold']!=args.fold:continue
  folder=target/f"fold{recipe['fold']:02d}"
  if folder.exists():raise ValueError('Output fold already exists; refusing overwrite')
  folder.mkdir();random.seed(recipe['python_seed']);a=np.asarray(recipe['train_indices']);b=np.asarray(recipe['test_indices'])
  model,history,metadata=fit(X[a],y[a],recipe['config'],recipe['init_seed'],recipe['shuffle_seed'],args.device,epochs=recipe['epochs'])
  save_atomic(model,folder/'model.pt');del model;restored=load(folder/'model.pt');reference=load(ROOT/recipe['reference_checkpoint']);pred=predict_rows(restored,X[b]).astype(np.float64);reference_pred=predict_rows(reference,X[b]).astype(np.float64)
  exact=all(torch.equal(v,reference.state_dict()[k]) for k,v in restored.state_dict().items())
  comparison=dict(fold=recipe['fold'],trained_epochs=recipe['epochs'],test_mae=float(np.abs(pred-y[b]).mean()),reference_test_mae=recipe['reference_test_mae'],state_dict_bitwise_equal=exact,max_test_prediction_difference=float(np.max(np.abs(pred-reference_pred))),prediction_close_atol_1e_3=bool(np.allclose(pred,reference_pred,atol=1e-3,rtol=0)),device=args.device,torch=torch.__version__,python=platform.python_version(),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None)
  (folder/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n');(folder/'history.json').write_text(json.dumps(history,indent=2)+'\n');np.save(folder/'test_predictions.npy',pred);reports.append(comparison);print(json.dumps(comparison),flush=True)
 (target/'retraining_summary.json').write_text(json.dumps(reports,indent=2)+'\n')
if __name__=='__main__':main()
