"""Refit cross-validation-selected neural models on all rows; no new test claims."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from Lab6 import baseline
from neural.train import fit_network,write_json
from neural.models import save_surrogate,load_surrogate
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument("run"); a=p.parse_args(); out=Path(a.run)
    status=json.loads((out/"status.json").read_text()); assert status["state"]=="complete"
    manifest=json.loads((out/"manifest.json").read_text()); args=SimpleNamespace(**manifest["config"])
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
    assert visible==manifest["gpu_uuid"] and visible.startswith("GPU-") and "," not in visible
    import subprocess
    idle=subprocess.check_output(["nvidia-smi","-i",visible,"--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    mem,util=[int(v.strip()) for v in idle.split(",")]; assert mem<=64 and util==0
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.05)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    rows=[r for r in json.loads((out/"comparison.json").read_text()) if r["model"] in args.names]
    by_mae=min(rows,key=lambda r:r["ensemble_mae"])
    by_decision=min(rows,key=lambda r:(r["ensemble_regret_at2"],r["ensemble_mae"]))
    names=sorted({by_mae["model"],by_decision["model"]})
    ids,X,y=baseline.load_table(ROOT/"Lab6/dataset_same8/train.csv"); train=np.arange(len(y))
    quad=None
    if "quadratic_residual" in names:
        base,meta=baseline.train_model(X,y); quad=baseline.quadratic_form(base)
    final=out/"final_models"; final.mkdir(exist_ok=True); models=[]
    for name in names:
        for seed in args.seeds:
            histories=[json.loads((out/f"fold{f}"/f"{name}_{seed}.json").read_text()) for f in range(args.folds)]
            epochs=int(np.median([h["inner"]["best_epoch"] for h in histories]))
            model,info=fit_network(name,X,y,train,None,quad if name=="quadratic_residual" else None,seed,args,fixed_epochs=epochs)
            path=final/f"{name}_{seed}.pt"; save_surrogate(model,path)
            loaded=load_surrogate(path,args.device)
            tx=torch.as_tensor(X,dtype=torch.float32,device=args.device)
            with torch.no_grad():
                pred=model(tx).cpu().numpy(); assert np.array_equal(pred,loaded(tx).cpu().numpy())
            models.append(dict(name=name,seed=seed,epochs=epochs,path=path.name,
                               training_mae=float(np.abs(y-pred).mean()),seconds=info["seconds"]))
            print(json.dumps(models[-1]),flush=True)
    write_json(final/"selection.json",dict(selected_by_mae=by_mae["model"],
        selected_by_regret_at2=by_decision["model"],models=models,gpu_uuid=visible,
        note="Fit on all 2048 rows, epochs from median inner-CV best epoch per seed. Training MAE is not validation accuracy. Use comparison.csv for OOF results."))
if __name__=="__main__": main()
