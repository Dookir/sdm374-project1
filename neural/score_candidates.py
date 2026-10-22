"""Generate one candidate per neural family, then call the unchanged teacher CLI."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse,hashlib,json,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from Lab6 import baseline
from neural.models import CONFIGS,load_surrogate,save_surrogate
from neural.train import fit_network,write_json
ROOT=Path(__file__).resolve().parents[1]

def project(x):
    rows=x.reshape(-1,7); ordered=rows.sort(dim=1,descending=True).values
    sums=ordered @ torch.ones((7,7),device=x.device,dtype=x.dtype).triu()-1
    counts=torch.arange(1,8,device=x.device)
    last=(ordered-sums/counts>0).sum(1)-1
    theta=sums.gather(1,last[:,None]).squeeze(1)/(last+1)
    return (rows-theta[:,None]).clamp_min(0).reshape_as(x)

def optimize(models,starts,steps=1000):
    w=torch.tensor(starts,dtype=torch.float32,device="cuda",requires_grad=True)
    for model in models:
        model.eval()
        for param in model.parameters(): param.requires_grad_(False)
    opt=torch.optim.Adam([w],lr=.01)
    best=float("inf"); chosen=None; last_change=0; history=[]
    for step in range(steps+1):
        value=torch.stack([m(w) for m in models]).mean(0)
        v,index=value.detach().min(0); v=float(v)
        if v<best:
            if best-v>1e-5: last_change=step
            best=v; chosen=w[int(index)].detach().clone()
        if step%100==0: history.append(dict(step=step,best_own_prediction=best))
        if step==steps or step-last_change>=200: break
        opt.zero_grad(set_to_none=True); value.sum().backward(); opt.step()
        with torch.no_grad(): w.copy_(project(w))
    weights=baseline.project_weights(chosen.cpu().double().numpy()).reshape(3,7)
    with torch.no_grad():
        t=torch.tensor(weights.reshape(1,21),dtype=torch.float32,device="cuda")
        individual=[float(m(t).item()) for m in models]
    return weights,dict(iterations=step,own_prediction=float(np.mean(individual)),
                        seed_prediction_std=float(np.std(individual)),history=history)

def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument("--run",default="neural/runs/first_training_20261022")
    p.add_argument("--out",default="neural/runs/teacher_scoring_20261022"); a=p.parse_args()
    run=Path(a.run); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
    assert visible.startswith("GPU-") and "," not in visible
    idle=subprocess.check_output(["nvidia-smi","-i",visible,"--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    mem,util=[int(v.strip()) for v in idle.split(",")]; assert mem<=64 and util==0
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.05)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    manifest=json.loads((run/"manifest.json").read_text()); args=SimpleNamespace(**manifest["config"])
    ids,X,y=baseline.load_table(ROOT/"Lab6/dataset_same8/train.csv")
    starts=np.concatenate([baseline.search_starts(X,y,32),np.array(json.loads((ROOT/"Lab6/weights.json").read_text())).reshape(1,21)])
    np.save(out/"starts.npy",starts)
    protected={f:hashlib.sha256((ROOT/"Lab6"/f).read_bytes()).hexdigest() for f in ("baseline.py","model.npz","weights.json")}
    write_json(out/"manifest.json",dict(gpu_uuid=visible,cuda_device_count=1,training_run=str(run),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protected_files=protected,
        protocol="Three seeds refit all rows for median inner-CV epochs. Same 33 starts for all neural families; Adam lr=.01, projected each step, at most 1000 steps, stop after 200 steps without >1e-5 best improvement. Candidate selected ONLY using its own neural ensemble; teacher CLI used afterward for comparison.",
        score_meaning="Teacher quadratic surrogate prediction, not a real simulator or official grade."))
    quad=None; rows=[]; started=time.monotonic()
    def score(name,weights,extra):
        path=out/f"weights_{name}.json"; write_json(path,np.asarray(weights).reshape(3,7).tolist())
        command=[str(Path(os.sys.executable)),"Lab6/baseline.py","predict","--weights",str(path)]
        raw=subprocess.check_output(command,text=True,cwd=ROOT)
        teacher=float(json.loads(raw)["predicted_ramping"])
        direct=float(baseline.predict(weights)); assert abs(teacher-direct)<1e-10
        row=dict(name=name,teacher_prediction=teacher,real_simulator_score=None,weights_file=path.name,**extra)
        rows.append(row); write_json(out/"comparison.json",rows); pd.DataFrame(rows).to_csv(out/"comparison.csv",index=False)
        write_json(out/f"teacher_stdout_{name}.json",dict(command=command,stdout=json.loads(raw)))
        print(json.dumps(row),flush=True)
    score("original_baseline",json.loads((ROOT/"Lab6/weights.json").read_text()),{})
    score("previous_catboost_conservative",json.loads((ROOT/"offline/results/weights_conservative.json").read_text()),dict(origin="Historical CatBoost/HistGB search; not rerun"))
    score("measured_best",X[y.argmin()].reshape(3,7),dict(measured_ramping=float(y.min()),measured_id=ids[int(y.argmin())]))
    modeldir=out/"full_models"; modeldir.mkdir(exist_ok=True)
    for name in CONFIGS:
        models=[]; metadata=[]
        if name=="quadratic_residual" and quad is None:
            base,_=baseline.train_model(X,y); quad=baseline.quadratic_form(base)
        for seed in args.seeds:
            existing=run/"final_models"/f"{name}_{seed}.pt"
            checkpoint=modeldir/f"{name}_{seed}.pt"
            epochs=int(np.median([json.loads((run/f"fold{f}"/f"{name}_{seed}.json").read_text())["inner"]["best_epoch"] for f in range(args.folds)]))
            if existing.exists():
                model=load_surrogate(existing,"cuda"); origin=str(existing)
            elif checkpoint.exists():
                model=load_surrogate(checkpoint,"cuda"); origin=str(checkpoint)
            else:
                model,info=fit_network(name,X,y,np.arange(len(y)),None,quad if name=="quadratic_residual" else None,seed,args,fixed_epochs=epochs)
                save_surrogate(model,checkpoint); origin=str(checkpoint)
                print(f"Refit {name} seed={seed} epochs={epochs} seconds={info['seconds']:.1f}",flush=True)
            models.append(model); metadata.append(dict(seed=seed,epochs=epochs,checkpoint=origin,
                sha256=hashlib.sha256(Path(origin).read_bytes()).hexdigest()))
        weights,info=optimize(models,starts)
        write_json(out/f"search_{name}.json",dict(models=metadata,**info))
        score(name,weights,dict(own_prediction=info["own_prediction"],seed_prediction_std=info["seed_prediction_std"],search_iterations=info["iterations"]))
        del models,model
    for f,h in protected.items(): assert hashlib.sha256((ROOT/"Lab6"/f).read_bytes()).hexdigest()==h
    write_json(out/"verification.json",dict(teacher_files_unchanged=True,all_weights_valid=True,
        CLI_matches_function=True,count=len(rows),elapsed_seconds=time.monotonic()-started,
        cuda_device_count=1,gpu_uuid=visible))
if __name__=="__main__": main()
