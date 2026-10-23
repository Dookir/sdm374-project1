"""Train networks -> project-gradient weight search -> unchanged teacher predict."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from pathlib import Path
import argparse,fcntl,hashlib,json,subprocess,sys,time
import numpy as np
import pandas as pd
import torch
from Lab6 import baseline
from neural_baseline.models import MLP,save,load
from neural_baseline.expanded_configs import CONFIGS
ROOT=Path(__file__).resolve().parents[1]

def dump(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n"); tmp.replace(path)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def project(w):
    rows=w.reshape(-1,7); s=rows.sort(1,descending=True).values
    # Fixed seven-element prefix sums avoid CUDA cumsum nondeterminism.
    prefix=s @ torch.ones((7,7),dtype=w.dtype,device=w.device).triu()-1
    n=torch.arange(1,8,device=w.device); rho=(s-prefix/n>0).sum(1)-1
    theta=prefix.gather(1,rho[:,None]).squeeze(1)/(rho+1)
    return (rows-theta[:,None]).clamp_min(0).reshape_as(w)

def search(models,starts,iterations=900):
    for model in models:
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
    # Same starts, three predeclared constant step sizes; no teacher score inside search.
    w=torch.tensor(np.tile(starts,(3,1)),dtype=torch.float32,device="cuda",requires_grad=True)
    steps=torch.tensor(np.repeat([.0001,.001,.01],len(starts)),dtype=w.dtype,device=w.device)[:,None]
    best=float("inf"); chosen=None; last_improvement=0; history=[]
    for i in range(iterations+1):
        prediction=torch.stack([m(w) for m in models]).mean(0)
        v,idx=prediction.detach().min(0); value=float(v)
        if not np.isfinite(value): raise ValueError("Nonfinite search prediction")
        if value<best:
            if best-value>1e-5: last_improvement=i
            best=value; chosen=w[int(idx)].detach().clone()
        if i%100==0: history.append(dict(iteration=i,own_prediction=best))
        if i==iterations or i-last_improvement>=200: break
        gradient=torch.autograd.grad(prediction.sum(),w)[0]
        with torch.no_grad(): w.copy_(project(w-steps*gradient))
    weights=baseline.project_weights(chosen.cpu().double().numpy()).reshape(3,7)
    return weights,dict(iterations=i,own_prediction=best,history=history,
                        start_count=len(starts),step_sizes=[.0001,.001,.01])

def run(args):
    out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    lock=(out/"run.lock").open("w"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"complete.json").exists(): raise RuntimeError("Completed run; choose another output directory")
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if not visible.startswith("GPU-") or "," in visible: raise ValueError("Exactly one GPU UUID required")
    state=subprocess.check_output(["nvidia-smi","-i",visible,"--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    mem,util=[int(s.strip()) for s in state.split(",")]
    if mem>64 or util>0: raise RuntimeError("Selected GPU is no longer idle")
    if torch.cuda.device_count()!=1: raise RuntimeError("Exactly one visible GPU required")
    torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.05)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    ids,X,y=baseline.load_table(ROOT/"Lab6/dataset_same8/train.csv")
    original=json.loads((ROOT/"Lab6/weights.json").read_text())
    starts=np.concatenate([baseline.search_starts(X,y,64),np.asarray(original).reshape(1,21)])
    protected={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"Lab6").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    source_files=["neural_baseline/models.py","neural_baseline/expanded_run.py","neural_baseline/expanded_configs.py"]
    manifest=dict(protocol="Train on original CSV ramping -> neural projected-gradient weight search -> unchanged Lab6/baseline.py predict, never a generated scoring model. ONLY teacher_script_score ranks candidates.",
                  configs=CONFIGS,seeds=args.seeds,epochs=args.epochs,checkpoints=args.checkpoints,
                  training=dict(optimizer="AdamW",learning_rate="per configuration",minimum_lr=.00001,batch="per configuration",
                                target="original train.csv ramping",criterion="normalized MSE, diagnostic only",fixed_epoch_schedule=True),
                  gpu_uuid=visible,cuda_device_count=1,protected_files=protected,previous_reference_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"neural_baseline/results/first_run_20261023/best_neural").glob("*") if p.is_file()},expected_scores=2+len(CONFIGS)*(len(args.seeds)+1)*len(args.checkpoints)+3,expanded_search=dict(starts=65,iterations=900,refine_top_configs=3,refine_starts=1089,refine_iterations=2500),
                  source_sha256={p:sha(ROOT/p) for p in source_files},torch_version=str(torch.__version__),
                  base_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,cwd=ROOT).strip())
    manifest_path=out/"manifest.json"
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text()); manifest["base_commit"]=old["base_commit"]
        if old!=manifest: raise ValueError("Resume manifest mismatch")
    else: dump(manifest_path,manifest)
    rows=json.loads((out/"comparison.json").read_text()) if (out/"comparison.json").exists() else []
    done={r["name"] for r in rows}; started=time.monotonic()
    def score(name,weights,metadata):
        if name in done: return
        path=out/"weights"/(name+".json"); dump(path,np.asarray(weights).reshape(3,7).tolist())
        # This command is the sole final scorer; deliberately no --model argument.
        command=[sys.executable,str(ROOT/"Lab6/baseline.py"),"predict","--weights",str(path)]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
        value=float(json.loads(result.stdout)["predicted_ramping"])
        dump(out/"commands"/(name+".json"),dict(argv=command,stdout=result.stdout,stderr=result.stderr))
        row=dict(name=name,teacher_script_score=value,weights=str(path.relative_to(ROOT)),**metadata)
        rows.append(row); done.add(name); dump(out/"comparison.json",rows)
        pd.DataFrame([dict(name=r["name"],teacher_script_score=r["teacher_script_score"],weights=r["weights"]) for r in rows]).to_csv(out/"comparison.csv",index=False)
        dump(out/"status.json",dict(state="running",scored=len(rows),elapsed_seconds=time.monotonic()-started,gpu_uuid=visible))
        print(f"SCORE {len(rows):02d}: {name} = {value:.9f}",flush=True)
    score("original_baseline",original,dict(kind="baseline"))
    previous=ROOT/"neural_baseline/results/first_run_20261023/best_neural"
    prev=json.loads((previous/"selection.json").read_text())
    previous_checkpoints=[str((previous/m["file"]).relative_to(ROOT)) for m in prev["models"]]
    score("previous_best_neural",json.loads((previous/"weights.json").read_text()),
          dict(kind="previous_neural",architecture="previous_mlp128_relu",epoch=prev["epoch"],
               checkpoints=previous_checkpoints,origin=prev["trial"]))
    tx=torch.tensor(X,dtype=torch.float32,device="cuda"); ty=torch.tensor(y,dtype=torch.float32,device="cuda")
    for name,config in CONFIGS.items():
        for seed in args.seeds:
            trial=out/"models"/f"{name}_{seed}"; trial.mkdir(parents=True,exist_ok=True)
            if all((trial/f"epoch{e}.pt").exists() and f"{name}_{seed}_e{e}" in done for e in args.checkpoints): continue
            torch.manual_seed(seed); np.random.seed(seed)
            model=MLP(config).fit_stats(X.copy(),y.copy()).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"],fused=True)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=.00001)
            history=[]; beginning=time.monotonic()
            for epoch in range(1,args.epochs+1):
                model.train(); order=torch.randperm(len(tx),device="cuda")
                for idx in order.split(config["batch_size"]):
                    optimizer.zero_grad(set_to_none=True)
                    loss=(((model(tx[idx])-ty[idx])/model.y_scale)**2).mean()
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                    optimizer.step()
                scheduler.step()
                if epoch%10==0 or epoch in args.checkpoints:
                    model.eval()
                    with torch.no_grad(): mse=float(((model(tx)-ty)**2).mean())
                    if not np.isfinite(mse): raise ValueError("Nonfinite training MSE")
                    history.append(dict(epoch=epoch,training_mse=mse,lr=float(scheduler.get_last_lr()[0])))
                    dump(trial/"history.json",history)
                if epoch in args.checkpoints:
                    checkpoint=trial/f"epoch{epoch}.pt"; save(model,checkpoint)
                    restored=load(checkpoint,"cuda")
                    with torch.no_grad(): assert torch.equal(model.eval()(tx[:64]),restored(tx[:64]))
                    weights,info=search([restored],starts)
                    dump(trial/f"search_epoch{epoch}.json",info)
                    score(f"{name}_{seed}_e{epoch}",weights,dict(kind="single",architecture=name,seed=seed,epoch=epoch,
                          checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_exact=True))
                    del restored
                    print(f"TRAIN {name} seed={seed} epoch={epoch} training_MSE={mse:.5f} seconds={time.monotonic()-beginning:.1f}",flush=True)
            del model,optimizer,scheduler
        for epoch in args.checkpoints:
            label=f"{name}_ensemble_e{epoch}"
            if label in done: continue
            checkpoints=[out/"models"/f"{name}_{s}"/f"epoch{epoch}.pt" for s in args.seeds]
            models=[load(path,"cuda") for path in checkpoints]
            weights,info=search(models,starts); dump(out/"ensemble_search"/(label+".json"),info)
            score(label,weights,dict(kind="ensemble",architecture=name,epoch=epoch,
                  seeds=args.seeds,checkpoints=[str(p.relative_to(ROOT)) for p in checkpoints]))
            del models
    # Select by native score only, then widen input search for three distinct configurations.
    candidates=sorted((r for r in rows if r["kind"] not in ("baseline","refined")),key=lambda r:r["teacher_script_score"])
    selected=[]; seen=set()
    for r in candidates:
        if r["architecture"] not in seen:
            selected.append(r); seen.add(r["architecture"])
        if len(selected)==3: break
    rng=np.random.default_rng(20261023)
    additional=np.concatenate([rng.dirichlet(np.full(7,a),size=(256,3)).reshape(256,21) for a in (.1,.5,1.,5.)])
    for row in selected:
        label="refined_"+row["name"]
        if label in done: continue
        paths=[row["checkpoint"]] if row["kind"]=="single" else row["checkpoints"]
        models=[load(ROOT/p,"cuda") for p in paths]
        extra_starts=np.concatenate([starts,additional])
        extra_starts[0]=np.asarray(json.loads((ROOT/row["weights"]).read_text())).reshape(21)
        weights,info=search(models,extra_starts,iterations=2500)
        dump(out/"refined_search"/(label+".json"),info)
        score(label,weights,dict(kind="refined",architecture=row["architecture"],epoch=row["epoch"],
              checkpoints=paths,parent=row["name"],parent_score=row["teacher_script_score"]))
        del models
    winner=rows[0]
    for row in rows[1:]:
        if row["teacher_script_score"]<winner["teacher_script_score"]-1e-8: winner=row
    dump(out/"weights_recommended.json",json.loads((ROOT/winner["weights"]).read_text()))
    best_neural=min((r for r in rows if r["kind"]!="baseline"),key=lambda r:r["teacher_script_score"])
    for p,h in protected.items(): assert sha(ROOT/p)==h,p
    dump(out/"complete.json",dict(state="complete",scored=len(rows),training_trials=len(CONFIGS)*len(args.seeds),
         selected=winner["name"],selected_score=winner["teacher_script_score"],
         baseline_score=rows[0]["teacher_script_score"],best_neural=best_neural,
         improvement=rows[0]["teacher_script_score"]-winner["teacher_script_score"],
         best_new_network=min((r for r in rows if r["kind"] in ("single","ensemble")),key=lambda r:r["teacher_script_score"]),
         teacher_files_unchanged=True,cuda_device_count=1,gpu_uuid=visible,
         peak_tensor_memory_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started))
    dump(out/"status.json",json.loads((out/"complete.json").read_text()))
    print("COMPLETE",winner["name"],winner["teacher_script_score"],flush=True)

def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument("--out",default="neural_baseline/results/expanded_run_20261023")
    p.add_argument("--seeds",type=int,nargs="+",default=[2026,2027,2028])
    p.add_argument("--epochs",type=int,default=1200)
    p.add_argument("--checkpoints",type=int,nargs="+",default=[50,100,300,1200])
    a=p.parse_args()
    if sorted(set(a.checkpoints))!=a.checkpoints or a.checkpoints[-1]!=a.epochs: raise ValueError("Ordered unique checkpoints must end at --epochs")
    run(a)
if __name__=="__main__": main()
