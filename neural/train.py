"""Sequential, one-GPU, nested early-stopping comparison of six networks."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, fcntl, hashlib, json, signal, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostRegressor
from Lab6 import baseline
from neural.models import CONFIGS, make_surrogate, save_surrogate, load_surrogate
from neural.training_metrics import metrics, oof_metrics
ROOT=Path(__file__).resolve().parents[1]

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)

def make_splits(n,folds=5):
    result=[]
    for f,(train,test) in enumerate(KFold(folds,shuffle=True,random_state=20261023).split(np.arange(n))):
        fit,stop=train_test_split(train,test_size=.2,random_state=20261024+f)
        assert not (set(fit)&set(stop) or set(train)&set(test))
        result.append((train,test,fit,stop))
    assert sorted(np.concatenate([s[1] for s in result]).tolist())==list(range(n))
    return result

def fit_network(name,X,y,train,stop,quadratic,seed,args,fixed_epochs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model=make_surrogate(name,X[train],y[train],quadratic=quadratic).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,
                               fused=args.device=="cuda")
    tx=torch.as_tensor(X[train],dtype=torch.float32,device=args.device)
    ty=torch.as_tensor(y[train],dtype=torch.float32,device=args.device)
    if stop is not None:
        vx=torch.as_tensor(X[stop],dtype=torch.float32,device=args.device)
        vy=torch.as_tensor(y[stop],dtype=torch.float32,device=args.device)
    best=float("inf"); best_epoch=0; best_state=None; history=[]; started=time.monotonic()
    max_epochs=args.max_epochs if fixed_epochs is None else fixed_epochs
    for epoch in range(max_epochs+1):
        if stop is not None and (epoch%args.eval_every==0 or epoch==max_epochs):
            model.eval()
            with torch.no_grad():
                val=float((model(vx)-vy).abs().mean().item())
                tr=float((model(tx)-ty).abs().mean().item())
            history.append(dict(epoch=epoch,train_mae=tr,inner_mae=val))
            if val<best-args.min_delta:
                best=val; best_epoch=epoch
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            if epoch-best_epoch>=args.patience: break
        if epoch==max_epochs: break
        model.train(); order=torch.randperm(len(tx),device=args.device)
        for index in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss=(((model(tx[index])-ty[index])/model.target_scale)**2).mean()
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
    if best_state is not None: model.load_state_dict(best_state)
    return model.eval(),dict(best_epoch=best_epoch if stop is not None else max_epochs,
                            best_inner_mae=best if stop is not None else None,
                            epochs_run=epoch,seconds=time.monotonic()-started,history=history)

def references(X,y,split,directory):
    path=directory/"references.npz"
    if path.exists():
        with np.load(path) as z: return {k:z[k] for k in z.files}
    train,test,fit,stop=split
    inner,imeta=baseline.train_model(X[fit],y[fit])
    outer,ometa=baseline.train_model(X[train],y[train])
    cq,aq,Hq=baseline.quadratic_form(inner)
    co,ao,Ho=baseline.quadratic_form(outer)
    cat=CatBoostRegressor(iterations=1600,depth=5,learning_rate=.08,l2_leaf_reg=10,
                         random_strength=.1,random_seed=2026,thread_count=4,
                         task_type="CPU",verbose=False,allow_writing_files=False)
    cat.fit(X[train],y[train])
    data=dict(quadratic_ridge=outer.predict(X[test]),catboost_reference=cat.predict(X[test]),
              inner_c=cq,inner_a=aq,inner_H=Hq,outer_c=co,outer_a=ao,outer_H=Ho)
    np.savez_compressed(path,**data)
    write_json(directory/"references.json",dict(inner_alpha=imeta["alpha"],outer_alpha=ometa["alpha"],
        catboost=cat.get_params(),note="CatBoost fixed historical parameters; no new tuning on outer folds."))
    return data

def summarize(out,y,splits,args):
    fold_ids=np.empty(len(y),dtype=int)
    for f,(_,test,_,_) in enumerate(splits): fold_ids[test]=f
    rows=[]; predictions=dict(y_true=y,fold=fold_ids)
    for name in ["quadratic_ridge","catboost_reference",*args.names]:
        seeds=[0] if name in ("quadratic_ridge","catboost_reference") else args.seeds
        scores=[]; pp=[]; completed=[]
        for seed in seeds:
            p=np.full(len(y),np.nan)
            for f,(_,test,_,_) in enumerate(splits):
                path=out/f"fold{f}"/("references.npz" if seed==0 else f"{name}_{seed}.npz")
                if path.exists():
                    with np.load(path) as z: p[test]=z[name if seed==0 else "prediction"]
            if np.isfinite(p).all():
                scores.append(oof_metrics(y,p,fold_ids)); pp.append(p); completed.append(seed)
                predictions[f"{name}_{seed}"]=p
        if scores:
            avg={key:float(np.mean([s[key] for s in scores])) for key in scores[0]}
            avg.update(model=name,completed_seeds=len(scores),seed_values=completed,
                       mae_seed_std=float(np.std([s["mae"] for s in scores])))
            ep=np.mean(pp,axis=0)
            avg.update({"ensemble_"+k:v for k,v in oof_metrics(y,ep,fold_ids).items()})
            predictions[name+"_ensemble"]=ep; rows.append(avg)
    if rows:
        pd.DataFrame(rows).to_csv(out/"comparison.csv",index=False)
        write_json(out/"comparison.json",rows)
        pd.DataFrame(predictions).to_csv(out/"oof_predictions.csv",index=False)
    return rows

def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    lock=(out/"run.lock").open("w"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    visible=os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.device=="cuda":
        if not visible.startswith("GPU-") or "," in visible: raise ValueError("Set exactly one GPU UUID")
        free=subprocess.check_output(["nvidia-smi","-i",visible,
            "--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
        memory,util=[int(v.strip()) for v in free.strip().split(",")]
        if memory>64 or util>0: raise RuntimeError("Selected card is no longer idle")
        if torch.cuda.device_count()!=1: raise RuntimeError("Exactly one visible GPU required")
        torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.05,0)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    ids,X,y=baseline.load_table(ROOT/"Lab6/dataset_same8/train.csv")
    splits=make_splits(len(y),args.folds)
    config={k:v for k,v in vars(args).items() if k!="out"}
    manifest=dict(config=config,gpu_uuid=visible,data_sha256=hashlib.sha256((ROOT/"Lab6/dataset_same8/train.csv").read_bytes()).hexdigest(),
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                 ("neural/train.py","neural/models.py","neural/training_metrics.py")},
        torch_version=str(torch.__version__),git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        protocol="Outer 5-fold OOF; inner 20% stop selects epoch; refit outer train for selected epochs; fold-specific preprocessing/quadratic. Three seeds. Fixed hyperparameters, exploratory research benchmark, not an untouched final test or simulator.",
        splits=[{key:[ids[i] for i in a] for key,a in zip(("train","test","fit","stop"),s)} for s in splits])
    path=out/"manifest.json"
    if path.exists():
        previous=json.loads(path.read_text())
        # HEAD may advance with documentation/results, but executable sources/config must match.
        manifest["git_commit"]=previous["git_commit"]
        if previous!=manifest: raise ValueError("Resume manifest mismatch")
    else: write_json(path,manifest)
    start=time.monotonic(); completed=0; total=args.folds*len(args.names)*len(args.seeds)
    try:
        for f,split in enumerate(splits):
            train,test,fit,stop=split; directory=out/f"fold{f}"; directory.mkdir(exist_ok=True)
            ref=references(X,y,split,directory)
            for name in args.names:
                for seed in args.seeds:
                    prefix=directory/f"{name}_{seed}"
                    if all(prefix.with_suffix(s).exists() for s in (".json",".npz",".pt")):
                        completed+=1; continue
                    qi=tuple(ref["inner_"+k] for k in ("c","a","H")) if name=="quadratic_residual" else None
                    qo=tuple(ref["outer_"+k] for k in ("c","a","H")) if name=="quadratic_residual" else None
                    _,inner=fit_network(name,X,y,fit,stop,qi,seed,args)
                    model,refit=fit_network(name,X,y,train,None,qo,seed,args,fixed_epochs=inner["best_epoch"])
                    tx=torch.as_tensor(X[test],dtype=torch.float32,device=args.device)
                    with torch.no_grad(): p=model(tx).cpu().numpy()
                    save_surrogate(model,prefix.with_suffix(".pt"))
                    loaded=load_surrogate(prefix.with_suffix(".pt"),args.device)
                    with torch.no_grad():
                        assert np.array_equal(p,loaded(tx).cpu().numpy())
                        tr=float((model(torch.as_tensor(X[train],dtype=torch.float32,device=args.device))-
                                  torch.as_tensor(y[train],dtype=torch.float32,device=args.device)).abs().mean())
                    score=metrics(y[test],p)
                    np.savez_compressed(prefix.with_suffix(".npz"),row_indices=test,prediction=p)
                    write_json(prefix.with_suffix(".json"),dict(model=name,seed=seed,fold=f,
                        inner=inner,refit=refit,test=score,train_mae=tr,checkpoint_exact=True))
                    completed+=1
                    write_json(out/"status.json",dict(state="running",completed=completed,total=total,
                        elapsed_seconds=time.monotonic()-start,gpu_uuid=visible))
                    print(f"{completed}/{total} fold={f} {name} seed={seed} epoch={inner['best_epoch']} MAE={score['mae']:.4f} regret2={score['regret_at2']:.4f} seconds={inner['seconds']+refit['seconds']:.1f}",flush=True)
            summarize(out,y,splits,args)
        rows=summarize(out,y,splits,args)
        write_json(out/"status.json",dict(state="complete",completed=completed,total=total,
            elapsed_seconds=time.monotonic()-start,gpu_uuid=visible,
            cuda_device_count=torch.cuda.device_count() if args.device=="cuda" else 0,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if args.device=="cuda" else 0))
        print(pd.DataFrame(rows).to_string(index=False),flush=True)
    except BaseException as exc:
        write_json(out/"status.json",dict(state="failed",completed=completed,total=total,error=str(exc)))
        raise

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--out",required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--names",nargs="+",choices=list(CONFIGS),default=list(CONFIGS))
    parser.add_argument("--seeds",nargs="+",type=int,default=[2026,2027,2028])
    parser.add_argument("--max-epochs",type=int,default=1500)
    parser.add_argument("--patience",type=int,default=150)
    parser.add_argument("--eval-every",type=int,default=5)
    parser.add_argument("--min-delta",type=float,default=.001)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--lr",type=float,default=.001)
    parser.add_argument("--weight-decay",type=float,default=.0001)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(InterruptedError("SIGTERM")))
    run(args)

if __name__=="__main__": main()
