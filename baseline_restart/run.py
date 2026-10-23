"""Original baseline structure: generate model -> native optimize -> native predict."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key,"4")
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from Lab6 import baseline
ROOT=Path(__file__).resolve().parents[1]

def dump(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")

def native(out,label,command):
    args=[sys.executable,str(ROOT/"Lab6/baseline.py"),*map(str,command)]
    result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,check=True)
    dump(out/"commands"/(label+".json"),dict(argv=args,stdout=result.stdout,stderr=result.stderr))
    return result.stdout

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(__doc__); parser.add_argument("--out",default="baseline_restart/results/first_run_20261023")
    args=parser.parse_args(); out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists(): raise ValueError("Completed directory exists; choose a new --out")
    protected={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/"Lab6").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    ids,X,y=baseline.load_table(ROOT/"Lab6/dataset_same8/train.csv")
    starts=baseline.search_starts(X,y,16)
    rows=[]; started=time.monotonic()
    def score(name,weights_file,model_file=None,model_info=None,search_info=None):
        raw=native(out,name+"_predict",["predict","--weights",weights_file])
        # NEVER pass the generating model as --model to the judge.
        value=float(json.loads(raw)["predicted_ramping"])
        weights=np.asarray(json.loads(Path(weights_file).read_text()),float)
        assert weights.shape==(3,7) and np.isfinite(weights).all() and (weights>=0).all()
        assert np.allclose(weights.sum(1),1,atol=1e-12,rtol=0)
        row=dict(name=name,teacher_script_score=value,weights=str(Path(weights_file).relative_to(ROOT)),
                 generating_model=str(Path(model_file).relative_to(ROOT)) if model_file else None,
                 model_info=model_info,search_info=search_info)
        rows.append(row); dump(out/"comparison.json",rows)
        pd.DataFrame([{k:r[k] for k in ("name","teacher_script_score","weights","generating_model")} for r in rows]).to_csv(out/"comparison.csv",index=False)
        print(f"{len(rows):02d} {name}: {value:.9f}",flush=True)
        return value
    def optimize_and_score(name,model_path,info):
        path=out/"weights"/(name+".json")
        path.parent.mkdir(exist_ok=True)
        native(out,name+"_optimize",["optimize","--model",model_path,"--output",path])
        score(name,path,model_path,info)
    score("provided_baseline",ROOT/"Lab6/weights.json")
    optimize_and_score("provided_model_native_optimize",ROOT/"Lab6/model.npz",dict(source="unchanged supplied teacher model"))
    modeldir=out/"models"; modeldir.mkdir(exist_ok=True)
    retrained=modeldir/"native_train.npz"
    native(out,"native_train",["train","--train",ROOT/"Lab6/dataset_same8/train.csv","--model",retrained])
    with np.load(retrained) as z: alpha=float(z["alpha"])
    optimize_and_score("native_train_then_optimize",retrained,dict(alpha=alpha,source="unmodified baseline.py train"))
    specs=[("linear_ols",1,"ols",0.),("quadratic_ols",2,"ols",0.)]
    specs += [(f"quadratic_ridge_{alpha:g}",2,"ridge",alpha) for alpha in (.1,1.,10.,100.,1000.)]
    specs += [(f"quadratic_lasso_{alpha:g}",2,"lasso",alpha) for alpha in (.01,.1,1.)]
    for name,degree,kind,alpha in specs:
        if kind=="ridge":
            model=baseline.make_model(alpha)
        else:
            reg=LinearRegression() if kind=="ols" else Lasso(alpha=alpha,max_iter=100000,tol=1e-7,selection="cyclic")
            # Preserve original transformer names so the teacher's exact exporter works.
            model=Pipeline([("poly",PolynomialFeatures(degree=degree,include_bias=False)),
                            ("scale",StandardScaler()),("ridge",reg)])
        model.fit(X,y)
        if kind=="ols":
            # Original NPZ exporter expects alpha; zero means no regularization.
            model.named_steps["ridge"].alpha=0.0
        path=modeldir/(name+".npz"); baseline.save_model(model,starts,path)
        c,a,H=baseline.quadratic_form(model)
        np.testing.assert_allclose(model.predict(X),baseline.predict_quadratic(X,c,a,H),atol=1e-8,rtol=0)
        optimize_and_score(name,path,dict(degree=degree,kind=kind,alpha=alpha,rows=len(y),
            feature_count=int(model.named_steps["poly"].n_output_features_)))
    # Same original coefficients and optimizer; increase only the number of starts.
    # Includes all 7^3 vertices and mixed-concentration simplex samples.
    with np.load(ROOT/"Lab6/model.npz") as z: original={k:z[k].copy() for k in z.files}
    rng=np.random.default_rng(20261023)
    vertices=np.array([np.concatenate([np.eye(7)[i],np.eye(7)[j],np.eye(7)[k]])
                       for i in range(7) for j in range(7) for k in range(7)])
    extra=np.concatenate([rng.dirichlet(np.full(7,alpha),size=(256,3)).reshape(256,21)
                          for alpha in (.1,.5,1.,5.)])
    expanded=np.concatenate([starts,np.asarray(json.loads((ROOT/"Lab6/weights.json").read_text())).reshape(1,21),vertices,extra])
    original["search_starts"]=expanded
    path=modeldir/"provided_model_expanded_starts.npz"; np.savez_compressed(path,**original)
    optimize_and_score("provided_model_expanded_starts",path,dict(source="unchanged teacher coefficients",starts=len(expanded),
                       optimizer="unchanged baseline.py optimize"))
    # Accept only an actual reduction in the fixed teacher CLI output.
    winner=rows[0]
    for row in rows[1:]:
        if row["teacher_script_score"]<winner["teacher_script_score"]-1e-8: winner=row
    best=np.asarray(json.loads((ROOT/winner["weights"]).read_text()),float)
    dump(out/"weights_recommended.json",best.tolist())
    recommended_score=float(json.loads(native(out,"recommended_predict",["predict","--weights",out/"weights_recommended.json"]))["predicted_ramping"])
    assert abs(recommended_score-winner["teacher_script_score"])<1e-9
    for p,h in protected.items(): assert digest(ROOT/p)==h,p
    dump(out/"complete.json",dict(state="complete",count=len(rows),selected=winner["name"],
        baseline_score=rows[0]["teacher_script_score"],selected_score=recommended_score,
        improvement=rows[0]["teacher_script_score"]-recommended_score,
        selection_rule="Only unchanged Lab6/baseline.py predict output; lower is better. Ties within 1e-8 retain original.",
        teacher_files_unchanged=True,protected_files=protected,all_weights_valid=True,
        generation_matches_saved_model=True,gpu_used=False,elapsed_seconds=time.monotonic()-started))
    print("SELECTED",winner["name"],recommended_score,flush=True)
if __name__=="__main__": main()
