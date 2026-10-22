"""Audit completed or partially completed experiments against raw held-out labels."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from Lab6 import baseline
from neural.training_metrics import metrics
ROOT=Path(__file__).resolve().parents[1]

def audit(out, require_complete=True):
    out=Path(out); manifest=json.loads((out/"manifest.json").read_text())
    ids,X,y=baseline.load_table(ROOT/"Lab6/dataset_same8/train.csv")
    assert len(np.unique(X,axis=0))==len(X)
    id_to_row={v:i for i,v in enumerate(ids)}; tests=[]; count=0
    assert hashlib.sha256((ROOT/"Lab6/dataset_same8/train.csv").read_bytes()).hexdigest()==manifest["data_sha256"]
    for p,h in manifest["sources"].items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    for f,split in enumerate(manifest["splits"]):
        train,test,fit,stop=[set(split[k]) for k in ("train","test","fit","stop")]
        assert not train&test and not fit&stop and fit|stop==train
        assert train|test==set(ids); tests.extend(test)
        expected=np.array([id_to_row[v] for v in split["test"]])
        for name in manifest["config"]["names"]:
            for seed in manifest["config"]["seeds"]:
                prefix=out/f"fold{f}"/f"{name}_{seed}"
                if not prefix.with_suffix(".json").exists():
                    assert not require_complete, str(prefix)
                    continue
                row=json.loads(prefix.with_suffix(".json").read_text())
                with np.load(prefix.with_suffix(".npz")) as z:
                    assert np.array_equal(z["row_indices"],expected)
                    recomputed=metrics(y[expected],z["prediction"])
                for k,v in recomputed.items(): assert np.isclose(v,row["test"][k],rtol=0,atol=1e-10)
                assert row["checkpoint_exact"] and prefix.with_suffix(".pt").exists()
                assert row["inner"]["best_epoch"]==row["refit"]["epochs_run"]
                count+=1
    assert len(tests)==len(ids) and set(tests)==set(ids)
    fingerprints=json.loads((ROOT/"offline/results/manifest.json").read_text())["protected_files"]
    for p,h in fingerprints.items(): assert hashlib.sha256((ROOT/"Lab6"/p).read_bytes()).hexdigest()==h
    status=json.loads((out/"status.json").read_text())
    if require_complete:
        assert status["state"]=="complete" and status["completed"]==count==status["total"]
        assert status["cuda_device_count"]==1
    report=dict(completed_experiments=count,folds=len(manifest["splits"]),rows=len(ids),
        disjoint_splits=True,all_rows_predicted_once_per_seed=require_complete,metrics_match_labels=True,
        checkpoints_exact=True,teacher_files_unchanged=True,sources_unchanged=True,
        gpu_uuid=manifest["gpu_uuid"],status=status)
    print(json.dumps(report,indent=2))
    if require_complete: (out/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__); p.add_argument("out"); p.add_argument("--partial",action="store_true")
    a=p.parse_args(); audit(a.out,not a.partial)
