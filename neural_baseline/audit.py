"""Audit actual teacher CLI records, legal weights, checkpoints and file integrity."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from neural_baseline.models import CONFIGS
ROOT=Path(__file__).resolve().parents[1]

def audit(out,partial=False):
    out=Path(out); manifest=json.loads((out/"manifest.json").read_text())
    for p,h in manifest["protected_files"].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    for p,h in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    rows=json.loads((out/"comparison.json").read_text())
    assert len({r["name"] for r in rows})==len(rows)
    for row in rows:
        cmd=json.loads((out/"commands"/(row["name"]+".json")).read_text())
        assert cmd["argv"][2]=="predict" and "--model" not in cmd["argv"]
        assert Path(cmd["argv"][1])==ROOT/"Lab6/baseline.py"
        assert float(json.loads(cmd["stdout"])["predicted_ramping"])==row["teacher_script_score"]
        w=np.asarray(json.loads((ROOT/row["weights"]).read_text()),float)
        assert w.shape==(3,7) and np.isfinite(w).all() and (w>=0).all()
        np.testing.assert_allclose(w.sum(1),1,atol=1e-12,rtol=0)
        if row["kind"]=="single": assert row["checkpoint_exact"] and (ROOT/row["checkpoint"]).exists()
        if row["kind"]=="ensemble": assert all((ROOT/p).exists() for p in row["checkpoints"])
    expected=1+len(CONFIGS)*(len(manifest["seeds"])+1)*len(manifest["checkpoints"])
    if not partial:
        done=json.loads((out/"complete.json").read_text())
        assert done["state"]=="complete" and done["scored"]==len(rows)==expected
        assert done["cuda_device_count"]==1
        assert done["selected_score"]<=min(r["teacher_script_score"] for r in rows)+1e-8
        selected=next(r for r in rows if r["name"]==done["selected"])
        assert json.loads((ROOT/selected["weights"]).read_text())==json.loads((out/"weights_recommended.json").read_text())
    report=dict(scored=len(rows),expected=expected,teacher_files_unchanged=True,
                sole_scorer="unchanged original baseline.py predict",CLI_outputs_match=True,
                all_weights_legal=True,all_recorded_checkpoints_exist=True,source_hashes_match=True,
                gpu_uuid=manifest["gpu_uuid"],complete=not partial)
    if not partial: (out/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2)); return report
if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__); p.add_argument("out"); p.add_argument("--partial",action="store_true")
    a=p.parse_args(); audit(a.out,a.partial)
