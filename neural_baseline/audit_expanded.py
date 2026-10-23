"""Audit expanded experiments using only recorded outputs of the original scorer."""
from pathlib import Path
import argparse,hashlib,json
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def audit(out,partial=False):
    out=Path(out); m=json.loads((out/"manifest.json").read_text())
    for family in ("protected_files","source_sha256","previous_reference_hashes"):
        for path,h in m[family].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    rows=json.loads((out/"comparison.json").read_text()); assert len({r["name"] for r in rows})==len(rows)
    for row in rows:
        c=json.loads((out/"commands"/(row["name"]+".json")).read_text())
        assert c["argv"][2]=="predict" and "--model" not in c["argv"]
        assert Path(c["argv"][1])==ROOT/"Lab6/baseline.py"
        assert float(json.loads(c["stdout"])["predicted_ramping"])==row["teacher_script_score"]
        w=np.array(json.loads((ROOT/row["weights"]).read_text()),float)
        assert w.shape==(3,7) and np.isfinite(w).all() and (w>=0).all()
        np.testing.assert_allclose(w.sum(1),1,rtol=0,atol=1e-12)
        if row["kind"]=="single": assert row["checkpoint_exact"] and (ROOT/row["checkpoint"]).exists()
        for path in row.get("checkpoints",[]): assert (ROOT/path).exists()
    counts=dict(Counter(r["kind"] for r in rows))
    if not partial:
        d=json.loads((out/"complete.json").read_text())
        assert d["state"]=="complete" and d["scored"]==len(rows)==m["expected_scores"]
        assert d["cuda_device_count"]==1
        assert d["selected_score"]<=min(r["teacher_script_score"] for r in rows)+1e-8
        winner=next(r for r in rows if r["name"]==d["selected"])
        assert json.loads((ROOT/winner["weights"]).read_text())==json.loads((out/"weights_recommended.json").read_text())
        n=len(m["configs"]); seeds=len(m["seeds"]); cps=len(m["checkpoints"])
        assert counts==dict(baseline=1,previous_neural=1,single=n*seeds*cps,ensemble=n*cps,refined=3)
    report=dict(scored=len(rows),expected=m["expected_scores"],kinds=counts,complete=not partial,
                teacher_files_unchanged=True,previous_best_unchanged=True,source_hashes_match=True,
                original_CLI_only=True,all_scores_match_stdout=True,all_weights_legal=True,gpu_uuid=m["gpu_uuid"])
    if not partial: (out/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2)); return report
if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__); p.add_argument("out"); p.add_argument("--partial",action="store_true")
    a=p.parse_args(); audit(a.out,a.partial)
