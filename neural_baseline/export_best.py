"""Export the best neural checkpoint(s) selected solely by original teacher CLI."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument("out"); a=p.parse_args(); out=Path(a.out)
    done=json.loads((out/"complete.json").read_text()); assert done["state"]=="complete"
    row=done["best_neural"]; best=out/"best_neural"; best.mkdir(exist_ok=True)
    sources=[row["checkpoint"]] if row["kind"]=="single" else row["checkpoints"]
    exported=[]
    for i,source in enumerate(sources):
        target=best/("model.pt" if len(sources)==1 else f"model_{i}.pt")
        shutil.copyfile(ROOT/source,target)
        h=hashlib.sha256(target.read_bytes()).hexdigest()
        assert h==hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
        exported.append(dict(file=target.name,source=source,sha256=h))
    shutil.copyfile(ROOT/row["weights"],best/"weights.json")
    data=dict(selection_criterion="lowest unchanged original teacher CLI score among neural candidates",
              trial=row["name"],architecture=row["architecture"],epoch=row["epoch"],kind=row["kind"],
              teacher_script_score=row["teacher_script_score"],baseline_score=done["baseline_score"],
              beats_baseline=row["teacher_script_score"]<done["baseline_score"]-1e-8,
              models=exported)
    (best/"selection.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(data,indent=2))
if __name__=="__main__": main()
