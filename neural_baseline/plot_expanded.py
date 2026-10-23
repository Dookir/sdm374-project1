"""Plot native scorer minima for declared configurations; no surrogate metric."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument("out"); a=p.parse_args(); out=Path(a.out)
    rows=json.loads((out/"comparison.json").read_text()); m=json.loads((out/"manifest.json").read_text())
    values=[]
    for name in m["configs"]:
        candidates=[r for r in rows if r.get("architecture")==name]
        if candidates: values.append(min(candidates,key=lambda r:r["teacher_script_score"]))
    values.sort(key=lambda r:r["teacher_script_score"])
    baseline=next(r["teacher_script_score"] for r in rows if r["kind"]=="baseline")
    previous=next(r["teacher_script_score"] for r in rows if r["kind"]=="previous_neural")
    fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    for i,row in enumerate(values):
        color="#3286b8" if m["configs"][row["architecture"]]["activation"]=="relu" else "#48a078"
        ax.scatter([row["teacher_script_score"]],[i],s=45,color=color)
        ax.annotate(f"{row['teacher_script_score']:.3f}",(row["teacher_script_score"],i),xytext=(6,0),textcoords="offset points",va="center",fontsize=9)
    ax.set_yticks(range(len(values)),[r["architecture"] for r in values]); ax.invert_yaxis()
    ax.axvline(baseline,color="black",ls="--",label=f"Original baseline {baseline:.3f}")
    ax.axvline(previous,color="#a24b8c",ls=":",label=f"Previous neural {previous:.3f}")
    ax.set_xlim(min(baseline,min(r["teacher_script_score"] for r in values))-3,max(r["teacher_script_score"] for r in values)+8)
    ax.set_xlabel("Unchanged original teacher CLI score (lower is better)")
    ax.set_title("Expanded neural search: best scored candidate per configuration")
    ax.legend(fontsize=9); ax.grid(axis="x",alpha=.2)
    fig.savefig(out/"expanded_teacher_scores.png",dpi=160); plt.close(fig)
if __name__=="__main__": main()
