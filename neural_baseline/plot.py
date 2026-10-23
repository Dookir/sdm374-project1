"""Plot only the fixed teacher scores and separately labeled training diagnostics."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neural_baseline.models import CONFIGS

def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument("out"); a=p.parse_args(); out=Path(a.out)
    rows=json.loads((out/"comparison.json").read_text()); baseline=rows[0]["teacher_script_score"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for name in CONFIGS:
        epochs=sorted({r["epoch"] for r in rows if r.get("architecture")==name})
        ens=[min((r for r in rows if r.get("architecture")==name and r["epoch"]==e),key=lambda r:r["teacher_script_score"]) for e in epochs]
        axes[0].plot([r["epoch"] for r in ens],[r["teacher_script_score"] for r in ens],marker="o",label=name)
        h=json.loads((out/"models"/f"{name}_2026"/"history.json").read_text())
        axes[1].plot([r["epoch"] for r in h],[r["training_mse"] for r in h],label=name)
    axes[0].axhline(baseline,color="black",linestyle="--",label="Original baseline")
    axes[0].set_xlabel("Training checkpoint epoch"); axes[0].set_ylabel("Original teacher CLI score (lower is better)")
    axes[0].set_title("Best original score per checkpoint: seeds + ensemble")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Training MSE (diagnostic only)"); axes[1].set_yscale("log")
    axes[1].set_title("Seed 2026 training curves; NOT used for ranking")
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(out/"training_and_teacher_scores.png",dpi=160); plt.close(fig)
if __name__=="__main__": main()
