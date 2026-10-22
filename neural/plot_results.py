"""Render validation plots from completed experiment artifacts."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument("run"); a=p.parse_args(); out=Path(a.run)
    table=pd.read_csv(out/"comparison.csv"); table=table.sort_values("ensemble_mae")
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    colors=["#df9635" if n in ("catboost_reference","quadratic_ridge") else "#3577b0" for n in table.model]
    for ax,key,label in zip(axes,("ensemble_mae","ensemble_regret_at2"),("OOF MAE (lower is better)","Mean fold regret@2 (lower is better)")):
        bars=ax.barh(table.model,table[key],color=colors)
        ax.invert_yaxis(); ax.set_xlabel(label); ax.grid(axis="x",alpha=.2); ax.set_axisbelow(True)
        ax.bar_label(bars,fmt="%.2f",padding=3); ax.set_xlim(0,table[key].max()*1.18)
    fig.suptitle("Five outer folds; neural predictions averaged across 3 seeds\nKnown-label validation, not scores for new optimized weights")
    fig.savefig(out/"comparison.png",dpi=160); plt.close(fig)
    pred=pd.read_csv(out/"oof_predictions.csv")
    neural=table[~table.model.isin(["quadratic_ridge","catboost_reference"])].iloc[0].model
    names=["quadratic_ridge","catboost_reference",neural]
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    lo=min(pred.y_true.min(),min(pred[n+"_ensemble"].min() for n in names))
    hi=max(pred.y_true.max(),max(pred[n+"_ensemble"].max() for n in names))
    for ax,name in zip(axes,names):
        ax.scatter(pred.y_true,pred[name+"_ensemble"],s=8,alpha=.35)
        ax.plot([lo,hi],[lo,hi],"k--",lw=1); ax.set_title(name)
        ax.set_xlabel("Observed ramping"); ax.set_ylabel("Out-of-fold prediction"); ax.set_xlim(lo-3,hi+3); ax.set_ylim(lo-3,hi+3)
    fig.savefig(out/"predictions.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for seed in (2026,2027,2028):
        row=json.loads((out/"fold0"/f"{neural}_{seed}.json").read_text()); h=row["inner"]["history"]
        axes[0].plot([r["epoch"] for r in h],[r["train_mae"] for r in h],label=str(seed))
        axes[1].plot([r["epoch"] for r in h],[r["inner_mae"] for r in h],label=str(seed))
        axes[1].scatter([row["inner"]["best_epoch"]],[row["inner"]["best_inner_mae"]],s=30)
    for ax,label in zip(axes,("Fitting subset MAE","Inner stopping subset MAE")):
        ax.set_xlabel("Epoch"); ax.set_ylabel(label); ax.legend(title="Seed"); ax.grid(alpha=.2)
    fig.suptitle(f"{neural}: fold 0 training curves (outer test never used for stopping)")
    fig.savefig(out/"training_curves.png",dpi=160); plt.close(fig)
if __name__=="__main__": main()
