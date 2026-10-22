"""Metrics on held-out true labels, not surrogate minima."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim != 1 or p.shape != y.shape or len(y)<2 or not np.isfinite([y,p]).all():
        raise ValueError("Need matching finite 1D arrays")
    order = np.argsort(p, kind="stable")[:2]
    result = dict(mae=float(mean_absolute_error(y,p)), rmse=float(mean_squared_error(y,p)**.5),
                  r2=float(r2_score(y,p)), top2_true=float(y[order].min()),
                  regret_at2=float(y[order].min()-y.min()),
                  selected_prediction_bias=float((p[order]-y[order]).mean()))
    for percent in (5,20):
        idx=np.argsort(y,kind="stable")[:max(1,int(np.ceil(len(y)*percent/100))) ]
        result[f"low{percent}_mae"]=float(np.abs(p[idx]-y[idx]).mean())
    return result

def oof_metrics(y,p,fold_ids):
    result=metrics(y,p)
    per_fold=[metrics(y[fold_ids==f],p[fold_ids==f]) for f in np.unique(fold_ids)]
    for key in ("top2_true","regret_at2","selected_prediction_bias"):
        result[key]=float(np.mean([row[key] for row in per_fold]))
    return result
