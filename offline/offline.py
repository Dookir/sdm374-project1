"""Reproducible offline surrogate comparison and constrained weight search.

No simulator calls. All scores on new weights are predictions, never measurements.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterSampler, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "Lab6"
sys.path.insert(0, str(LAB))
import baseline

FEATURES = baseline.FEATURES
SEEDS = (2026, 2027, 2028)
VERSION = 1


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    return {name: sha(LAB / name) for name in ("baseline.py", "model.npz", "weights.json")}


def environment():
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib")}}


def load_data(path):
    ids, X, y = baseline.load_table(path)
    if np.unique(X, axis=0).shape[0] != len(X):
        raise ValueError("Duplicate weight rows: group duplicates before validation.")
    return np.asarray(ids), X, y


def split_indices(n):
    dev, check = train_test_split(np.arange(n), test_size=0.2, random_state=20261022)
    return np.sort(dev), np.sort(check)


def init_run(out, data):
    out.mkdir(parents=True, exist_ok=True)
    ids, X, y = load_data(data)
    dev, check = split_indices(len(y))
    manifest = {"version": VERSION, "data_sha256": sha(data), "source_sha256": sha(__file__),
                "protected_files": fingerprint(), "features": FEATURES,
                "development_ids": ids[dev].tolist(), "internal_check_ids": ids[check].tolist()}
    target = out / "manifest.json"
    if target.exists() and read_json(target) != manifest:
        raise ValueError("Run provenance changed. Use a new --out directory; do not mix experiments.")
    if not target.exists():
        write_json(target, manifest)
        write_json(out / "environment.json", environment())
    return ids, X, y, dev, check


def make_model(spec, seed=2026):
    p = dict(spec["params"])
    family = spec["family"]
    if family == "ridge2":
        return baseline.make_model(**p)
    if family == "histgb":
        return HistGradientBoostingRegressor(**p, random_state=seed, early_stopping=False)
    if family == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**p, random_seed=seed, loss_function="RMSE", verbose=False,
                                 allow_writing_files=False, thread_count=4, task_type="CPU")
    if family == "extratrees":
        return ExtraTreesRegressor(**p, random_state=seed, n_jobs=4)
    if family == "svr":
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", cache_size=512, **p))
    if family == "krr":
        return make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", **p))
    raise ValueError(f"Unknown family: {family}")


def trial_specs(scale=1.0):
    spaces = {
        "histgb": (30, {"max_iter": [250, 500, 900], "learning_rate": [.025, .05, .08],
                        "max_leaf_nodes": [7, 15, 31], "min_samples_leaf": [8, 15, 25, 40],
                        "l2_regularization": [0.1, 1., 5., 20.]}),
        "catboost": (30, {"iterations": [500, 1000, 1600], "depth": [4, 5, 6, 7],
                          "learning_rate": [.025, .05, .08], "l2_leaf_reg": [1., 3., 10., 30.],
                          "random_strength": [.1, 1.]}),
        "extratrees": (12, {"n_estimators": [400], "min_samples_leaf": [1, 2, 4, 8],
                            "max_features": [.7, 1.], "max_depth": [None, 16, 24]}),
        "svr": (12, {"C": [30., 100., 300., 1000.], "epsilon": [.1, .5, 1.],
                      "gamma": [.003, .01, .03, .1]}),
        "krr": (12, {"alpha": [.0001, .001, .01, .1, 1.], "gamma": [.001, .003, .01, .03, .1]}),
    }
    yield {"id": "ridge2_000", "family": "ridge2", "params": {"alpha": 100.}}
    schedules = []
    for i, (family, (count, space)) in enumerate(spaces.items()):
        sampled = list(ParameterSampler(space, n_iter=max(1, round(count * scale)), random_state=2026 + i))
        schedules.append([{"id": f"{family}_{j:03d}", "family": family, "params": params}
                          for j, params in enumerate(sampled)])
    for group in itertools.zip_longest(*schedules):
        yield from (spec for spec in group if spec is not None)


def scores(y, pred, folds):
    # Fold-specific top-10 metrics evaluate choosing unseen combinations, not fitted rows.
    topmeans, topbests = [], []
    for idx in folds:
        chosen = idx[np.argsort(pred[idx], kind="stable")[:min(10, len(idx))]]
        topmeans.append(float(y[chosen].mean()))
        topbests.append(float(y[chosen].min()))
    low = y <= np.quantile(y, .2)
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred)),
            "low20_mae": float(mean_absolute_error(y[low], pred[low])),
            "top10_true_mean": float(np.mean(topmeans)), "top10_true_best_mean": float(np.mean(topbests))}


class BudgetExpired(Exception):
    pass


def cross_validate(spec, X, y, seed, cache, deadline):
    prefix = cache / f"{spec['id']}_seed{seed}"
    meta_path, pred_path = prefix.with_suffix(".json"), prefix.with_suffix(".npz")
    folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X))
    held_out = [v for _, v in folds]
    if meta_path.exists() and pred_path.exists():
        meta = read_json(meta_path)
        if meta["spec"] != spec:
            raise ValueError("Cached trial specification differs; use a new output directory.")
        with np.load(pred_path) as p:
            return meta, p["predictions"]
    cache.mkdir(parents=True, exist_ok=True)
    pred = np.full(len(y), np.nan)
    started = time.monotonic()
    for train, val in folds:
        if time.monotonic() >= deadline:
            raise BudgetExpired()
        model = make_model(spec, seed)
        model.fit(X[train], y[train])
        pred[val] = model.predict(X[val])
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite OOF predictions")
    meta = {"spec": spec, "seed": seed, "seconds": time.monotonic() - started,
            "metrics": scores(y, pred, held_out)}
    np.savez_compressed(pred_path, predictions=pred)
    write_json(meta_path, meta)
    return meta, pred


def choose(rows):
    best = min(r["metrics"]["mae"] for r in rows)
    eligible = [r for r in rows if r["metrics"]["mae"] <= best * 1.05]
    # Treat top-10 differences below 0.1 kWh as ties, then favor low-tail accuracy.
    lowest = min(r["metrics"]["top10_true_mean"] for r in eligible)
    eligible = [r for r in eligible if r["metrics"]["top10_true_mean"] <= lowest + .1]
    return min(eligible, key=lambda r: (r["metrics"]["low20_mae"], r["metrics"]["mae"], r["id"]))


def mean_metrics(results):
    return {key: float(np.mean([r[key] for r in results])) for key in results[0]}


def export_trials(out):
    rows = []
    for path in sorted((out / "cv").glob("*_seed2026.json")):
        r = read_json(path)
        rows.append({"id": r["spec"]["id"], "family": r["spec"]["family"], **r["metrics"],
                     "seconds": r["seconds"], "params": json.dumps(r["spec"]["params"], sort_keys=True)})
    if rows:
        pd.DataFrame(rows).to_csv(out / "trial_comparison.csv", index=False)


def benchmark(args):
    out = args.out
    ids, X, y, dev, check = init_run(out, args.data)
    if (out / "selection.json").exists():
        selection = read_json(out / "selection.json")
        print("Selection already locked; no retuning or repeated internal-check evaluation.", flush=True)
        final_check(out, X, y, dev, check, selection)
        return
    started = time.monotonic()
    tuning_deadline = started + args.tune_minutes * 60
    full_deadline = started + (args.tune_minutes + args.validation_minutes) * 60
    write_json(out / "search_config.json", {"trial_scale": args.trial_scale, "tune_minutes": args.tune_minutes,
                                          "validation_minutes": args.validation_minutes,
                                          "trial_schedule": list(trial_specs(args.trial_scale))})
    completed, failures = [], []
    try:
        for spec in trial_specs(args.trial_scale):
            try:
                meta, _ = cross_validate(spec, X[dev], y[dev], 2026, out / "cv", tuning_deadline)
                completed.append(meta)
                print(f"{spec['id']}: MAE={meta['metrics']['mae']:.4f}, "
                      f"top10={meta['metrics']['top10_true_mean']:.4f}, {meta['seconds']:.1f}s", flush=True)
                export_trials(out)
            except BudgetExpired:
                print("Tuning budget reached; completed trials are saved.", flush=True)
                break
            except Exception as error:
                failures.append({"spec": spec, "error": repr(error)})
                write_json(out / "failed_trials.json", failures)
                print(f"Trial failed: {spec['id']}: {error}", flush=True)
    finally:
        export_trials(out)
    if not completed:
        raise RuntimeError("No completed trials. Increase the budget or inspect failed_trials.json.")
    family_best = []
    for family in dict.fromkeys(r["spec"]["family"] for r in completed):
        candidates = [{"id": r["spec"]["id"], "spec": r["spec"], "metrics": r["metrics"]}
                      for r in completed if r["spec"]["family"] == family]
        family_best.append(choose(candidates))
    nonbaseline = sorted((r for r in family_best if r["spec"]["family"] != "ridge2"),
                         key=lambda r: r["metrics"]["mae"])
    finalists = nonbaseline[:3]
    baseline_rows = [r for r in family_best if r["spec"]["family"] == "ridge2"]
    finalists += baseline_rows
    stable, oofs = [], {}
    for row in finalists:
        seed_metrics, predictions, seeds = [], [], []
        for seed in SEEDS:
            try:
                meta, pred = cross_validate(row["spec"], X[dev], y[dev], seed, out / "cv", full_deadline)
            except BudgetExpired:
                break
            seed_metrics.append(meta["metrics"])
            predictions.append(pred)
            seeds.append(seed)
        if len(seeds) != len(SEEDS):
            write_json(out / "status.json", {"status": "validation_incomplete", "id": row["id"], "seeds": seeds})
            raise RuntimeError("Stability budget exhausted. Resume benchmark; cached trials are reused, check set untouched.")
        stable.append({**row, "metrics": mean_metrics(seed_metrics), "per_seed": seed_metrics, "seeds": seeds})
        oofs[row["id"]] = predictions
        write_json(out / "stability_progress.json", stable)
        print(f"Stable {row['id']}: {stable[-1]['metrics']}", flush=True)
    review = [r for r in stable if r["spec"]["family"] != "ridge2"]
    if not review:
        review = stable[:1]
    review_best = min(r["metrics"]["mae"] for r in review)
    review = [r for r in review if r["metrics"]["mae"] <= 1.2 * review_best]
    weights = np.array([1 / r["metrics"]["mae"] ** 2 for r in review])
    weights /= weights.sum()
    ensemble_metrics = []
    for j, seed in enumerate(SEEDS):
        p = sum(w * oofs[r["id"]][j] for w, r in zip(weights, review))
        folds = [v for _, v in KFold(5, shuffle=True, random_state=seed).split(dev)]
        ensemble_metrics.append(scores(y[dev], p, folds))
    if len(review) > 1:
        stable.append({"id": "ensemble", "metrics": mean_metrics(ensemble_metrics),
                       "per_seed": ensemble_metrics, "seeds": list(SEEDS)})
    chosen = choose(stable)
    selection = {"selected": chosen["id"], "metrics": chosen["metrics"],
                 "review_specs": [r["spec"] for r in review], "review_weights": weights.tolist(),
                 "selected_spec": next((r["spec"] for r in stable if r["id"] == chosen["id"] and "spec" in r), None),
                 "baseline_spec": {"id": "ridge2_000", "family": "ridge2", "params": {"alpha": 100.}},
                 "seeds": list(SEEDS), "selection_rule": "MAE within 5%; lowest top10 mean (0.1 tie); low20 MAE", 
                 "warning": "Development CV is used for tuning, selection and ensemble weights, not an independent test.",
                 "benchmark_seconds": time.monotonic() - started}
    # Persist decision before touching any internal-check target for evaluation.
    write_json(out / "selection.json", selection)
    write_json(out / "stability.json", stable)
    pd.DataFrame([{"id": r["id"], **r["metrics"]} for r in stable]).to_csv(out / "model_comparison.csv", index=False)
    final_check(out, X, y, dev, check, selection)
    write_json(out / "status.json", {"status": "benchmark_complete", "selected": chosen["id"]})
    print(f"Locked selection: {chosen['id']}", flush=True)


def fit_members(selection, X, y):
    specs = {s["id"]: s for s in selection["review_specs"]}
    if selection["selected_spec"]:
        specs[selection["selected_spec"]["id"]] = selection["selected_spec"]
    return {key: make_model(spec).fit(X, y) for key, spec in specs.items()}


def predict_members(models, selection, X):
    individual = {key: np.asarray(m.predict(X)).reshape(-1) for key, m in models.items()}
    matrix = np.stack([individual[s["id"]] for s in selection["review_specs"]], axis=1)
    weights = np.asarray(selection["review_weights"])
    review_mean = matrix @ weights
    disagreement = np.sqrt(np.maximum(0, (matrix - review_mean[:, None]) ** 2 @ weights))
    pred = review_mean if selection["selected"] == "ensemble" else individual[selection["selected"]]
    if not np.isfinite(pred).all() or not np.isfinite(disagreement).all():
        raise ValueError("Non-finite model predictions")
    return pred, disagreement, review_mean, individual


def final_check(out, X, y, dev, check, selection):
    target = out / "internal_check.json"
    if target.exists():
        print("Internal check already recorded:", read_json(target), flush=True)
        return
    pred_file = out / "internal_check_predictions.csv"
    if pred_file.exists():
        table = pd.read_csv(pred_file, float_precision="round_trip")
        if table.row_index.tolist() != check.tolist():
            raise ValueError("Internal check row mismatch")
        chosen, base = table.selected_prediction.to_numpy(), table.baseline_prediction.to_numpy()
    else:
        models = fit_members(selection, X[dev], y[dev])
        chosen = predict_members(models, selection, X[check])[0]
        base = make_model(selection["baseline_spec"]).fit(X[dev], y[dev]).predict(X[check])
        # Save predictions once, enabling a metrics-only recovery after interruption.
        table = pd.DataFrame({"row_index": check, "true_ramping": y[check],
                              "selected_prediction": chosen, "baseline_prediction": base})
        tmp = pred_file.with_suffix(".csv.tmp")
        table.to_csv(tmp, index=False)
        tmp.replace(pred_file)
    result = {"selected": selection["selected"], "n": len(check),
              "selected_metrics": scores(y[check], chosen, [np.arange(len(check))]),
              "baseline_metrics": scores(y[check], base, [np.arange(len(check))]),
              "purpose": "One internal check on unseen weight combinations in the same eight-week environment. Not simulator performance.",
              "decision_frozen_before_check": True}
    write_json(target, result)
    print("Internal check:", result, flush=True)


def train(args):
    _, X, y, _, _ = init_run(args.out, args.data)
    selection = read_json(args.out / "selection.json")
    if not (args.out / "internal_check.json").exists():
        raise ValueError("Complete benchmark's internal check before full-data training.")
    dest = args.out / "model.joblib"
    if dest.exists():
        bundle = joblib.load(dest)
        if bundle["selection"] != selection or bundle["data_sha256"] != sha(args.data):
            raise ValueError("Saved model provenance mismatch")
        print("Full-data model already exists; retaining it.")
        return
    models = fit_members(selection, X, y)
    before = predict_members(models, selection, X[:32])[0]
    bundle = {"version": VERSION, "selection": selection, "models": models,
              "features": FEATURES, "data_sha256": sha(args.data), "environment": environment()}
    tmp = dest.with_suffix(".joblib.tmp")
    joblib.dump(bundle, tmp, compress=3)
    reloaded = joblib.load(tmp)
    after = predict_members(reloaded["models"], selection, X[:32])[0]
    if not np.array_equal(before, after):
        raise ValueError("Saved/reloaded model changed predictions")
    tmp.replace(dest)
    write_json(args.out / "training.json", {"n": len(y), "selected": selection["selected"],
                                            "serialization_exact_match": True,
                                            "note": "Refitted on all rows AFTER the internal check; no new test score calculated."})
    print(f"Saved full-data model: {dest}", flush=True)


def validate_weights(weights):
    a = np.asarray(weights, dtype=float)
    if a.shape == (3, 7):
        a = a.reshape(1, 21)
    elif a.shape == (21,):
        a = a.reshape(1, 21)
    elif a.ndim == 3 and a.shape[1:] == (3, 7):
        a = a.reshape(-1, 21)
    if a.ndim != 2 or a.shape[1] != 21 or len(a) == 0:
        raise ValueError("Expected (3,7), (21,), (N,21), or (N,3,7) weights")
    if not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.reshape(-1, 3, 7).sum(2), 1, atol=1e-12, rtol=0):
        raise ValueError("Weights must be finite, nonnegative and sum to one per building")
    return a


def load_bundle(args):
    # Only load locally created trusted joblib files: pickle formats can execute code.
    bundle = joblib.load(args.out / "model.joblib")
    if bundle["data_sha256"] != sha(args.data) or bundle["features"] != FEATURES:
        raise ValueError("Model/data provenance mismatch")
    return bundle


def predict_batches(bundle, X, batch=4096):
    values = [[], [], []]
    individual = {}
    for start in range(0, len(X), batch):
        pred, std, mean, members = predict_members(bundle["models"], bundle["selection"], X[start:start + batch])
        for collector, a in zip(values, (pred, std, mean)):
            collector.append(a)
        for key, a in members.items():
            individual.setdefault(key, []).append(a)
    return (*[np.concatenate(v) for v in values], {k: np.concatenate(v) for k, v in individual.items()})


def propose_candidates(X, y, count, rng):
    global_n = count // 2
    parts = [rng.dirichlet(np.full(7, alpha), size=(n, 3)).reshape(n, 21)
             for alpha, n in zip((.1, .3, 1., 3.), [len(x) for x in np.array_split(np.arange(global_n), 4)]) if n]
    local_n = count - global_n
    anchors = X[np.argsort(y, kind="stable")[:min(50, len(y))]]
    base = anchors[rng.integers(len(anchors), size=local_n)].reshape(-1, 3, 7)
    concentration = rng.choice([20., 80., 300.], size=local_n)
    alpha = .01 + base * concentration[:, None, None]
    local = rng.gamma(alpha, 1.)
    local /= local.sum(axis=2, keepdims=True)
    return np.vstack(parts + [local.reshape(-1, 21)])


def pair_transfers(X, rng, scale):
    z = np.array(X, copy=True).reshape(-1, 3, 7)
    rows = np.arange(len(z))
    building = rng.integers(3, size=len(z))
    source = rng.integers(7, size=len(z))
    target = (source + rng.integers(1, 7, size=len(z))) % 7
    amount = z[rows, building, source] * rng.uniform(0, scale, size=len(z))
    z[rows, building, source] -= amount
    z[rows, building, target] += amount
    return z.reshape(-1, 21)


def optimize(args):
    ids, X, y, _, _ = init_run(args.out, args.data)
    bundle = load_bundle(args)
    rng = np.random.default_rng(args.seed)
    deadline = time.monotonic() + args.search_minutes * 60
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute", n_jobs=2).fit(X)
    training_distances = nn.kneighbors(X, return_distance=True)[0][:, 1]
    limits = np.quantile(training_distances, [.5, .95])
    bestidx = int(np.argmin(y))
    base_weights = validate_weights(read_json(LAB / "weights.json"))
    pool = np.vstack([X[bestidx:bestidx + 1], base_weights,
                      propose_candidates(X, y, args.candidates, rng)])
    pred, std, mean, members = predict_batches(bundle, pool)
    dist = nn.kneighbors(pool, n_neighbors=1)[0][:, 0]
    profile_specs = [("conservative", 1., float(limits[0])), ("aggressive", .25, float(limits[1]))]
    # Refine top candidates from both profiles; every transfer preserves feasibility.
    history = []
    for round_no in range(args.local_rounds):
        if time.monotonic() >= deadline:
            break
        parents = []
        for _, penalty, radius in profile_specs:
            valid = np.flatnonzero(dist <= radius + 1e-12)
            ranked = valid[np.argsort(pred[valid] + penalty * std[valid], kind="stable")[:32]]
            parents.append(pool[ranked])
        parents = np.vstack(parents)
        trial = pair_transfers(np.repeat(parents, 16, axis=0), rng,
                               [1., .5, .25, .1, .05][min(round_no // 4, 4)])
        p, s, m, mem = predict_batches(bundle, trial)
        d = nn.kneighbors(trial, n_neighbors=1)[0][:, 0]
        pool, pred, std, mean, dist = (np.concatenate([a, b]) for a, b in
                                      zip((pool, pred, std, mean, dist), (trial, p, s, m, d)))
        for key in members:
            members[key] = np.concatenate([members[key], mem[key]])
        history.append({"round": round_no + 1, "evaluated": len(pool)})
        print(f"Local search {round_no + 1}: evaluated {len(pool)} weights", flush=True)
    validate_weights(pool)
    # Exact duplicates of observed training combinations have a known label.
    observed = {row.tobytes(): (str(candidate_id), float(label)) for candidate_id, row, label in zip(ids, X, y)}
    def describe(index, penalty=None):
        row = pool[index]
        measured = observed.get(row.tobytes())
        return {"predicted_ramping": float(pred[index]), "review_mean": float(mean[index]),
                "model_disagreement": float(std[index]), "nearest_training_distance": float(dist[index]),
                "individual_predictions": {k: float(v[index]) for k, v in members.items()},
                "objective": None if penalty is None else float(pred[index] + penalty * std[index]),
                "measured_ramping": None if measured is None else measured[1],
                "measured_candidate_id": None if measured is None else measured[0],
                "weights": row.reshape(3, 7).tolist()}
    candidates, chosen_indices = {}, []
    for name, penalty, radius in profile_specs:
        objective = pred + penalty * std
        valid = np.flatnonzero(dist <= radius + 1e-12)
        ranked = valid[np.argsort(objective[valid], kind="stable")]
        # Do not replace a known measured incumbent by a worse surrogate objective.
        improving = ranked[objective[ranked] < objective[0] - 1e-9]
        if chosen_indices:
            improving = improving[np.linalg.norm(pool[improving] - pool[chosen_indices[0]], axis=1) > 1e-3]
        idx = int(improving[0]) if len(improving) else 0
        chosen_indices.append(idx)
        candidates[name] = {**describe(idx, penalty), "penalty": penalty, "distance_limit": radius,
                            "fallback_to_measured_best": idx == 0,
                            "new_weight_true_score": "unknown; requires real evaluator" if idx else "known training measurement"}
        write_json(args.out / f"weights_{name}.json", candidates[name]["weights"])
    candidates["measured_best"] = describe(0)
    candidates["original_baseline"] = {**describe(1), "original_model_prediction": float(baseline.predict(base_weights.reshape(3, 7), LAB / "model.npz"))}
    write_json(args.out / "weights_measured_best.json", candidates["measured_best"]["weights"])
    write_json(args.out / "candidates.json", {"seed": args.seed, "requested_random_candidates": args.candidates,
                                             "total_evaluated": len(pool), "search_history": history,
                                             "distance_quantiles": {"q50": float(limits[0]), "q95": float(limits[1])},
                                             "uncertainty_warning": "Model disagreement is a heuristic, not a calibrated confidence interval.",
                                             "candidates": candidates})
    shortlist = []
    for name, penalty, radius in profile_specs:
        valid = np.flatnonzero(dist <= radius + 1e-12)
        ranked = valid[np.argsort(pred[valid] + penalty * std[valid], kind="stable")[:100]]
        for idx in ranked:
            shortlist.append({"profile": name, "predicted_ramping": pred[idx], "disagreement": std[idx],
                              "nearest_distance": dist[idx], "objective": pred[idx] + penalty * std[idx],
                              **dict(zip(FEATURES, pool[idx]))})
    pd.DataFrame(shortlist).to_csv(args.out / "candidate_shortlist.csv", index=False)
    for name, info in candidates.items():
        print(f"{name}: prediction={info['predicted_ramping']:.4f}, disagreement={info['model_disagreement']:.4f}, "
              f"distance={info['nearest_training_distance']:.4f}, measured={info['measured_ramping']}", flush=True)


def predict(args):
    _, X, _, _, _ = init_run(args.out, args.data)
    bundle = load_bundle(args)
    weights = validate_weights(read_json(args.weights))
    pred, std, mean, members = predict_batches(bundle, weights)
    dist = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(X).kneighbors(weights)[0][:, 0]
    result = {"predicted_ramping": pred.tolist(), "review_mean": mean.tolist(),
              "model_disagreement": std.tolist(), "nearest_training_distance": dist.tolist(),
              "individual_predictions": {k: v.tolist() for k, v in members.items()},
              "score_type": "prediction, not a new simulator measurement"}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, action in (("benchmark", benchmark), ("train", train), ("optimize", optimize), ("predict", predict)):
        cmd = sub.add_parser(name)
        cmd.add_argument("--data", type=Path, default=LAB / "dataset_same8/train.csv")
        cmd.add_argument("--out", type=Path, default=HERE / "results")
        cmd.set_defaults(action=action)
        if name == "benchmark":
            cmd.add_argument("--tune-minutes", type=float, default=60)
            cmd.add_argument("--validation-minutes", type=float, default=20)
            cmd.add_argument("--trial-scale", type=float, default=1.)
        if name == "optimize":
            cmd.add_argument("--candidates", type=int, default=100000)
            cmd.add_argument("--local-rounds", type=int, default=20)
            cmd.add_argument("--search-minutes", type=float, default=20)
            cmd.add_argument("--seed", type=int, default=2026)
        if name == "predict":
            cmd.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.data = args.data.resolve()
    if args.command == "benchmark" and (args.tune_minutes <= 0 or args.validation_minutes <= 0 or not 0 < args.trial_scale <= 1):
        parser.error("Positive budgets and 0 < trial-scale <= 1 required")
    if args.command == "optimize" and (args.candidates < 8 or args.local_rounds < 0 or args.search_minutes <= 0):
        parser.error("At least 8 candidates, nonnegative rounds and a positive search budget required")
    args.action(args)


if __name__ == "__main__":
    main()
