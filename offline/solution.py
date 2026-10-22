"""Unified teacher-baseline prediction, incumbent-preserving search and comparison.

The teacher's predict() is a surrogate, NOT the real simulator/evaluator.
Keep its original weights unless a feasible candidate actually improves its score.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

import offline as core

HERE, LAB = core.HERE, core.LAB
DATA = LAB / "dataset_same8/train.csv"
TOL = 1e-6


def teacher_score(weights):
    """The ONLY primary comparison score. Directly calls the unmodified teacher API."""
    X = core.validate_weights(weights)
    result = np.asarray(core.baseline.predict(X, LAB / "model.npz"), dtype=float).reshape(-1)
    if result.shape != (len(X),) or not np.isfinite(result).all():
        raise ValueError("Invalid prediction from the teacher baseline")
    return result


def preserve_incumbent(original, candidates, tolerance=TOL):
    original = core.validate_weights(original)[0]
    candidates = core.validate_weights(candidates)
    reference = float(teacher_score(original)[0])
    scores = teacher_score(candidates)
    best = int(np.argmin(scores))
    if scores[best] < reference - tolerance:
        return candidates[best].copy(), True
    return original.copy(), False


def optimize_teacher(original, X, y, starts=512, seed=20261022):
    """Multi-start constrained refinement; no claim of certified global optimality."""
    original = core.validate_weights(original)[0]
    rng = np.random.default_rng(seed)
    with np.load(LAB / "model.npz", allow_pickle=False) as saved:
        c, a, H = float(saved["offset"]), saved["linear"], saved["hessian"]
    A = np.zeros((3, 21))
    for b in range(3):
        A[b, 7 * b:7 * b + 7] = 1
    points = np.vstack([original, X[np.argsort(y, kind="stable")[:min(50, len(X))]],
                        rng.dirichlet(np.full(7, .3), size=(starts, 3)).reshape(-1, 21)])
    fun = lambda w: float(core.baseline.predict_quadratic(w, c, a, H))
    jac = lambda w: a + H @ w
    # Verify that the numerical optimizer uses exactly the function exposed by predict().
    np.testing.assert_allclose(core.baseline.predict_quadratic(points[:16], c, a, H),
                               teacher_score(points[:16]), rtol=1e-12, atol=1e-10)
    solutions = [original]
    stats = {"starts": len(points), "solver_successes": 0, "feasible_results": 0,
             "method": "SLSQP; bounds [0,1], three per-building sum equalities"}
    for w in points:
        fit = minimize(fun, w, jac=jac, method="SLSQP", bounds=[(0., 1.)] * 21,
                       constraints={"type": "eq", "fun": lambda v: A @ v - 1,
                                    "jac": lambda v: A},
                       options={"ftol": 1e-10, "maxiter": 400})
        stats["solver_successes"] += int(fit.success)
        if np.isfinite(fit.x).all() and np.min(fit.x) >= -1e-7 and np.max(np.abs(A @ fit.x - 1)) <= 1e-7:
            solutions.append(core.baseline.project_weights(fit.x))
            stats["feasible_results"] += 1
    solutions = np.asarray(solutions)
    selected, improved = preserve_incumbent(original, solutions)
    stats.update({"reference_teacher_prediction": float(teacher_score(original)[0]),
                  "best_search_teacher_prediction": float(teacher_score(solutions).min()),
                  "improved_beyond_tolerance": improved, "improvement_tolerance": TOL,
                  "note": "A multi-start search is not a global optimality proof."})
    return selected, stats


def near_incumbent_samples(incumbent, anchors, count, seed):
    """Search both the baseline neighborhood and connections to data-supported solutions."""
    rng = np.random.default_rng(seed)
    incumbent = core.validate_weights(incumbent)[0]
    anchors = core.validate_weights(anchors)
    n = count // 2
    chosen = anchors[rng.integers(len(anchors), size=n)]
    amount = np.exp(rng.uniform(np.log(1e-5), np.log(.4), size=(n, 1)))
    interpolated = (1 - amount) * incumbent + amount * chosen
    sigma = rng.choice([.002, .01, .03, .08], size=(count - n, 1))
    jitter = core.baseline.project_weights(incumbent + rng.normal(size=(count - n, 21)) * sigma)
    return core.validate_weights(np.vstack([incumbent, interpolated, jitter]))


def load_auxiliary():
    bundle_path = HERE / "results/model.joblib"
    if not bundle_path.exists():
        return None
    bundle = core.joblib.load(bundle_path)
    if bundle["data_sha256"] != core.sha(DATA) or bundle["features"] != core.FEATURES:
        raise ValueError("Auxiliary model does not match the input data")
    return bundle


def candidate_record(name, weights, X, y, ids, bundle=None):
    w = core.validate_weights(weights)
    if len(w) != 1:
        raise ValueError("Each comparison entry must contain exactly one 3x7 scheme")
    matches = np.flatnonzero(np.all(X == w, axis=1))
    record = {"name": name, "teacher_prediction": float(teacher_score(w)[0]),
              "measured_ramping": float(y[matches[0]]) if len(matches) else None,
              "measured_candidate_id": str(ids[matches[0]]) if len(matches) else None,
              "official_evaluator_score": None,
              "learned_model_prediction": None, "auxiliary_model_disagreement": None,
              "nearest_training_distance": float(NearestNeighbors(n_neighbors=1, algorithm="brute").fit(X).kneighbors(w)[0][0, 0]),
              "weights": w.reshape(3, 7).tolist()}
    if bundle is not None:
        pred, std, _, _ = core.predict_batches(bundle, w)
        record.update({"learned_model_prediction": float(pred[0]), "auxiliary_model_disagreement": float(std[0])})
    return record


def inputs(out):
    paths = [("teacher_original", LAB / "weights.json"),
             ("previous_conservative", HERE / "results/weights_conservative.json"),
             ("previous_aggressive", HERE / "results/weights_aggressive.json"),
             ("measured_best", HERE / "results/weights_measured_best.json"),
             ("recommended", out / "weights_recommended.json"),
             ("experimental_challenger", out / "weights_challenger.json")]
    return [(name, path) for name, path in paths if path.exists()]


def compare(out):
    ids, X, y = core.load_data(DATA)
    bundle = load_auxiliary()
    records = [candidate_record(name, core.read_json(path), X, y, ids, bundle) for name, path in inputs(out)]
    reference = next(r["teacher_prediction"] for r in records if r["name"] == "teacher_original")
    for r in records:
        r["teacher_prediction_delta"] = r["teacher_prediction"] - reference
        r["beats_teacher_original_prediction"] = r["teacher_prediction_delta"] < -TOL
    core.write_json(out / "comparison.json", {
        "primary_score": "teacher baseline.predict using original Lab6/model.npz; lower is better",
        "official_evaluator_available": False,
        "warning": "Teacher prediction and learned-model prediction are estimates. Neither is a new simulator measurement.",
        "records": records})
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "weights"} for r in records])
    table.to_csv(out / "comparison.csv", index=False)
    print(table[["name", "teacher_prediction", "teacher_prediction_delta", "learned_model_prediction", "measured_ramping"]].to_string(index=False))
    return records


def optimize(args):
    args.out.mkdir(parents=True, exist_ok=True)
    before = core.fingerprint()
    ids, X, y = core.load_data(DATA)
    original = core.validate_weights(core.read_json(LAB / "weights.json"))[0]
    primary, stats = optimize_teacher(original, X, y, args.starts, args.seed)
    if stats["improved_beyond_tolerance"]:
        core.write_json(args.out / "weights_recommended.json", primary.reshape(3, 7).tolist())
    else:
        # Keep the original bytes, avoiding a fake improvement caused by rounding.
        shutil.copyfile(LAB / "weights.json", args.out / "weights_recommended.json")
    bundle = load_auxiliary()
    challenger_stats = {"available": False}
    challenger_path = args.out / "weights_challenger.json"
    if bundle is not None:
        anchor_paths = [p for name, p in inputs(args.out) if name.startswith("previous_")]
        anchors = np.vstack([X[np.argsort(y, kind="stable")[:50]],
                             *[core.validate_weights(core.read_json(p)) for p in anchor_paths]])
        pool = near_incumbent_samples(primary, anchors, args.candidates, args.seed)
        teacher = teacher_score(pool)
        limit = float(teacher_score(primary)[0]) + abs(float(teacher_score(primary)[0])) * args.teacher_slack
        eligible = np.flatnonzero(teacher <= limit + 1e-10)
        eligible = eligible[np.linalg.norm(pool[eligible] - primary, axis=1) > 1e-3]
        if len(eligible):
            pred, std, _, _ = core.predict_batches(bundle, pool[eligible])
            base_pred, base_std, _, _ = core.predict_batches(bundle, primary.reshape(1, 21))
            objective = pred + std
            idx = int(np.argmin(objective))
            if objective[idx] < base_pred[0] + base_std[0] - .1:
                w = pool[eligible[idx]]
                core.write_json(challenger_path, w.reshape(3, 7).tolist())
                challenger_stats = {"available": True, "teacher_prediction_limit": limit,
                                    "allowed_teacher_prediction_slack_fraction": args.teacher_slack,
                                    "teacher_prediction": float(teacher[eligible[idx]]),
                                    "auxiliary_objective": float(objective[idx]),
                                    "role": "Experimental tradeoff; does NOT replace the primary recommendation or claim to beat baseline."}
        if not challenger_stats["available"] and challenger_path.exists():
            challenger_path.unlink()  # Remove an obsolete candidate from a prior run.
    records = compare(args.out)
    assert core.fingerprint() == before, "Teacher files changed"
    recommended = next(r for r in records if r["name"] == "recommended")
    assert recommended["teacher_prediction_delta"] <= TOL
    manifest = {"source_sha256": core.sha(__file__), "data_sha256": core.sha(DATA),
                "teacher_files_sha256": before, "seed": args.seed,
                "optimization": stats, "challenger": challenger_stats,
                "score_type": "teacher_model_prediction", "official_evaluations_used": 0,
                "recommendation": "improved_teacher_prediction" if stats["improved_beyond_tolerance"] else "retain_teacher_original",
                "reason": "Do not replace the original with a worse teacher prediction; auxiliary fit quality is not the project grade.",
                "previous_validation": "Existing model/check results are retained. No holdout refit, retuning or repeated validation.",
                "auxiliary_model_sha256": core.sha(HERE / "results/model.joblib") if bundle else None}
    core.write_json(args.out / "run.json", manifest)
    print("Recommendation:", manifest["recommendation"], flush=True)
    print("Official simulator score remains unknown.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("optimize", "compare", "score"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--out", type=Path, default=HERE / "revised_results")
        if name == "optimize":
            cmd.add_argument("--starts", type=int, default=512)
            cmd.add_argument("--candidates", type=int, default=50000)
            cmd.add_argument("--seed", type=int, default=20261022)
            cmd.add_argument("--teacher-slack", type=float, default=.05)
        elif name == "score":
            cmd.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.command == "optimize":
        if args.starts < 1 or args.candidates < 2 or not 0 <= args.teacher_slack <= 1:
            parser.error("Positive starts, at least 2 candidates and 0 <= teacher-slack <= 1 required")
        optimize(args)
    elif args.command == "compare":
        compare(args.out)
    else:
        ids, X, y = core.load_data(DATA)
        r = candidate_record(str(args.weights), core.read_json(args.weights), X, y, ids, load_auxiliary())
        r.pop("weights")
        r["score_type"] = "teacher_model_prediction; official simulator score unknown"
        print(json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
