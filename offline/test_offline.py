"""Checks for validation isolation, checkpointing and constrained search."""
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import KFold

import offline as app


class MeanEstimator:
    def fit(self, X, y):
        self.mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean)


class OfflineTests(unittest.TestCase):
    def test_split_is_fixed_disjoint_and_covers_all_rows(self):
        dev, check = app.split_indices(2048)
        self.assertEqual((len(dev), len(check)), (1638, 410))
        self.assertFalse(set(dev) & set(check))
        np.testing.assert_array_equal(np.sort(np.r_[dev, check]), np.arange(2048))
        np.testing.assert_array_equal(check, app.split_indices(2048)[1])

    def test_oof_never_fits_held_out_targets_and_cache_survives_deadline(self):
        X, y = np.arange(40).reshape(20, 2), np.arange(20.) ** 2
        spec = {"id": "toy", "family": "toy", "params": {}}
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            with patch.object(app, "make_model", side_effect=lambda *a: MeanEstimator()):
                _, pred = app.cross_validate(spec, X, y, 2026, cache, time.monotonic() + 10)
            for tr, va in KFold(5, shuffle=True, random_state=2026).split(X):
                np.testing.assert_array_equal(pred[va], np.repeat(y[tr].mean(), len(va)))
            with patch.object(app, "make_model", side_effect=AssertionError("cache was ignored")):
                _, resumed = app.cross_validate(spec, X, y, 2026, cache, 0)
            np.testing.assert_array_equal(resumed, pred)
            with self.assertRaises(app.BudgetExpired):
                app.cross_validate(spec, X, y, 2027, cache, 0)
            self.assertFalse((cache / "toy_seed2027.json").exists())

    def test_kernel_scaler_is_fitted_inside_each_model(self):
        X = np.arange(210.).reshape(10, 21)
        model = app.make_model({"family": "svr", "params": {"C": 1.}})
        self.assertFalse(hasattr(model[0], "mean_"))
        model.fit(X[:6], np.arange(6.))
        np.testing.assert_allclose(model[0].mean_, X[:6].mean(0))

    def test_selection_obeys_mae_gate_then_selection_quality(self):
        def row(key, mae, top, low):
            return {"id": key, "metrics": {"mae": mae, "top10_true_mean": top, "low20_mae": low}}
        rows = [row("best_mae", 5., 130., 8.), row("best_selection", 5.2, 122., 7.),
                row("outside_gate", 5.3, 110., 1.)]
        self.assertEqual(app.choose(rows)["id"], "best_selection")
        rows += [row("lowtail_tie", 5.21, 122.05, 6.)]
        self.assertEqual(app.choose(rows)["id"], "lowtail_tie")

    def test_candidates_and_local_search_are_legal_and_reproducible(self):
        rng = np.random.default_rng(2)
        X = rng.dirichlet(np.ones(7), size=(60, 3)).reshape(-1, 21)
        y = np.arange(len(X), dtype=float)
        a = app.propose_candidates(X, y, 1001, np.random.default_rng(2026))
        b = app.propose_candidates(X, y, 1001, np.random.default_rng(2026))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (1001, 21))
        app.validate_weights(a)
        for i in range(20):
            a = app.pair_transfers(a, rng, .5)
            app.validate_weights(a)

    def test_weight_validation_rejects_invalid_inputs(self):
        valid = np.full((3, 7), 1 / 7)
        self.assertEqual(app.validate_weights(valid).shape, (1, 21))
        for bad in (np.zeros((3, 7)), np.ones((3, 7)), np.ones((7, 3)), np.full((3, 7), np.nan)):
            with self.assertRaises(ValueError):
                app.validate_weights(bad)
        bad = valid.copy()
        bad[0, :2] = [-.1, .1 + 2 / 7]
        with self.assertRaises(ValueError):
            app.validate_weights(bad)

    def test_duplicate_combinations_are_rejected(self):
        X = np.full((5, 21), 1 / 7)
        with patch.object(app.baseline, "load_table", return_value=(list("abcde"), X, np.arange(5.))):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                app.load_data("unused")

    def test_internal_check_is_not_evaluated_twice(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            app.write_json(out / "internal_check.json", {"already": "recorded"})
            with patch.object(app, "fit_members", side_effect=AssertionError("refitted check")):
                app.final_check(out, None, None, None, None, {})

    def test_full_candidate_export_with_baseline_prediction(self):
        # Exercise the full export boundary using real baseline inference and a toy surrogate.
        rng = np.random.default_rng(77)
        X = rng.dirichlet(np.ones(7), size=(30, 3)).reshape(-1, 21)
        y = np.linspace(120, 150, len(X))
        bundle = {"models": {"toy": MeanEstimator().fit(X, y)},
                  "selection": {"selected": "toy", "review_specs": [{"id": "toy"}], "review_weights": [1.]}}
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(out=Path(d), data=Path('unused'), seed=2026,
                                   candidates=16, local_rounds=1, search_minutes=1)
            with patch.object(app, "init_run", return_value=(np.arange(len(X)).astype(str), X, y, None, None)), \
                 patch.object(app, "load_bundle", return_value=bundle):
                app.optimize(args)
            result = app.read_json(Path(d) / 'candidates.json')
            self.assertAlmostEqual(result['candidates']['original_baseline']['original_model_prediction'], 75.89570106874649)
            for name in ('conservative', 'aggressive', 'measured_best'):
                weights = app.read_json(Path(d) / f'weights_{name}.json')
                app.validate_weights(weights)


if __name__ == "__main__":
    unittest.main()
