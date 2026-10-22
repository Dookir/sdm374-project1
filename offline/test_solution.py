"""Regression tests for common scoring and incumbent preservation."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import solution as app


class SolutionTests(unittest.TestCase):
    def test_teacher_score_is_exactly_original_predict(self):
        rng = np.random.default_rng(42)
        X = rng.dirichlet(np.ones(7), size=(20, 3)).reshape(-1, 21)
        np.testing.assert_array_equal(app.teacher_score(X), app.core.baseline.predict(X, app.LAB / 'model.npz'))
        original = app.core.read_json(app.LAB / 'weights.json')
        self.assertAlmostEqual(app.teacher_score(original)[0], app.core.baseline.predict(original))

    def test_worse_candidates_cannot_replace_original(self):
        original = np.full(21, 1 / 7)
        worse = original.copy()
        worse[0] += .02
        worse[6] -= .02
        scoring = lambda w: app.core.validate_weights(w)[:, 0] * 100
        with patch.object(app, 'teacher_score', side_effect=scoring):
            selected, improved = app.preserve_incumbent(original, np.vstack([worse, original]))
        self.assertFalse(improved)
        np.testing.assert_array_equal(selected, original)

    def test_actual_improvement_is_accepted(self):
        original = np.full(21, 1 / 7)
        better = original.copy()
        better[0] -= .02
        better[6] += .02
        with patch.object(app, 'teacher_score', side_effect=lambda w: app.core.validate_weights(w)[:, 0] * 100):
            selected, improved = app.preserve_incumbent(original, np.vstack([original, better]))
        self.assertTrue(improved)
        np.testing.assert_array_equal(selected, better)

    def test_rounding_is_not_claimed_as_improvement(self):
        original = np.full(21, 1 / 7)
        candidate = original.copy()
        candidate[0] -= 1e-10
        candidate[6] += 1e-10
        with patch.object(app, 'teacher_score', side_effect=lambda w: app.core.validate_weights(w)[:, 0] * 100):
            selected, improved = app.preserve_incumbent(original, candidate.reshape(1, -1))
        self.assertFalse(improved)
        np.testing.assert_array_equal(selected, original)

    def test_neighborhood_sampling_reproducible_and_feasible_at_boundary(self):
        original = app.core.validate_weights(app.core.read_json(app.LAB / 'weights.json'))
        anchors = np.random.default_rng(7).dirichlet(np.ones(7), size=(30, 3)).reshape(-1, 21)
        a = app.near_incumbent_samples(original, anchors, 1001, 2026)
        b = app.near_incumbent_samples(original, anchors, 1001, 2026)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (1002, 21))
        app.core.validate_weights(a)

    def test_complete_export_has_common_scores_and_retains_original_bytes(self):
        before = app.core.fingerprint()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            args = SimpleNamespace(out=out, starts=3, candidates=128, seed=20261022, teacher_slack=.05)
            with patch.object(app, 'load_auxiliary', return_value=None), contextlib.redirect_stdout(io.StringIO()):
                app.optimize(args)
            self.assertEqual((out / 'weights_recommended.json').read_bytes(), (app.LAB / 'weights.json').read_bytes())
            result = app.core.read_json(out / 'comparison.json')
            for row in result['records']:
                self.assertIsNone(row['official_evaluator_score'])
                self.assertAlmostEqual(row['teacher_prediction'], app.teacher_score(row['weights'])[0])
            recommended = next(r for r in result['records'] if r['name'] == 'recommended')
            self.assertLessEqual(recommended['teacher_prediction_delta'], app.TOL)
            self.assertFalse(result['official_evaluator_available'])
        self.assertEqual(before, app.core.fingerprint())


if __name__ == '__main__':
    unittest.main()
