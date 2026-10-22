import tempfile
import unittest
from pathlib import Path
import torch
from .models import CONFIGS, Surrogate, make_surrogate, helmert_basis, save_surrogate, load_surrogate


def data(n=32):
    generator = torch.Generator().manual_seed(18)
    X = torch.rand(n, 3, 7, generator=generator, dtype=torch.float64) + .1
    X /= X.sum(-1, keepdim=True)
    return X.reshape(-1, 21), torch.linspace(120, 190, n, dtype=torch.float64)


def quadratic():
    a = torch.linspace(-1, 1, 21, dtype=torch.float64)
    H = torch.eye(21, dtype=torch.float64) * 2
    return torch.tensor(100., dtype=torch.float64), a, H


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_coordinates_preserve_simplex_tangent_geometry(self):
        Q = helmert_basis()
        torch.testing.assert_close(Q.T @ Q, torch.eye(6, dtype=Q.dtype))
        torch.testing.assert_close(Q.sum(0), torch.zeros(6, dtype=Q.dtype))
        X, y = data()
        model = make_surrogate("mlp_small", X, y, dtype=torch.float64)
        d = (X[0] - X[1]).reshape(3, 7)
        torch.testing.assert_close((d @ Q).square().sum(), d.square().sum())
        shift = torch.tensor([.1, -.2, .3], dtype=X.dtype).repeat_interleave(7)
        torch.testing.assert_close(model.coordinates(X + shift), model.coordinates(X))

    def test_all_networks_have_batch_outputs_and_input_gradients(self):
        X, y = data()
        for name in CONFIGS:
            with self.subTest(name=name):
                model = make_surrogate(name, X, y, quadratic=quadratic(), dtype=torch.float64)
                probe = X[:2].clone().requires_grad_(True)
                self.assertEqual(model(probe).shape, (2,))
                self.assertEqual(model(probe[:1]).shape, (1,))
                g = torch.autograd.grad(model(probe).sum(), probe)[0]
                self.assertTrue(torch.isfinite(g).all())
                self.assertGreater(float(g.abs().sum()), 0)
                self.assertTrue(torch.autograd.gradcheck(model, (probe,), fast_mode=True))

    def test_scaler_uses_only_fitting_rows_and_stays_frozen(self):
        X, y = data()
        model = make_surrogate("mlp_small", X[:20], y[:20], dtype=torch.float64)
        expected = model.coordinates(X[:20]).mean(0)
        torch.testing.assert_close(model.feature_mean, expected)
        before = {k: v.clone() for k, v in model.named_buffers()}
        model.train()
        model(X[20:])
        for k, v in model.named_buffers():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        torch.testing.assert_close(model.target_mean, y[:20].mean())

    def test_quadratic_residual_starts_exactly_at_base_and_can_learn(self):
        X, y = data()
        c, a, H = quadratic()
        model = make_surrogate("quadratic_residual", X, y, quadratic=(c, a, H), dtype=torch.float64)
        expected = c + X @ a + .5 * (X * (X @ H)).sum(-1)
        torch.testing.assert_close(model(X), expected, rtol=0, atol=1e-12)
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for _ in range(2):
            optimizer.zero_grad()
            (((model(X) - y) / model.target_scale) ** 2).mean().backward()
            optimizer.step()
        self.assertFalse(torch.equal(model(X), expected))
        for k, v in model.named_buffers():
            torch.testing.assert_close(v, buffers[k], rtol=0, atol=0)
            self.assertFalse(v.requires_grad)

    def test_building_identity_is_preserved(self):
        X, y = data()
        torch.manual_seed(32)
        model = make_surrogate("building_mlp", X, y, dtype=torch.float64)
        swapped = X.reshape(-1, 3, 7)[:, [1, 0, 2]].reshape(-1, 21)
        self.assertFalse(torch.allclose(model(X), model(swapped)))
        params = [next(b.parameters()).data_ptr() for b in model.network.encoders]
        self.assertEqual(len(set(params)), 3)

    def test_constant_training_values_and_invalid_inputs(self):
        X = torch.full((5, 21), 1 / 7, dtype=torch.float64)
        y = torch.ones(5, dtype=torch.float64)
        m = make_surrogate("mlp_small", X, y, dtype=torch.float64)
        self.assertTrue(torch.isfinite(m(X)).all())
        for xx, yy in [(X, y[:2]), (X[:, :20], y), (torch.zeros_like(X), y), (X * float('nan'), y)]:
            with self.assertRaises(ValueError):
                make_surrogate("mlp_small", xx, yy)
        with self.assertRaises(ValueError):
            make_surrogate("quadratic_residual", X, y)
        with self.assertRaises(ValueError):
            make_surrogate("building_mlp", X, y, representation="raw21")
        with self.assertRaises(ValueError):
            m(torch.ones(3, 7, dtype=torch.float64))

    def test_initialization_and_checkpoint_roundtrip(self):
        X, y = data()
        for name in CONFIGS:
            with self.subTest(name=name):
                torch.manual_seed(42)
                a = make_surrogate(name, X, y, quadratic=quadratic(), dtype=torch.float64).eval()
                torch.manual_seed(42)
                b = make_surrogate(name, X, y, quadratic=quadratic(), dtype=torch.float64).eval()
                torch.testing.assert_close(a(X), b(X), rtol=0, atol=0)
                with tempfile.TemporaryDirectory() as d:
                    path = Path(d) / 'network.pt'
                    save_surrogate(a, path)
                    restored = load_surrogate(path)
                    self.assertEqual(restored.feature_mean.dtype, torch.float64)
                    torch.testing.assert_close(a(X), restored(X), rtol=0, atol=0)

    def test_activation_and_input_ablation_interfaces(self):
        X, y = data()
        for act in ('relu', 'tanh', 'silu'):
            for representation in ('raw21', 'helmert18'):
                model = make_surrogate('mlp_small', X, y, activation_name=act, representation=representation)
                self.assertTrue(torch.isfinite(model(X.float())).all())


if __name__ == '__main__':
    unittest.main()
