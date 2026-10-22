"""Small-data tabular neural architectures; inputs are raw (batch,21) weights.

Fit preprocessing only on a training fold. All forwards return original ramping
units and retain autograd with respect to the original 21 control weights.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import math

import torch
from torch import nn


@dataclass(frozen=True)
class NetworkConfig:
    name: str
    kind: str
    hidden: tuple = (64, 32)
    width: int = 64
    bottleneck: int = 32
    blocks: int = 2
    activation: str = "silu"
    representation: str = "helmert18"
    dropout: float = 0.0


CONFIGS = {
    "mlp_small": NetworkConfig("mlp_small", "mlp", (64, 32)),
    "mlp_medium": NetworkConfig("mlp_medium", "mlp", (128, 64)),
    "mlp_large": NetworkConfig("mlp_large", "mlp", (256, 256, 128)),
    "resnet": NetworkConfig("resnet", "resnet"),
    "building_mlp": NetworkConfig("building_mlp", "building"),
    "quadratic_residual": NetworkConfig("quadratic_residual", "quadratic_residual", (64, 32)),
}


def helmert_basis(dtype=torch.float64):
    """7x6 orthonormal basis of the sum-zero plane; no data are used."""
    basis = torch.zeros(7, 6, dtype=dtype)
    for j in range(6):
        denom = math.sqrt((j + 1) * (j + 2))
        basis[:j + 1, j] = 1 / denom
        basis[j + 1, j] = -(j + 1) / denom
    return basis


def activation(name):
    choices = {"silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU}
    if name not in choices:
        raise ValueError("activation must be silu, tanh or relu")
    return choices[name]()


def mlp(dims, act, dropout):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(d_in, d_out))
        if i < len(dims) - 2:
            layers.append(activation(act))
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, width, bottleneck, act, dropout):
        super().__init__()
        self.act = activation(act)
        self.branch = nn.Sequential(nn.Linear(width, bottleneck), activation(act),
                                    nn.Dropout(dropout), nn.Linear(bottleneck, width))

    def forward(self, x):
        return x + 0.1 * self.branch(self.act(x))


class ResidualBody(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.input = nn.Linear(dim, config.width)
        self.blocks = nn.Sequential(*[ResidualBlock(config.width, config.bottleneck,
                                                    config.activation, config.dropout)
                                      for _ in range(config.blocks)])
        self.act = activation(config.activation)
        self.output = nn.Linear(config.width, 1)

    def forward(self, x):
        return self.output(self.act(self.blocks(self.input(x))))


class BuildingBody(nn.Module):
    """Three independent encoders: building identities are not interchangeable."""
    def __init__(self, config):
        super().__init__()
        self.encoders = nn.ModuleList([mlp([6, 32, 16], config.activation, config.dropout)
                                       for _ in range(3)])
        self.fusion = mlp([48, 64, 32, 1], config.activation, config.dropout)

    def forward(self, x):
        encoded = [net(x[:, 6 * b:6 * b + 6]) for b, net in enumerate(self.encoders)]
        return self.fusion(torch.cat(encoded, dim=-1))


class Surrogate(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.kind not in {"mlp", "resnet", "building", "quadratic_residual"}:
            raise ValueError("Unknown architecture kind")
        if config.representation not in {"helmert18", "raw21"}:
            raise ValueError("representation must be helmert18 or raw21")
        if config.kind == "building" and config.representation != "helmert18":
            raise ValueError("building_mlp requires three groups of six Helmert coordinates")
        if not 0 <= config.dropout < 1 or any(d <= 0 for d in config.hidden):
            raise ValueError("Invalid dropout or hidden dimensions")
        self.config = config
        dim = 18 if config.representation == "helmert18" else 21
        self.register_buffer("basis", helmert_basis())
        self.register_buffer("feature_mean", torch.zeros(dim))
        self.register_buffer("feature_scale", torch.ones(dim))
        self.register_buffer("target_mean", torch.zeros(()))
        self.register_buffer("target_scale", torch.ones(()))
        self.register_buffer("quad_offset", torch.zeros(()))
        self.register_buffer("quad_linear", torch.zeros(21))
        self.register_buffer("quad_hessian", torch.zeros(21, 21))
        if config.kind in {"mlp", "quadratic_residual"}:
            self.network = mlp([dim, *config.hidden, 1], config.activation, config.dropout)
        elif config.kind == "resnet":
            self.network = ResidualBody(dim, config)
        else:
            self.network = BuildingBody(config)
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if config.kind == "quadratic_residual":
            # Start exactly at the fitted quadratic; initially the correction is zero.
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def coordinates(self, weights):
        if weights.ndim != 2 or weights.shape[1] != 21:
            raise ValueError("Expected raw control weights with shape (batch,21)")
        if self.config.representation == "raw21":
            return weights
        return ((weights.reshape(-1, 3, 7) - 1 / 7) @ self.basis).reshape(-1, 18)

    def quadratic(self, weights):
        return self.quad_offset + weights @ self.quad_linear + 0.5 * (weights * (weights @ self.quad_hessian)).sum(-1)

    def forward(self, weights):
        z = (self.coordinates(weights) - self.feature_mean) / self.feature_scale
        correction = self.target_scale * self.network(z).squeeze(-1)
        if self.config.kind == "quadratic_residual":
            return self.quadratic(weights) + correction
        return self.target_mean + correction

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def validate_training_data(X, y):
    X = torch.as_tensor(X).detach().to(device="cpu", dtype=torch.float64)
    y = torch.as_tensor(y).detach().to(device="cpu", dtype=torch.float64)
    if X.ndim != 2 or X.shape[1] != 21 or y.ndim != 1 or len(X) != len(y) or len(y) < 2:
        raise ValueError("Need at least two rows of (N,21) weights and (N,) targets")
    if not bool(torch.isfinite(X).all() and torch.isfinite(y).all()) or bool((X < 0).any() or (y < 0).any()):
        raise ValueError("Training values must be finite and nonnegative")
    if not torch.allclose(X.reshape(-1, 3, 7).sum(-1), torch.ones(len(X), 3, dtype=X.dtype), atol=1e-6, rtol=0):
        raise ValueError("Each building's seven weights must sum to one")
    return X, y


def make_surrogate(name, X_train, y_train, *, quadratic=None, activation_name="silu",
                   representation="helmert18", dropout=0., dtype=torch.float32):
    """Caller supplies TRAINING-fold data and optionally its fitted quadratic (c,a,H).

    No teacher model file, validation targets, or full-data fitted preprocessing
    are read here. All learned input/target statistics become frozen buffers.
    """
    if name not in CONFIGS:
        raise ValueError("Unknown network: " + name)
    X, y = validate_training_data(X_train, y_train)
    config = replace(CONFIGS[name], activation=activation_name, representation=representation, dropout=dropout)
    # Fit statistics in float64 before converting the fully configured model.
    model = Surrogate(config).to(dtype=torch.float64)
    with torch.no_grad():
        z = model.coordinates(X)
        scale = z.std(0, unbiased=False)
        model.feature_mean.copy_(z.mean(0))
        model.feature_scale.copy_(torch.where(scale >= 1e-6, scale, torch.ones_like(scale)))
        model.target_mean.copy_(y.mean())
        model.target_scale.copy_(y.std(unbiased=False).clamp_min(1e-6))
        if config.kind == "quadratic_residual":
            if quadratic is None:
                raise ValueError("quadratic_residual needs coefficients fitted only on the current training fold")
            c, a, H = [torch.as_tensor(v, dtype=torch.float64) for v in quadratic]
            if c.ndim != 0 or a.shape != (21,) or H.shape != (21, 21):
                raise ValueError("Expected scalar c, (21,) a and (21,21) H")
            if not all(bool(torch.isfinite(v).all()) for v in (c, a, H)) or not torch.allclose(H, H.T):
                raise ValueError("Quadratic coefficients must be finite with symmetric Hessian")
            model.quad_offset.copy_(c)
            model.quad_linear.copy_(a)
            model.quad_hessian.copy_(H)
    return model.to(dtype=dtype)


def save_surrogate(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "config": asdict(model.config),
                "dtype": str(model.feature_mean.dtype).removeprefix("torch."),
                "state_dict": model.state_dict()}, path)


def load_surrogate(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported neural checkpoint format")
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(payload.get("dtype"))
    if dtype is None:
        raise ValueError("Unsupported checkpoint dtype")
    model = Surrogate(NetworkConfig(**payload["config"])).to(dtype=dtype)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()
