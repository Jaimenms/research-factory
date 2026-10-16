"""Rede multicamada (MLP) em JAX: camadas totalmente ligadas, todas as escondidas com o mesmo número de
neurónios e a mesma ativação (sigmóide, tangente hiperbólica ou ReLU), saída linear; entradas e saídas
padronizadas pela média e desvio padrão do conjunto de treino. Inicialização uniforme ±1/sqrt(fan_in)
(a do PyTorch, usado na dissertação). `MLP` é o modelo empírico treinado: `predict` em unidades
físicas, `save`/`load` em JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ACTIVATIONS = ("relu", "sigmoid", "tanh", "identity")


@dataclass(frozen=True)
class Hyperparameters:
    neurons: int = 20
    layers: int = 1
    activation: str = "relu"
    learning_rate: float = 0.01
    epochs: int = 2000
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def sizes(self, n_in: int, n_out: int):
        return [n_in] + [self.neurons] * self.layers + [n_out]


def activation_fn(name: str):
    import jax
    import jax.numpy as jnp

    return {"relu": jax.nn.relu, "sigmoid": jax.nn.sigmoid, "tanh": jnp.tanh, "identity": lambda x: x}[name]


def init_params(key, sizes):
    """Lista de (W (fan_in, fan_out), b (fan_out,)) ~ U(−1/sqrt(fan_in), 1/sqrt(fan_in))."""
    import jax
    import jax.numpy as jnp

    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, k1, k2 = jax.random.split(key, 3)
        bound = 1.0 / np.sqrt(fan_in)
        W = jax.random.uniform(k1, (fan_in, fan_out), minval=-bound, maxval=bound)
        b = jax.random.uniform(k2, (fan_out,), minval=-bound, maxval=bound)
        params.append((W, b))
    return params


def forward(params, x, activation: str):
    """x (n, n_in) padronizado -> (n, n_out) padronizado."""
    act = activation_fn(activation)
    for W, b in params[:-1]:
        x = act(x @ W + b)
    W, b = params[-1]
    return x @ W + b


@dataclass
class MLP:
    """Modelo empírico treinado: hiperparâmetros, pesos (numpy) e padronização."""

    hp: Hyperparameters
    params: list  # [(W, b), ...] em numpy
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    input_names: tuple = ("P_in_bar", "T_in_K", "m_kg_s")
    output_names: tuple = ("P_out_bar", "T_out_K")
    meta: dict = field(default_factory=dict)

    @property
    def n_parameters(self) -> int:
        return int(sum(W.size + b.size for W, b in self.params))

    def predict(self, X) -> np.ndarray:
        """X (n, n_in) ou (n_in,) em unidades físicas -> (n, n_out) em unidades físicas (numpy puro)."""
        X = np.atleast_2d(np.asarray(X, float))
        act = {
            "relu": lambda v: np.maximum(v, 0.0),
            "sigmoid": lambda v: 1.0 / (1.0 + np.exp(-v)),
            "tanh": np.tanh,
            "identity": lambda v: v,
        }[self.hp.activation]
        h = (X - self.x_mean) / self.x_std
        for W, b in self.params[:-1]:
            h = act(h @ W + b)
        W, b = self.params[-1]
        return (h @ W + b) * self.y_std + self.y_mean

    def to_dict(self) -> dict:
        return {
            "format": "cmfsim-surrogate-mlp/1",
            "hyperparameters": asdict(self.hp),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "normalization": {
                "x_mean": self.x_mean.tolist(),
                "x_std": self.x_std.tolist(),
                "y_mean": self.y_mean.tolist(),
                "y_std": self.y_std.tolist(),
            },
            "layers": [{"W": np.asarray(W).tolist(), "b": np.asarray(b).tolist()} for W, b in self.params],
            "meta": self.meta,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1) + "\n")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "MLP":
        if d.get("format") != "cmfsim-surrogate-mlp/1":
            raise ValueError(f"formato desconhecido: {d.get('format')}")
        n = d["normalization"]
        return cls(
            hp=Hyperparameters(**d["hyperparameters"]),
            params=[(np.asarray(l["W"], float), np.asarray(l["b"], float)) for l in d["layers"]],
            x_mean=np.asarray(n["x_mean"], float),
            x_std=np.asarray(n["x_std"], float),
            y_mean=np.asarray(n["y_mean"], float),
            y_std=np.asarray(n["y_std"], float),
            input_names=tuple(d["input_names"]),
            output_names=tuple(d["output_names"]),
            meta=dict(d.get("meta", {})),
        )

    @classmethod
    def load(cls, path) -> "MLP":
        return cls.from_dict(json.loads(Path(path).read_text()))
