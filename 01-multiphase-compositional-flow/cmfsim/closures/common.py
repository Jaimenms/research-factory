"""Estruturas comuns aos fechamentos mecânicos (§2.4.1 da dissertação): entradas locais por ponto e
resultado de um fechamento (holdup, termo dissipativo, velocidades), mais a função degrau suave
usada na identificação de padrão."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from ..backend import NUMPY

G = 9.80665


@dataclass
class FlowInputs:
    """Variáveis locais (arrays (n,)): geometria, inclinação, propriedades de fase e velocidades
    superficiais. `A` e `v_M` são derivadas."""

    D: np.ndarray
    eps: np.ndarray
    theta: np.ndarray  # rad, positivo ascendente
    rho_L: np.ndarray
    rho_G: np.ndarray
    mu_L: np.ndarray
    mu_G: np.ndarray
    sigma: np.ndarray
    v_SL: np.ndarray
    v_SG: np.ndarray
    f_wall_mult: np.ndarray = (
        None  # multiplicador dos fatores de Churchill: 1 (Fanning) ou 4 (Darcy como impresso em (50))
    )

    @property
    def A(self):
        return np.pi * self.D**2 / 4.0

    @property
    def v_M(self):
        return self.v_SL + self.v_SG

    def names(self):
        return [f.name for f in fields(self)]

    @classmethod
    def build(cls, bk=NUMPY, **kw):
        """Constrói com arrays do backend, difundindo escalares para a forma de `v_SL`."""
        xp = bk.xp
        n = xp.asarray(kw["v_SL"], float).shape[0]
        kw.setdefault("f_wall_mult", 1.0)
        return cls(**{f.name: xp.broadcast_to(xp.asarray(kw[f.name], float), (n,)) for f in fields(cls)})


@dataclass
class ClosureResult:
    """Saída de um fechamento por ponto: holdup H_L, termo dissipativo total F = -dP/dx [Pa/m]
    (atrito + gravidade, sem aceleração; eqs. 42, 60, 80, 83), as partes de atrito e gravidade,
    velocidades de fase e um escalar característico do padrão (h_L/D, espessura do filme...)."""

    H_L: np.ndarray
    F: np.ndarray
    F_friction: np.ndarray
    F_gravity: np.ndarray
    v_L: np.ndarray
    v_G: np.ndarray
    level: np.ndarray

    def names(self):
        return [f.name for f in fields(self)]

    @classmethod
    def blend(cls, results, weights, bk=NUMPY):
        """Soma ponderada de resultados (pesos (n,) que somam 1)."""
        xp = bk.xp
        out = {}
        for f in fields(cls):
            out[f.name] = sum(w * getattr(r, f.name) for r, w in zip(results, weights))
        return cls(**out)


WALL_FRICTION_MULT = {"fanning": 1.0, "darcy_as_printed": 4.0}


def smooth_step(x, k, bk=NUMPY):
    """Degrau suave sigma(k x) (k = None: degrau duro (x > 0) como float). Com x já
    adimensional (relativo ao valor crítico), k ~ 20–100 dá transições de 2–10 %."""
    xp = bk.xp
    if k is None:
        return xp.where(x > 0.0, 1.0, 0.0)
    return bk.sigmoid(k * x)
