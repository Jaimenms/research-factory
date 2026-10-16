"""Amostragem do espaço de entradas: hipercubo latino próprio (a dissertação usou o da biblioteca
SMT, Bouhlel et al. 2019), com os critérios «random», «center» e «maximin» (melhor de `n_candidates`
sorteios pela distância mínima entre pontos)."""

from __future__ import annotations

import numpy as np


def _unit_lhs(rng, n, d, center):
    u = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        u[:, j] = (perm + (0.5 if center else rng.random(n))) / n
    return u


def _min_distance(u):
    from scipy.spatial.distance import pdist

    return float(pdist(u).min()) if len(u) > 1 else np.inf


def latin_hypercube(n: int, bounds, seed=0, criterion="random", n_candidates=20) -> np.ndarray:
    """`n` pontos em `bounds` (d, 2) [[mín, máx], ...]; devolve (n, d). Cada coordenada visita cada um dos
    `n` estratos exatamente uma vez."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("bounds deve ser (d, 2) com máximo > mínimo")
    d = bounds.shape[0]
    if criterion == "maximin":
        best, best_d = None, -1.0
        for _ in range(n_candidates):
            u = _unit_lhs(rng, n, d, center=False)
            dm = _min_distance(u)
            if dm > best_d:
                best, best_d = u, dm
        u = best
    elif criterion in ("random", "center"):
        u = _unit_lhs(rng, n, d, center=criterion == "center")
    else:
        raise ValueError(f"critério desconhecido: {criterion}")
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
