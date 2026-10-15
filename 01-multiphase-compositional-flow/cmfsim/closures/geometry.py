"""Copiado de mfsim/geometry.py (doutorado, 2026-10-14, commit 535e6f8) em 2026-10-15; alterações: só `stratified_level` e `alpha_from_level` (o traçado do duto fica em `cmfsim/pipe.py`).

Geometria do estratificado em seção circular (eqs. 43–48 da dissertação): h/D e ângulo molhado a partir da fração de área de gás.
"""

from __future__ import annotations

import numpy as np

# Polinômios (calcula_nivel_estratificado.m) usados como estimativa inicial
_P_H = np.array(
    [
        0.00000000000197,
        -3.66048527474641,
        9.15121318685793,
        -8.89104962298284,
        4.18536124761860,
        -1.75567780293538,
        0.98531913309307,
    ]
)
_P_BETA = 1.0e2 * np.array(
    [
        0.00000000000056,
        -0.67451330764088,
        1.68628326909991,
        -1.58733054088785,
        0.69471254223251,
        -0.17280444315654,
        0.05824216671205,
    ]
)


def stratified_level(alphaG, tol=1e-9, max_iter=100, bk=None):
    """Altura adimensional de líquido hD=h/D e ângulo molhado beta para a
    fração de área de gás alphaG numa seção circular.

    Resolve  sin(beta) - beta + 2*pi*alphaL = 0  por Newton (vetorizado),
    partindo da aproximação polinomial do código original. O critério de saída é
    global (todos os pontos intermédios com |passo| <= tol). No backend numpy o
    Newton corre só nos pontos intermédios (subconjunto, como o código original: as
    funções transcendentais do numpy podem diferir em ulps entre o laço vetorial e o
    escalar, logo o subconjunto preserva os resultados bit a bit); no JAX corre em
    todos os pontos com `while_loop` e o mesmo critério.
    """
    from ..backend import NUMPY

    bk = bk or NUMPY
    xp = bk.xp
    if not bk.is_jax:
        alphaG = np.asarray(alphaG, dtype=float)
        alphaL = 1.0 - alphaG
        beta = np.polyval(_P_BETA, alphaG)
        beta = np.clip(beta, 0.0, 2 * np.pi)
        only_liq = alphaL > 1 - 1e-6
        only_gas = alphaL < 1e-6
        mid = ~(only_liq | only_gas)
        b = beta.copy()
        b[only_liq] = 2 * np.pi * alphaL[only_liq]
        b[only_gas] = 2 * np.pi * alphaG[only_gas]
        bm = b[mid]
        aL = alphaL[mid]
        for _ in range(max_iter):
            F0 = np.sin(bm) - bm + 2 * np.pi * aL
            dF = np.cos(bm) - 1.0
            dF = np.where(np.abs(dF) < 1e-14, -1e-14, dF)
            step = F0 / dF
            bm = bm - step
            if np.all(np.abs(step) <= tol):
                break
        b[mid] = bm
        hD = 0.5 * (1.0 - np.cos(b / 2.0))
        return hD, b
    alphaL = 1.0 - alphaG
    beta = xp.clip(xp.polyval(xp.asarray(_P_BETA), alphaG), 0.0, 2 * xp.pi)
    only_liq = alphaL > 1 - 1e-6
    only_gas = alphaL < 1e-6
    mid = ~(only_liq | only_gas)
    b0 = xp.where(only_liq, 2 * xp.pi * alphaL, xp.where(only_gas, 2 * xp.pi * alphaG, beta))

    def body(c):
        bm, _, i = c
        F0 = xp.sin(bm) - bm + 2 * xp.pi * alphaL
        dF = xp.cos(bm) - 1.0
        dF = xp.where(xp.abs(dF) < 1e-14, -1e-14, dF)
        step = xp.where(mid, F0 / dF, 0.0)
        return bm - step, step, i + 1

    def cond(c):
        return (c[2] < max_iter) & ~xp.all(xp.abs(c[1]) <= tol)

    bm, _, _ = bk.while_loop(cond, body, (b0, xp.full(b0.shape, xp.inf), 0))
    b = xp.where(mid, bm, b0)
    hD = 0.5 * (1.0 - xp.cos(b / 2.0))
    return hD, b


def alpha_from_level(hD):
    """Fração de área de gás a partir de h/D (estratificado)."""
    beta = 2.0 * np.arccos(1.0 - 2.0 * np.asarray(hD, dtype=float))
    return 1.0 - (beta - np.sin(beta)) / (2 * np.pi), beta
