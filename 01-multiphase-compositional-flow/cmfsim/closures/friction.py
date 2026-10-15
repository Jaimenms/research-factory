"""Copiado de mfsim/friction.py (doutorado, 2026-10-14, commit 535e6f8) em 2026-10-15; alterações: nenhuma (Churchill = eqs. 50–52 da dissertação, em fator de Fanning).

Fatores de atrito de Fanning (uma implementação para numpy e JAX via `bk`)."""

from __future__ import annotations

from ..backend import NUMPY

# Regularização de |v| em torno de zero (m/s): torna a tensão de cisalhamento
# tau = f(Re) rho v |v| / 2 suave em v = 0 (no regime laminar tau -> 8 mu v / D,
# linear em v, em vez de ter derivada descontínua). Para |v| >> V_EPS o efeito
# é desprezível (erro relativo ~ (V_EPS/v)^2 / 2).
V_EPS = 1e-4


def vabs(v, bk=NUMPY):
    """|v| regularizado: sqrt(v^2 + V_EPS^2)."""
    return bk.xp.sqrt(v**2 + V_EPS**2)


def fanning(v, rho, mu, D, eps, correlation="churchill", bk=NUMPY):
    """Fator de atrito de Fanning e número de Reynolds (fator.m).

    Com V_EPS = 0 retorna f = 0 em v == 0 (como no código original); com V_EPS > 0
    (o caso) o atrito é regularizado e nunca se anula. D pode ser um diâmetro
    hidráulico (array).
    """
    xp = bk.xp
    Dd = xp.where(D <= 0, 1e-12, D)
    if V_EPS > 0:
        Re = Dd * vabs(v, bk) * rho / mu
    else:  # pragma: no cover - ramo do código original
        zero = v == 0
        vv = xp.where(zero, 1e-6, v)
        Re = Dd * xp.abs(vv) * rho / mu
    if correlation == "churchill":
        A = (2.457 * xp.log(1.0 / (0.27 * eps / Dd + (7.0 / Re) ** 0.9))) ** 16
        B = (37530.0 / Re) ** 16
        f = 2.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)
    elif correlation == "haaland":
        f = 0.07716 / (xp.log10(6.9 / Re + (eps / Dd / 3.7) ** 1.11)) ** 2
    elif correlation == "laminar":
        f = 16.0 / Re
    elif correlation == "blasius":
        f = 0.046 * Re ** (-0.2)
    else:
        raise ValueError(correlation)
    if V_EPS <= 0:  # pragma: no cover
        f = xp.where(zero, 0.0, f)
        Re = xp.where(zero, 0.0, Re)
    return f, Re
