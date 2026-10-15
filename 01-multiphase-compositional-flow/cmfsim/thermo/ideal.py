"""Entalpia de gás ideal (148) com a forma de Cp de Smith et al. (2018) e a entalpia de formação a
298.15 K: H_gi = sum_i x_i [Hf_i + R (A (T - T0) + B/2 (T^2 - T0^2) + C/3 (T^3 - T0^3) - D (1/T - 1/T0))].
A dissertação escreve (148) sem os fatores 1/2, 1/3, -1 e R (notação abreviada); aqui está o
integral exato de Cp."""

from __future__ import annotations

from ..backend import NUMPY
from .components import R_GAS, T_REF, ComponentSet


def pure_h_ideal(T, cs: ComponentSet, bk=NUMPY):
    """H_gi de cada componente puro [J/kmol]; T (n,) -> (n, nc)."""
    xp = bk.xp
    T = xp.asarray(T, float)[:, None]
    A = xp.asarray(cs.cpA)[None, :]
    B = xp.asarray(cs.cpB)[None, :]
    C = xp.asarray(cs.cpC)[None, :]
    D = xp.asarray(cs.cpD)[None, :]
    T0 = T_REF
    if cs.cp_temperature == "celsius":  # hipótese da issue 004: polinómio avaliado em °C
        T = T - 273.15
        T0 = T0 - 273.15
        D = 0.0 * D  # o termo D/T^2 não é avaliável em °C (T = 0 °C); nulo nos hidrocarbonetos
    integ = A * (T - T0) + B / 2.0 * (T**2 - T0**2) + C / 3.0 * (T**3 - T0**3) - D * (1.0 / T - 1.0 / T0)
    return xp.asarray(cs.Hf)[None, :] + R_GAS * integ


def pure_cp_ideal(T, cs: ComponentSet, bk=NUMPY):
    """Cp de gás ideal de cada componente [J/(kmol K)]."""
    xp = bk.xp
    T = xp.asarray(T, float)[:, None]
    return R_GAS * (
        xp.asarray(cs.cpA)[None, :]
        + xp.asarray(cs.cpB)[None, :] * T
        + xp.asarray(cs.cpC)[None, :] * T**2
        + xp.asarray(cs.cpD)[None, :] / T**2
    )


def h_ideal(T, x, cs: ComponentSet, bk=NUMPY):
    """H_gi da mistura [J/kmol]; x (n, nc)."""
    xp = bk.xp
    return xp.einsum("ni,ni->n", xp.asarray(x, float), pure_h_ideal(T, cs, bk))
