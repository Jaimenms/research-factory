"""Equação de estado de Peng–Robinson (1976) para misturas, com regras de mistura clássicas
(eqs. 101–112 da dissertação): fator de compressibilidade por raízes analíticas da cúbica (102)
polidas por Newton, coeficientes de fugacidade (121), entalpia residual (146–147) e massa
específica (126, na forma correta rho = P MW / (Z R T)). Uma implementação para numpy e JAX
(`bk`), vetorizada em pontos: T, P de forma (n,), composições de forma (n, nc).
"""

from __future__ import annotations

import numpy as np

from ..backend import NUMPY
from .components import R_GAS, ComponentSet

SQRT2 = float(np.sqrt(2.0))


def kappa(cs: ComponentSet, xp=np):
    """kappa(omega) de Peng–Robinson (109); a dissertação imprime -0,2699, o valor original é -0,26992."""
    w = xp.asarray(cs.omega)
    return 0.37464 + 1.54226 * w - 0.26992 * w**2


def pure_params(T, cs: ComponentSet, bk=NUMPY):
    """a_i(T), b_i e da_i/dT (106–108). T de forma (n,); saídas (n, nc)."""
    xp = bk.xp
    T = xp.asarray(T, float)[:, None]
    Tc = xp.asarray(cs.Tc)[None, :]
    Pc = xp.asarray(cs.Pc)[None, :]
    k = kappa(cs, xp)[None, :]
    sqrt_alpha = 1.0 + k * (1.0 - xp.sqrt(T / Tc))
    ac = cs.omega_a * (R_GAS * Tc) ** 2 / Pc
    a = ac * sqrt_alpha**2
    b = cs.omega_b * R_GAS * Tc / Pc * xp.ones_like(T)
    dadT = -ac * k * sqrt_alpha / xp.sqrt(T * Tc)
    return a, b, dadT


def mixture_params(T, x, cs: ComponentSet, bk=NUMPY):
    """a, b, da/dT da mistura e o vetor sum_j x_j a_ij (110–112, 147). x de forma (n, nc)."""
    xp = bk.xp
    a_i, b_i, dadT_i = pure_params(T, cs, bk)
    x = xp.asarray(x, float)
    one_m_k = 1.0 - xp.asarray(cs.kij)[None, :, :]
    sqrt_a = xp.sqrt(a_i)
    a_ij = one_m_k * sqrt_a[:, :, None] * sqrt_a[:, None, :]
    sum_xa = xp.einsum("nj,nij->ni", x, a_ij)
    a_mix = xp.einsum("ni,ni->n", x, sum_xa)
    b_mix = xp.einsum("ni,ni->n", x, b_i)
    # d a_ij/dT = (1 - k_ij) (a_j da_i/dT + a_i da_j/dT) / (2 sqrt(a_i a_j))
    da_ij = (
        one_m_k
        * (
            dadT_i[:, :, None] * sqrt_a[:, None, :] / sqrt_a[:, :, None]
            + dadT_i[:, None, :] * sqrt_a[:, :, None] / sqrt_a[:, None, :]
        )
        / 2.0
    )
    dadT_mix = xp.einsum("ni,nij,nj->n", x, da_ij, x)
    return a_mix, b_mix, dadT_mix, sum_xa, b_i


def cubic_roots_sorted(c2, c1, c0, bk=NUMPY):
    """Raízes reais de Z^3 + c2 Z^2 + c1 Z + c0 = 0, ordenadas (z1 <= z2 <= z3), e o número de raízes
    reais (1 ou 3). Com uma só raiz real as três saídas são iguais a ela. Solução analítica
    (trigonométrica / Cardano) sem ramos em valores, seguida de três iterações de Newton."""
    xp = bk.xp
    p = c1 - c2**2 / 3.0
    q = 2.0 * c2**3 / 27.0 - c2 * c1 / 3.0 + c0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    three = disc < 0.0
    # três raízes reais (p < 0): argumentos seguros fora do ramo (where antes das funções, para
    # que as tangentes da diferenciação automática fiquem finitas)
    p_neg = xp.where(three, p, -1.0)
    r = 2.0 * xp.sqrt(-p_neg / 3.0)
    arg = xp.clip(3.0 * q / xp.where(three, p_neg * r, 1.0), -1.0, 1.0)
    phi = xp.arccos(arg) / 3.0
    t1 = r * xp.cos(phi)
    t2 = r * xp.cos(phi - 2.0 * xp.pi / 3.0)
    t3 = r * xp.cos(phi - 4.0 * xp.pi / 3.0)
    lo3 = xp.minimum(xp.minimum(t1, t2), t3)
    hi3 = xp.maximum(xp.maximum(t1, t2), t3)
    mid3 = t1 + t2 + t3 - lo3 - hi3
    # uma raiz real
    sd = xp.sqrt(xp.where(three, 1.0, xp.maximum(disc, 0.0)))
    t_one = xp.cbrt(-q / 2.0 + sd) + xp.cbrt(-q / 2.0 - sd)
    shift = -c2 / 3.0
    z1 = xp.where(three, lo3, t_one) + shift
    z2 = xp.where(three, mid3, t_one) + shift
    z3 = xp.where(three, hi3, t_one) + shift

    def polish(z):
        for _ in range(3):
            f = ((z + c2) * z + c1) * z + c0
            df = (3.0 * z + 2.0 * c2) * z + c1
            df = xp.where(xp.abs(df) < 1e-300, 1e-300, df)
            z = z - f / df
        return z

    return polish(z1), polish(z2), polish(z3), xp.where(three, 3, 1)


def compressibility(A, B, bk=NUMPY):
    """Z do líquido (menor raiz > B) e do vapor (maior raiz) da cúbica (102)."""
    xp = bk.xp
    c2 = -(1.0 - B)
    c1 = A - 3.0 * B**2 - 2.0 * B
    c0 = -(A * B - B**2 - B**3)
    z1, z2, z3, nroots = cubic_roots_sorted(c2, c1, c0, bk)
    z_liq = xp.where(z1 > B, z1, xp.where(z2 > B, z2, z3))
    z_vap = z3
    return z_liq, z_vap


def phase_Z(T, P, x, cs: ComponentSet, phase: str, bk=NUMPY):
    """Z e os parâmetros adimensionais A, B para a fase `phase` ("liquid" | "vapor")."""
    xp = bk.xp
    T = xp.asarray(T, float)
    P = xp.asarray(P, float)
    a, b, dadT, sum_xa, b_i = mixture_params(T, x, cs, bk)
    A = a * P / (R_GAS * T) ** 2
    B = b * P / (R_GAS * T)
    z_liq, z_vap = compressibility(A, B, bk)
    Z = z_liq if phase == "liquid" else z_vap
    return Z, A, B, a, b, dadT, sum_xa, b_i


def ln_phi(T, P, x, cs: ComponentSet, phase: str, bk=NUMPY):
    """ln do coeficiente de fugacidade de cada componente na mistura (121); saída (n, nc)."""
    xp = bk.xp
    Z, A, B, a, b, dadT, sum_xa, b_i = phase_Z(T, P, x, cs, phase, bk)
    bk_b = b_i / b[:, None]
    zmb = xp.maximum(Z - B, 1e-300)
    log_term = xp.log((Z + (1.0 + SQRT2) * B) / (Z + (1.0 - SQRT2) * B))
    return (
        bk_b * (Z - 1.0)[:, None]
        - xp.log(zmb)[:, None]
        - (A / (2.0 * SQRT2 * B))[:, None] * (2.0 * sum_xa / a[:, None] - bk_b) * log_term[:, None]
    )


def residual_enthalpy(T, P, x, cs: ComponentSet, phase: str, bk=NUMPY):
    """Entalpia residual molar H - H_gi [J/kmol] (146–147)."""
    xp = bk.xp
    Z, A, B, a, b, dadT, sum_xa, b_i = phase_Z(T, P, x, cs, phase, bk)
    T = xp.asarray(T, float)
    log_term = xp.log((Z + (1.0 + SQRT2) * B) / (Z + (1.0 - SQRT2) * B))
    return R_GAS * T * (Z - 1.0) + (T * dadT - a) / (2.0 * SQRT2 * b) * log_term


def molar_mass(x, cs: ComponentSet, bk=NUMPY):
    """MW da mistura [kg/kmol]; x (n, nc)."""
    xp = bk.xp
    return xp.asarray(x, float) @ xp.asarray(cs.MW)


def density(T, P, x, cs: ComponentSet, phase: str, bk=NUMPY):
    """Massa específica [kg/m3] da fase: rho = P MW / (Z R T) (126 corrigida)."""
    xp = bk.xp
    Z = phase_Z(T, P, x, cs, phase, bk)[0]
    return xp.asarray(P, float) * molar_mass(x, cs, bk) / (Z * R_GAS * xp.asarray(T, float))
