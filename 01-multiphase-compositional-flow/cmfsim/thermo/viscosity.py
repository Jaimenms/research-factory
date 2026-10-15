"""Viscosidades: vapor por estados correspondentes de Pedersen et al. (1984, 1987) com o metano de
referência de Hanley, McCarty & Haynes (1975) (eqs. 127–136 e Tabela 2 da dissertação); líquido
por Allan & Teja (1991) com número de carbonos efetivo (137–141). Uma implementação para numpy
e JAX (`bk`).

O que a dissertação não diz e aqui vem da literatura (Pedersen & Fredenslund 1987): as regras
de mistura de Tc, Pc e MW, o parâmetro alpha = 1 + 7.378e-3 rho_r^1.847 MW^0.5173 (alpha_0 do
metano com 0.031), a massa específica crítica do metano (162.84 kg/m3) e Delta T = T - 91 K na
tangente hiperbólica de (136). A coluna s da Tabela 2 imprime 1,696986e5 para s1; o valor de
Hanley et al. é 1.696985927 (sem a potência), usado aqui.
"""

from __future__ import annotations

import numpy as np

from ..backend import NUMPY
from .components import ComponentSet
from .pr import ln_phi, phase_Z, molar_mass

# metano de referência (Tabela 7)
CH4 = ComponentSet(
    names=("C1",),
    MW=[16.0],
    Pc=[46.0e5],
    Tc=[190.6],
    omega=[0.012],
    Tb=[111.4],
    Hf=[0.0],
    cpA=[0.0],
    cpB=[0.0],
    cpC=[0.0],
    cpD=[0.0],
    Nc_eff=[1.0],
)
RHO_C0 = 162.84  # kg/m3, massa específica crítica do metano
# Tabela 2 (Hanley, McCarty & Haynes 1975 ajustados por Pedersen 1984); unidades: micropoise, g/cm3, K
GV = np.array(
    [-2.090975e5, 2.647269e5, -1.472818e5, 4.716740e4, -9.491872e3, 1.219979e3, -9.627993e1, 4.274152, -8.141531e-2]
)
S = (1.696985927, -0.133372346, 1.4, 168.0)
J = (-10.3506, 17.5716, -3019.39, 188.730, 0.0429036, 145.290, 6127.68)
KK = (-9.74602, 18.0834, -4126.66, 44.6055, 0.976544, 81.8134, 15649.9)


def methane_density_stable(T, P, bk=NUMPY):
    """Massa específica do metano puro [kg/m3] na fase estável (menor fugacidade) por PR."""
    xp = bk.xp
    T = xp.asarray(T, float)
    P = xp.asarray(P, float)
    x = xp.ones((T.shape[0], 1))
    Zl = phase_Z(T, P, x, CH4, "liquid", bk)[0]
    Zv = phase_Z(T, P, x, CH4, "vapor", bk)[0]
    phil = ln_phi(T, P, x, CH4, "liquid", bk)[:, 0]
    phiv = ln_phi(T, P, x, CH4, "vapor", bk)[:, 0]
    Z = xp.where(phil < phiv, Zl, Zv)
    from .components import R_GAS

    return P * 16.0 / (Z * R_GAS * T)


def methane_viscosity(T, rho_g_cm3, bk=NUMPY):
    """Viscosidade do metano [Pa s] de Hanley et al. (129–136) com rho em g/cm3."""
    xp = bk.xp
    T = xp.asarray(T, float)
    rho = xp.asarray(rho_g_cm3, float)
    eta0 = sum(float(GV[k]) * T ** ((k + 1 - 4) / 3.0) for k in range(9))
    eta1 = S[0] + S[1] * (S[2] - xp.log(T / S[3])) ** 2
    rho_c = RHO_C0 / 1000.0
    theta = (rho - rho_c) / rho_c
    rho_p = bk.safe_pow(rho, 0.1)
    rho_h = bk.safe_sqrt(rho)
    d1 = xp.exp(J[0] + J[3] / T) * (
        xp.exp(rho_p * (J[1] + J[2] / T**1.5) + theta * rho_h * (J[4] + J[5] / T + J[6] / T**2)) - 1.0
    )
    d2 = xp.exp(KK[0] + KK[3] / T) * (
        xp.exp(rho_p * (KK[1] + KK[2] / T**1.5) + theta * rho_h * (KK[4] + KK[5] / T + KK[6] / T**2)) - 1.0
    )
    htan = xp.tanh(T - 91.0)
    F1 = (htan + 1.0) / 2.0
    F2 = (1.0 - htan) / 2.0
    mu_micropoise = eta0 + eta1 * rho + F1 * d1 + F2 * d2
    return mu_micropoise * 1e-7


def pedersen_mixing(x, cs: ComponentSet, bk=NUMPY):
    """Tc, Pc [Pa] e MW [kg/kmol] da mistura pelas regras de Pedersen (128); x (n, nc)."""
    xp = bk.xp
    x = xp.asarray(x, float)
    Tc = xp.asarray(cs.Tc)
    Pc = xp.asarray(cs.Pc)
    MW = xp.asarray(cs.MW)
    v = (Tc / Pc) ** (1.0 / 3.0)
    cub = (v[:, None] + v[None, :]) ** 3
    sT = xp.sqrt(Tc[:, None] * Tc[None, :])
    num = xp.einsum("ni,nj,ij->n", x, x, cub * sT)
    den = xp.einsum("ni,nj,ij->n", x, x, cub)
    Tc_mix = num / den
    Pc_mix = 8.0 * num / den**2
    Mn = x @ MW
    Mw = (x @ MW**2) / Mn
    MW_mix = 1.304e-4 * (Mw**2.303 - Mn**2.303) + Mn
    return Tc_mix, Pc_mix, MW_mix


def pedersen_viscosity(T, P, x, cs: ComponentSet, bk=NUMPY):
    """Viscosidade da mistura [Pa s] por estados correspondentes com o metano de referência (127–128)."""
    xp = bk.xp
    T = xp.asarray(T, float)
    P = xp.asarray(P, float)
    Tc_mix, Pc_mix, MW_mix = pedersen_mixing(x, cs, bk)
    Tc0, Pc0, MW0 = 190.6, 46.0e5, 16.0
    T0p = T * Tc0 / Tc_mix
    P0p = P * Pc0 / Pc_mix
    rho_r = methane_density_stable(T0p, P0p, bk) / RHO_C0
    alpha0 = 1.0 + 0.031 * bk.safe_pow(rho_r, 1.847)
    alpha = 1.0 + 7.378e-3 * bk.safe_pow(rho_r, 1.847) * MW_mix**0.5173
    T0 = T0p * alpha0 / alpha
    P0 = P0p * alpha0 / alpha
    rho0 = methane_density_stable(T0, P0, bk) / 1000.0  # g/cm3
    mu0 = methane_viscosity(T0, rho0, bk)
    return (
        (Tc_mix / Tc0) ** (-1.0 / 6.0) * (Pc_mix / Pc0) ** (2.0 / 3.0) * (MW_mix / MW0) ** 0.5 * (alpha / alpha0) * mu0
    )


def allan_teja_viscosity(T, x, cs: ComponentSet, bk=NUMPY):
    """Viscosidade do líquido [Pa s] por Allan & Teja (137–141); a correlação dá cP."""
    xp = bk.xp
    T = xp.asarray(T, float)
    Nc = xp.asarray(x, float) @ xp.asarray(cs.Nc_eff)
    A = 145.73 + 99.01 * Nc + 0.83 * Nc**2 - 0.125 * Nc**3
    B = 30.48 + 34.01 * Nc - 1.23 * Nc**2 + 0.017 * Nc**3
    C = -3.07 - 1.99 * Nc
    return xp.exp(A * (1.0 / (T + C) - 1.0 / B)) * 1e-3
