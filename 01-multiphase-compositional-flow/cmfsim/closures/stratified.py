"""Padrão estratificado (§2.4.1.1, eqs. 39–56): equilíbrio de momento entre as fases resolvido em
h_L/D (41) e perda de carga (42). Atrito com a parede por Churchill (49–52) em diâmetros
hidráulicos (53–54); atrito interfacial com f_i = 0,0142 constante (Xiao et al. 1990, eq. 56)
por omissão, ou f_i = f_G (Taitel & Dukler) ou Andritsos & Hanratty como opções."""

from __future__ import annotations

from ..backend import NUMPY
from .common import G, ClosureResult, FlowInputs
from .friction import fanning, vabs
from .scalar import solve_bracketed


def geometry_from_level(hD, D, bk=NUMPY):
    """Do nível h_L/D: ângulo molhado alpha (43–44), áreas e perímetros (45–47); dA_L/dh_L = S_i (48)."""
    xp = bk.xp
    alpha = 2.0 * xp.arccos(xp.clip(1.0 - 2.0 * hD, -1.0, 1.0))
    A = xp.pi * D**2 / 4.0
    AL = A * (alpha - xp.sin(alpha)) / (2.0 * xp.pi)
    AG = A - AL
    SL = alpha * D / 2.0
    SG = (2.0 * xp.pi - alpha) * D / 2.0  # a dissertação imprime (2 pi - alpha) D (eq. 46): gralha
    SI = D * xp.sin(alpha / 2.0)
    return alpha, A, AL, AG, SL, SG, SI


def stratified_terms(hD, inp: FlowInputs, interfacial="xiao_const", fi_const=0.0142, DG_rule="taitel_dukler", bk=NUMPY):
    """Tensões e geometria do estratificado para um nível hD; devolve dict."""
    xp = bk.xp
    alpha, A, AL, AG, SL, SG, SI = geometry_from_level(hD, inp.D, bk)
    AL = xp.maximum(AL, 1e-12 * A)
    AG = xp.maximum(AG, 1e-12 * A)
    HL = AL / A
    vL = inp.v_SL / HL
    vG = inp.v_SG / (1.0 - HL)
    DL = 4.0 * AL / xp.maximum(SL, 1e-12 * inp.D)
    den_G = (SG + SI) if DG_rule == "taitel_dukler" else (SL + SG)
    DG = 4.0 * AG / xp.maximum(den_G, 1e-12 * inp.D)
    fm = 1.0 if inp.f_wall_mult is None else inp.f_wall_mult
    fL, _ = fanning(vL, inp.rho_L, inp.mu_L, DL, inp.eps, bk=bk)
    fG, _ = fanning(vG, inp.rho_G, inp.mu_G, DG, inp.eps, bk=bk)
    fL = fL * fm
    fG = fG * fm
    tauL = fL * inp.rho_L * vL * vabs(vL, bk) / 2.0
    tauG = fG * inp.rho_G * vG * vabs(vG, bk) / 2.0
    dv = vG - vL
    if interfacial == "xiao_const":
        fi = fi_const * xp.ones_like(fG)
    elif interfacial == "taitel_dukler":
        fi = fG
    elif interfacial == "andritsos_hanratty":
        fi = fG * (1.0 + 15.0 * xp.maximum(0.0, xp.abs(inp.v_SG) / 5.5 - 1.0) * bk.safe_sqrt(hD))
    else:
        raise ValueError(interfacial)
    tauI = fi * inp.rho_G * dv * vabs(dv, bk) / 2.0
    return dict(
        alpha=alpha, A=A, AL=AL, AG=AG, SL=SL, SG=SG, SI=SI, HL=HL, vL=vL, vG=vG, tauL=tauL, tauG=tauG, tauI=tauI
    )


def stratified_residual(hD, inp: FlowInputs, bk=NUMPY, **kw):
    """Equilíbrio (41): tau_wL S_L/A_L - tau_wG S_G/A_G - tau_i S_i (1/A_L + 1/A_G) + (rho_L - rho_G) g sin theta."""
    t = stratified_terms(hD, inp, bk=bk, **kw)
    return (
        t["tauL"] * t["SL"] / t["AL"]
        - t["tauG"] * t["SG"] / t["AG"]
        - t["tauI"] * t["SI"] * (1.0 / t["AL"] + 1.0 / t["AG"])
        + (inp.rho_L - inp.rho_G) * G * bk.xp.sin(inp.theta)
    )


def stratified(inp: FlowInputs, bk=NUMPY, n_bisect=30, n_newton=3, **kw) -> ClosureResult:
    xp = bk.xp
    n = inp.v_SL.shape[0]
    hD = solve_bracketed(
        lambda h: stratified_residual(h, inp, bk=bk, **kw),
        xp.full((n,), 1e-6),
        xp.full((n,), 1.0 - 1e-6),
        bk,
        n_bisect,
        n_newton,
    )
    t = stratified_terms(hD, inp, bk=bk, **kw)
    F_fric = (t["tauL"] * t["SL"] + t["tauG"] * t["SG"]) / t["A"]
    F_grav = (t["HL"] * inp.rho_L + (1.0 - t["HL"]) * inp.rho_G) * G * xp.sin(inp.theta)
    return ClosureResult(
        H_L=t["HL"], F=F_fric + F_grav, F_friction=F_fric, F_gravity=F_grav, v_L=t["vL"], v_G=t["vG"], level=hD
    )
