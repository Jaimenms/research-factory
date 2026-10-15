"""Padrão intermitente (§2.4.1.4, eqs. 82–99), modelo unificado de Gomez et al. (2000): unidade
de golfada com pistão (holdup H_LS por (84)) e filme com bolha de Taylor (equilíbrio (82) resolvido
no holdup do filme H_LF), velocidades por conservação de massa (85–93), perda de carga ponderada
pelos comprimentos (83, 97–99). O comprimento L_S (87) não afeta H_L nem a perda de carga: só
entra em L_U por (98) e as razões L_S/L_U, L_F/L_U são independentes dele.

Leituras do texto registadas na issue 003: (84) com o expoente de Gomez (-7,85e-3 theta); (85)
com cos theta e sin theta fora da raiz (Bendiksen 1984); (86) com H_LS^0,1 como impresso; (89)
corrigida (v_LS isolado); f_i = 0,0142 no filme (p. 71, «similares ao estratificado»)."""

from __future__ import annotations

from ..backend import NUMPY
from .bubble import harmathy_v0
from .common import G, ClosureResult, FlowInputs
from .friction import fanning, vabs
from .geometry import stratified_level
from .scalar import solve_bracketed


def slug_holdup(inp: FlowInputs, bk=NUMPY, variant="gomez2000"):
    """H_LS de Gomez et al. (2000), eq. (84): exp(-(7,85e-3 theta + 2,48e-6 Re_SS)), theta em rad."""
    xp = bk.xp
    ReSS = inp.rho_L * vabs(inp.v_M, bk) * inp.D / inp.mu_L
    if variant == "gomez2000":
        return xp.clip(xp.exp(-(7.85e-3 * inp.theta + 2.48e-6 * ReSS)), 0.05, 1.0 - 1e-6)
    raise ValueError(variant)


def taylor_bubble_velocity(inp: FlowInputs, bk=NUMPY):
    """v_TB de Bendiksen (1984), eq. (85): 1,2 v_M + 0,542 sqrt(gD) cos theta + 0,351 sqrt(gD) sin theta."""
    xp = bk.xp
    gD = bk.safe_sqrt(G * inp.D)
    return 1.2 * inp.v_M + 0.542 * gD * xp.cos(inp.theta) + 0.351 * gD * xp.sin(inp.theta)


def film_terms(HLF, inp: FlowInputs, HLS, vTB, vLS, bk=NUMPY, fi_const=0.0142):
    xp = bk.xp
    HLF = xp.clip(HLF, 1e-6, 1.0 - 1e-6)
    vLF = vTB - (vTB - vLS) * HLS / HLF  # (91)
    vGF = (inp.v_M - vLF * HLF) / (1.0 - HLF)  # (93)
    hD, alpha = stratified_level(1.0 - HLF, bk=bk)  # alpha: ângulo molhado do filme
    A = inp.A
    AF = HLF * A
    AG = (1.0 - HLF) * A
    SF = alpha * inp.D / 2.0
    SG = (2.0 * xp.pi - alpha) * inp.D / 2.0
    SI = inp.D * xp.sin(alpha / 2.0)
    DF = 4.0 * AF / xp.maximum(SF, 1e-12 * inp.D)
    DG = 4.0 * AG / xp.maximum(SG + SI, 1e-12 * inp.D)
    fF, _ = fanning(vLF, inp.rho_L, inp.mu_L, DF, inp.eps, bk=bk)
    fG, _ = fanning(vGF, inp.rho_G, inp.mu_G, DG, inp.eps, bk=bk)
    tauF = fF * inp.rho_L * vLF * vabs(vLF, bk) / 2.0
    tauG = fG * inp.rho_G * vGF * vabs(vGF, bk) / 2.0
    dv = vGF - vLF
    tauI = fi_const * inp.rho_G * dv * vabs(dv, bk) / 2.0
    return dict(vLF=vLF, vGF=vGF, A=A, AF=AF, AG=AG, SF=SF, SG=SG, SI=SI, tauF=tauF, tauG=tauG, tauI=tauI, hD=hD)


def film_residual(HLF, inp, HLS, vTB, vLS, bk=NUMPY, **kw):
    """(82): tau_wF S_F/A_F - tau_wG S_G/A_G - tau_i S_i (1/A_F + 1/A_G) + (rho_L - rho_G) g sin theta."""
    t = film_terms(HLF, inp, HLS, vTB, vLS, bk=bk, **kw)
    return (
        t["tauF"] * t["SF"] / t["AF"]
        - t["tauG"] * t["SG"] / t["AG"]
        - t["tauI"] * t["SI"] * (1.0 / t["AF"] + 1.0 / t["AG"])
        + (inp.rho_L - inp.rho_G) * G * bk.xp.sin(inp.theta)
    )


def intermittent(
    inp: FlowInputs, bk=NUMPY, n_bisect=30, n_newton=3, slug_holdup_variant="gomez2000", fi_const=0.0142, **kw
) -> ClosureResult:
    xp = bk.xp
    n = inp.v_SL.shape[0]
    HLS = slug_holdup(inp, bk, slug_holdup_variant)
    vTB = taylor_bubble_velocity(inp, bk)
    vBS = 1.2 * inp.v_M + harmathy_v0(inp, bk) * bk.safe_pow(HLS, 0.1) * xp.sin(inp.theta)  # (86)
    vLS = (inp.v_M - vBS * (1.0 - HLS)) / HLS  # (89) corrigida
    HLF = solve_bracketed(
        lambda h: film_residual(h, inp, HLS, vTB, vLS, bk=bk, fi_const=fi_const),
        xp.full((n,), 1e-4),
        HLS * (1.0 - 1e-6),
        bk,
        n_bisect,
        n_newton,
    )
    t = film_terms(HLF, inp, HLS, vTB, vLS, bk=bk, fi_const=fi_const)
    vLF = t["vLF"]
    # razões de comprimento (98) invertida: L_S/L_U = (v_SL - v_LF H_LF)/(v_LS H_LS - v_LF H_LF)
    den = vLS * HLS - vLF * HLF
    LS_LU = xp.clip((inp.v_SL - vLF * HLF) / xp.where(xp.abs(den) < 1e-300, 1e-300, den), 0.0, 1.0)
    LF_LU = 1.0 - LS_LU
    HL = HLS * LS_LU + HLF * LF_LU  # (99)
    rhoS = inp.rho_L * HLS + inp.rho_G * (1.0 - HLS)
    muS = inp.mu_L * HLS + inp.mu_G * (1.0 - HLS)
    fS, _ = fanning(inp.v_M, rhoS, muS, inp.D, inp.eps, bk=bk)
    tauS = fS * rhoS * inp.v_M * vabs(inp.v_M, bk) / 2.0
    F_fric = tauS * 4.0 / inp.D * LS_LU + (t["tauF"] * t["SF"] + t["tauG"] * t["SG"]) / t["A"] * LF_LU
    rhoU = inp.rho_L * HL + inp.rho_G * (1.0 - HL)
    F_grav = rhoU * G * xp.sin(inp.theta)
    vL = inp.v_SL / HL
    vG = inp.v_SG / (1.0 - HL)
    return ClosureResult(H_L=HL, F=F_fric + F_grav, F_friction=F_fric, F_gravity=F_grav, v_L=vL, v_G=vG, level=HLF)
