"""Padrão anular (§2.4.1.2, eqs. 57–76): filme de líquido na parede e núcleo de gás com gotas
(entrainment de Wallis, 64–66), equilíbrio filme–núcleo resolvido na espessura do filme (59),
perda de carga (60), atrito interfacial unificado f_i = f_CS I_theta (72–76: Henstock & Hanratty
no horizontal, Wallis no vertical). Em (75) o quociente «v_L/v_G» é lido como o das viscosidades
cinemáticas nu_L/nu_G, como em Henstock & Hanratty (1976)."""

from __future__ import annotations

from ..backend import NUMPY
from .common import G, ClosureResult, FlowInputs
from .friction import fanning, vabs
from .scalar import solve_bracketed


def entrainment(inp: FlowInputs, bk=NUMPY):
    """Fração de líquido arrastada como gotas, Wallis (64–65), em [0, 1)."""
    xp = bk.xp
    phi = 1e4 * inp.v_SG * inp.mu_G / inp.sigma * bk.safe_sqrt(inp.rho_G / inp.rho_L)
    return xp.clip(1.0 - xp.exp(-0.125 * (phi - 1.5)), 0.0, 1.0 - 1e-9)


def annular_terms(delta, inp: FlowInputs, bk=NUMPY, entrainment_model="wallis"):
    """Termos para uma espessura relativa do filme delta = h_L/D em (0, 1/2)."""
    xp = bk.xp
    D = inp.D
    A = inp.A
    core = (1.0 - 2.0 * delta) ** 2
    AC = xp.maximum(A * core, 1e-12 * A)
    AF = xp.maximum(A - AC, 1e-12 * A)
    SF = xp.pi * D
    SI = xp.pi * D * bk.safe_sqrt(core)
    FE = entrainment(inp, bk) if entrainment_model == "wallis" else xp.zeros_like(inp.v_SL)
    RC = inp.v_SG / xp.maximum(inp.v_SG + inp.v_SL * FE, 1e-300)
    rhoC = inp.rho_G * RC + inp.rho_L * (1.0 - RC)
    muC = inp.mu_G * RC + inp.mu_L * (1.0 - RC)
    vF = inp.v_SL * (1.0 - FE) * A / AF
    vC = (inp.v_SG + inp.v_SL * FE) * A / AC
    DF = 4.0 * AF / SF
    fm = 1.0 if inp.f_wall_mult is None else inp.f_wall_mult
    fF, _ = fanning(vF, inp.rho_L, inp.mu_L, DF, inp.eps, bk=bk)
    fF = fF * fm
    tauF = fF * inp.rho_L * vF * vabs(vF, bk) / 2.0
    vCS = inp.v_SG + inp.v_SL * FE
    fCS, _ = fanning(vCS, rhoC, muC, D, inp.eps, bk=bk)
    fCS = fCS * fm
    ReLS = inp.rho_L * vabs(inp.v_SL, bk) * D / inp.mu_L
    ReGS = inp.rho_G * vabs(inp.v_SG, bk) * D / inp.mu_G
    nuL = inp.mu_L / inp.rho_L
    nuG = inp.mu_G / inp.rho_G
    FA = (
        ((0.707 * ReLS**0.5) ** 2.5 + (0.0379 * ReLS**0.9) ** 2.5) ** 0.4
        / xp.maximum(ReGS, 1e-300) ** 0.9
        * (nuL / nuG)
        * bk.safe_sqrt(inp.rho_L / inp.rho_G)
    )
    IH = 1.0 + 850.0 * FA
    IV = 1.0 + 300.0 * delta
    Itheta = IH * xp.cos(inp.theta) ** 2 + IV * xp.sin(inp.theta) ** 2
    fi = fCS * Itheta
    dv = vC - vF
    tauI = fi * rhoC * dv * vabs(dv, bk) / 2.0
    HL = AF / A + (1.0 - RC) * AC / A
    return dict(A=A, AF=AF, AC=AC, SF=SF, SI=SI, FE=FE, RC=RC, rhoC=rhoC, vF=vF, vC=vC, tauF=tauF, tauI=tauI, HL=HL)


def annular_residual(delta, inp: FlowInputs, bk=NUMPY, **kw):
    """Equilíbrio (59): tau_wF S_F/A_F - tau_i S_i (1/A_F + 1/A_C) + (rho_L - rho_C) g sin theta."""
    t = annular_terms(delta, inp, bk=bk, **kw)
    return (
        t["tauF"] * t["SF"] / t["AF"]
        - t["tauI"] * t["SI"] * (1.0 / t["AF"] + 1.0 / t["AC"])
        + (inp.rho_L - t["rhoC"]) * G * bk.xp.sin(inp.theta)
    )


def annular(inp: FlowInputs, bk=NUMPY, n_bisect=30, n_newton=3, **kw) -> ClosureResult:
    xp = bk.xp
    n = inp.v_SL.shape[0]
    delta = solve_bracketed(
        lambda d: annular_residual(d, inp, bk=bk, **kw),
        xp.full((n,), 1e-7),
        xp.full((n,), 0.5 - 1e-6),
        bk,
        n_bisect,
        n_newton,
    )
    t = annular_terms(delta, inp, bk=bk, **kw)
    F_fric = t["tauF"] * t["SF"] / t["A"]
    F_grav = (t["AF"] / t["A"] * inp.rho_L + t["AC"] / t["A"] * t["rhoC"]) * G * xp.sin(inp.theta)
    return ClosureResult(
        H_L=t["HL"], F=F_fric + F_grav, F_friction=F_fric, F_gravity=F_grav, v_L=t["vF"], v_G=t["vC"], level=delta
    )
