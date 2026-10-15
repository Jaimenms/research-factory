"""Padrão de bolhas (§2.4.1.3, eqs. 77–81): drift-flux v_G = C_0 v_M + v_0 sin theta H_L^0.5 com
C_0 = 1,15 (Chokshi et al.) e v_0 de Harmathy; holdup implícito em v_GS = v_G (1 - H_L) (79);
perda de carga homogénea (80–81) com a mistura líquido–bolhas como um só fluido (a viscosidade da
mistura, não dita na dissertação, é ponderada pelo holdup como a massa específica)."""

from __future__ import annotations

from ..backend import NUMPY
from .common import G, ClosureResult, FlowInputs
from .friction import fanning, vabs
from .scalar import solve_bracketed


def harmathy_v0(inp: FlowInputs, bk=NUMPY):
    return 1.53 * bk.safe_pow(G * inp.sigma * (inp.rho_L - inp.rho_G) / inp.rho_L**2, 0.25)


def bubble_residual(HL, inp: FlowInputs, bk=NUMPY, C0=1.15):
    """(79): v_GS - (C_0 v_M + v_0 sin theta H_L^0.5)(1 - H_L)."""
    xp = bk.xp
    vG = C0 * inp.v_M + harmathy_v0(inp, bk) * xp.sin(inp.theta) * bk.safe_sqrt(HL)
    return inp.v_SG - vG * (1.0 - HL)


def bubble(inp: FlowInputs, bk=NUMPY, n_bisect=30, n_newton=3, C0=1.15, **kw) -> ClosureResult:
    xp = bk.xp
    n = inp.v_SL.shape[0]
    # em H_L -> 0 o resíduo é v_GS - C_0 v_M (negativo no padrão de bolhas) e em H_L -> 1 é v_GS > 0
    HL = solve_bracketed(
        lambda h: bubble_residual(h, inp, bk=bk, C0=C0),
        xp.full((n,), 1e-6),
        xp.full((n,), 1.0 - 1e-6),
        bk,
        n_bisect,
        n_newton,
    )
    HL = xp.clip(HL, 1e-6, 1.0 - 1e-6)
    vG = inp.v_SG / (1.0 - HL)
    vL = inp.v_SL / HL
    rhoM = inp.rho_G * (1.0 - HL) + inp.rho_L * HL
    muM = inp.mu_G * (1.0 - HL) + inp.mu_L * HL
    fM, _ = fanning(inp.v_M, rhoM, muM, inp.D, inp.eps, bk=bk)
    tauM = fM * rhoM * inp.v_M * vabs(inp.v_M, bk) / 2.0
    F_fric = tauM * 4.0 / inp.D
    F_grav = rhoM * G * xp.sin(inp.theta)
    return ClosureResult(H_L=HL, F=F_fric + F_grav, F_friction=F_fric, F_gravity=F_grav, v_L=vL, v_G=vG, level=HL)
