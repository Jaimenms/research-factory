"""Fechamentos mecânicos e identificação de padrão (§2.3–2.4.1): um só ponto de entrada,
`flow_closure(inp, opts, bk)`, que devolve holdup, termo dissipativo, velocidades e os pesos dos
padrões. Opções estáticas em `ClosureOptions` (hashable: estáticas num jit)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..backend import NUMPY
from .annular import annular
from .bubble import bubble
from .common import ClosureResult, FlowInputs, smooth_step
from .intermittent import intermittent
from .regime import horizontal_weights, vertical_fraction, vertical_weights
from .stratified import stratified

PATTERNS = ("stratified", "annular", "bubble", "intermittent")
# códigos da Figura 37 da dissertação: 0 estratificado, 2 anular, 3 intermitente, 4 bolhas dispersas
PATTERN_CODES = {"stratified": 0, "annular": 2, "intermittent": 3, "bubble": 4}


@dataclass(frozen=True)
class ClosureOptions:
    regime: str = "automatic"  # automatic | stratified | annular | bubble | intermittent
    k_switch: Optional[float] = 50.0  # sigmoides relativas; None = degraus duros
    vertical_angle_deg: float = 45.0
    blend_deg: float = 0.0  # > 0: rampa C1 entre os conjuntos horizontal e vertical
    stratified_interfacial: str = "xiao_const"  # xiao_const | taitel_dukler | andritsos_hanratty
    stratified_fi_const: float = 0.0142
    stratified_DG: str = "taitel_dukler"  # taitel_dukler (S_G + S_i) | as_printed (S_L + S_G)
    annular_entrainment: str = "wallis"  # wallis | none
    bubble_C0: float = 1.15
    intermittent_fi_const: float = 0.0142
    dispersed_density: str = "taitel"  # eq. (34): taitel ((rho_L - rho_G)/rho_L) | as_printed (rho_G/rho_L)
    bubble_slug: str = "taitel"  # eq. (36): taitel | as_printed
    bubble_existence: str = "as_printed"  # eq. (37): as_printed (<= 4,36) | taitel (>= 4,36)
    dispersed_max_void: Optional[float] = (
        0.52  # eq. (38): limite de Taitel et al. (1980); None = como impresso (a Figura 23 só se reproduz com 0,52: 95 % contra 83 %)
    )
    kh_velocity: str = (
        "actual"  # eq. (32): actual (v_G no equilíbrio estratificado, Taitel & Dukler) | superficial_as_printed (v_SG, «se o gás escoasse sozinho», p. 53)
    )
    wall_friction: str = (
        "fanning"  # eqs. (49–50): fanning (tau = f_F rho v|v|/2) | darcy_as_printed (f de (50) como impresso: 4x)
    )
    n_bisect: int = 30
    n_newton: int = 3


def _solver_kw(opts):
    return dict(n_bisect=opts.n_bisect, n_newton=opts.n_newton)


def _strat_kw(opts):
    return dict(interfacial=opts.stratified_interfacial, fi_const=opts.stratified_fi_const, DG_rule=opts.stratified_DG)


def pattern_results(inp: FlowInputs, opts: ClosureOptions = ClosureOptions(), bk=NUMPY):
    """Os quatro fechamentos avaliados no mesmo ponto (dict por nome)."""
    return dict(
        stratified=stratified(inp, bk=bk, **_solver_kw(opts), **_strat_kw(opts)),
        annular=annular(inp, bk=bk, **_solver_kw(opts), entrainment_model=opts.annular_entrainment),
        bubble=bubble(inp, bk=bk, **_solver_kw(opts), C0=opts.bubble_C0),
        intermittent=intermittent(inp, bk=bk, **_solver_kw(opts), fi_const=opts.intermittent_fi_const),
    )


def pattern_weights(inp: FlowInputs, hD_eq, opts: ClosureOptions = ClosureOptions(), bk=NUMPY):
    """Pesos (n, 4) na ordem PATTERNS, misturando os conjuntos horizontal e vertical por ângulo."""
    xp = bk.xp
    wh = horizontal_weights(inp, hD_eq, opts, bk=bk, **_strat_kw(opts))
    wv = vertical_weights(inp, opts, bk=bk)
    fv = vertical_fraction(inp.theta, opts, bk=bk)
    return xp.stack([(1.0 - fv) * h + fv * v for h, v in zip(wh, wv)], axis=1)


def flow_closure(inp: FlowInputs, opts: ClosureOptions = ClosureOptions(), bk=NUMPY):
    """Holdup, termo dissipativo e velocidades no padrão fixo ou na mistura ponderada dos padrões
    identificados. Devolve (ClosureResult, pesos (n, 4))."""
    xp = bk.xp
    n = inp.v_SL.shape[0]
    if opts.regime != "automatic":
        res = pattern_results_single(inp, opts, bk)
        w = xp.zeros((n, 4))
        w = bk.set(w, (slice(None), PATTERNS.index(opts.regime)), 1.0)
        return res, w
    results = pattern_results(inp, opts, bk)
    w = pattern_weights(inp, results["stratified"].level, opts, bk)
    blended = ClosureResult.blend([results[p] for p in PATTERNS], [w[:, i] for i in range(4)], bk)
    return blended, w


def pattern_results_single(inp: FlowInputs, opts: ClosureOptions, bk=NUMPY):
    if opts.regime == "stratified":
        return stratified(inp, bk=bk, **_solver_kw(opts), **_strat_kw(opts))
    if opts.regime == "annular":
        return annular(inp, bk=bk, **_solver_kw(opts), entrainment_model=opts.annular_entrainment)
    if opts.regime == "bubble":
        return bubble(inp, bk=bk, **_solver_kw(opts), C0=opts.bubble_C0)
    if opts.regime == "intermittent":
        return intermittent(inp, bk=bk, **_solver_kw(opts), fi_const=opts.intermittent_fi_const)
    raise ValueError(opts.regime)


def pattern_code(weights, bk=NUMPY):
    """Código do padrão dominante (Figura 37: 0 estratificado, 2 anular, 3 intermitente, 4 bolhas)."""
    xp = bk.xp
    codes = xp.asarray([PATTERN_CODES[p] for p in PATTERNS], float)
    return codes[xp.argmax(weights, axis=1)]


__all__ = [
    "PATTERNS",
    "PATTERN_CODES",
    "ClosureOptions",
    "ClosureResult",
    "FlowInputs",
    "flow_closure",
    "pattern_results",
    "pattern_weights",
    "pattern_code",
    "smooth_step",
]
