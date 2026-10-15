"""Identificação do padrão de escoamento (§2.3, Tabela 1, Figura 4): critérios de Taitel & Dukler
(1976) no horizontal/inclinado (32–34), avaliados no equilíbrio estratificado, e de Taitel et al.
(1980) no vertical ascendente (35–38); chave horizontal/vertical a 45° (p. 51) com mistura
opcional por ângulo; comparações booleanas substituídas por sigmoides relativas (p. 56, Souza
2010). Os pesos são (estratificado, anular, bolhas, intermitente) e somam 1.

Leituras do texto (issue 003): (34) com (rho_L - rho_G)/rho_L de Taitel & Dukler ou rho_G/rho_L como
impresso; (36) na forma de Taitel (v_SL > 3 v_SG - 1,15[...]^1/4) ou como impresso; (37) com >= 4,36
(Taitel: bolhas só em tubos largos) ou <= como impresso."""

from __future__ import annotations

from ..backend import NUMPY
from .common import G, FlowInputs, smooth_step
from .friction import fanning, vabs
from .stratified import stratified_terms


def horizontal_weights(inp: FlowInputs, hD_eq, opts, bk=NUMPY, **strat_kw):
    """Pesos (estratificado, anular, bolhas, intermitente) pelos critérios horizontais, com o nível
    de equilíbrio estratificado hD_eq."""
    xp = bk.xp
    t = stratified_terms(hD_eq, inp, bk=bk, **strat_kw)
    k = opts.k_switch
    cos_t = xp.cos(inp.theta)
    # (32) Kelvin–Helmholtz: estratificado -> não estratificado
    v_kh = (1.0 - hD_eq) * bk.safe_sqrt(
        (inp.rho_L - inp.rho_G) * G * cos_t * t["AG"] / (inp.rho_G * xp.maximum(t["SI"], 1e-12 * inp.D))
    )
    v_gas = t["vG"] if opts.kh_velocity == "actual" else inp.v_SG
    w_ns = smooth_step(vabs(v_gas, bk) / xp.maximum(v_kh, 1e-300) - 1.0, k, bk)
    # (33) h_L/D < 0,5: anular (senão intermitente ou bolhas)
    w_ann = smooth_step((0.5 - hD_eq) / 0.5, k, bk)
    # (34) bolhas dispersas
    DL = 4.0 * t["AL"] / xp.maximum(t["SL"], 1e-12 * inp.D)
    fL, _ = fanning(t["vL"], inp.rho_L, inp.mu_L, DL, inp.eps, bk=bk)
    fL = fL * (1.0 if inp.f_wall_mult is None else inp.f_wall_mult)
    dens = (
        xp.maximum(inp.rho_L - inp.rho_G, 1e-6) / inp.rho_L
        if opts.dispersed_density == "taitel"
        else inp.rho_G / inp.rho_L
    )
    v_db = bk.safe_sqrt(4.0 * t["AG"] / xp.maximum(t["SI"], 1e-12 * inp.D) * G * cos_t / xp.maximum(fL, 1e-300) * dens)
    w_bub = smooth_step(vabs(t["vL"], bk) / xp.maximum(v_db, 1e-300) - 1.0, k, bk)
    strat = 1.0 - w_ns
    ann = w_ns * w_ann
    bub = w_ns * (1.0 - w_ann) * w_bub
    inter = w_ns * (1.0 - w_ann) * (1.0 - w_bub)
    return strat, ann, bub, inter


def vertical_weights(inp: FlowInputs, opts, bk=NUMPY):
    """Pesos (0, anular, bolhas, intermitente) pelos critérios verticais de Taitel et al. (1980)."""
    xp = bk.xp
    k = opts.k_switch
    drho = xp.maximum(inp.rho_L - inp.rho_G, 1e-6)  # piso: numa fase densa única as raízes coincidem
    c4 = bk.safe_pow(inp.sigma * G * drho, 0.25)
    # (35) anular
    v_ann = 3.1 * c4 / bk.safe_sqrt(inp.rho_G)
    w_ann = smooth_step(inp.v_SG / xp.maximum(v_ann, 1e-300) - 1.0, k, bk)
    # (36) bolhas -> intermitente
    vb = 1.15 * bk.safe_pow(G * drho * inp.sigma / inp.rho_L**2, 0.25)
    if opts.bubble_slug == "taitel":
        rhs = 3.0 * inp.v_SG - vb
    else:  # como impresso (sem o fator 3)
        rhs = inp.v_SG - vb
    scale = xp.maximum(xp.abs(rhs), 1e-3)
    w_b1 = smooth_step((inp.v_SL - rhs) / scale, k, bk)
    # (37) existência de bolhas (diâmetro)
    nd = bk.safe_pow(inp.rho_L**2 * G * inp.D**2 / (drho * inp.sigma), 0.25)
    if opts.bubble_existence == "taitel":
        w_ex = smooth_step(nd / 4.36 - 1.0, k, bk)
    else:  # como impresso: <= 4,36
        w_ex = smooth_step(1.0 - nd / 4.36, k, bk)
    # (38) bolhas dispersas
    nuL = inp.mu_L / inp.rho_L
    v_disp = (
        4.0 * (inp.D**0.429 * (inp.sigma / inp.rho_L) ** 0.089 / nuL**0.072 * (G * drho / inp.rho_L) ** 0.446)
        - inp.v_SG
    )
    scale = xp.maximum(xp.abs(v_disp), 1e-3)
    w_disp = smooth_step((inp.v_SL - v_disp) / scale, k, bk)
    if opts.dispersed_max_void is not None:
        # Taitel et al. (1980): bolhas dispersas só até a fração de vazio 0,52 (omitido na dissertação)
        a = opts.dispersed_max_void
        w_disp = w_disp * smooth_step((inp.v_SL - inp.v_SG * (1.0 - a) / a) / xp.maximum(inp.v_M, 1e-3), k, bk)
    w_bub = 1.0 - (1.0 - w_b1 * w_ex) * (1.0 - w_disp)  # (36 e 37) ou (38)
    ann = w_ann
    bub = (1.0 - w_ann) * w_bub
    inter = (1.0 - w_ann) * (1.0 - w_bub)
    return xp.zeros_like(ann), ann, bub, inter


def vertical_fraction(theta, opts, bk=NUMPY):
    """Peso do conjunto vertical: degrau em `vertical_angle_deg` ou rampa C1 de meia-largura `blend_deg`."""
    xp = bk.xp
    deg = xp.abs(theta) * 180.0 / xp.pi
    if opts.blend_deg <= 0.0:
        return xp.where(deg >= opts.vertical_angle_deg, 1.0, 0.0)
    u = xp.clip((deg - (opts.vertical_angle_deg - opts.blend_deg)) / (2.0 * opts.blend_deg), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)
