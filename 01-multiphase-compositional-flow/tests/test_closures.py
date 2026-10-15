"""Fechamentos mecânicos e identificação de padrão: resíduos nulos nas raízes, estrutura dos mapas
ar–água de Taitel–Dukler / Taitel, formas fechadas, paridade numpy × JAX e derivadas por AD."""

import numpy as np
import pytest

from cmfsim.backend import get_backend
from cmfsim.closures import PATTERNS, ClosureOptions, FlowInputs, flow_closure, pattern_results
from cmfsim.closures.annular import annular_residual
from cmfsim.closures.bubble import bubble_residual
from cmfsim.closures.intermittent import film_residual, slug_holdup, taylor_bubble_velocity
from cmfsim.closures.bubble import harmathy_v0
from cmfsim.closures.stratified import stratified_residual

AIR_WATER = dict(D=0.025, eps=0.00153, rho_L=1000.0, rho_G=1.65, mu_L=1e-3, mu_G=1.8e-5, sigma=0.072)


def inputs(v_SL, v_SG, theta=0.0, bk=None, **over):
    kw = dict(AIR_WATER, **over)
    v_SL = np.atleast_1d(np.asarray(v_SL, float))
    v_SG = np.broadcast_to(np.asarray(v_SG, float), v_SL.shape)
    from cmfsim.backend import NUMPY

    return FlowInputs.build(bk=bk or NUMPY, theta=theta, v_SL=v_SL, v_SG=v_SG, **kw)


def test_equilibria_are_roots():
    inp = inputs([0.1, 1.0, 0.05, 0.3], [1.0, 5.0, 30.0, 0.5], theta=0.0)
    res = pattern_results(inp)
    assert np.max(np.abs(stratified_residual(res["stratified"].level, inp))) < 1e-6
    assert np.max(np.abs(annular_residual(res["annular"].level, inp))) < 1e-6
    assert np.max(np.abs(bubble_residual(res["bubble"].level, inp))) < 1e-12
    HLS = slug_holdup(inp)
    vTB = taylor_bubble_velocity(inp)
    vBS = 1.2 * inp.v_M + harmathy_v0(inp) * HLS**0.1 * np.sin(inp.theta)
    vLS = (inp.v_M - vBS * (1.0 - HLS)) / HLS
    film = film_residual(res["intermittent"].level, inp, HLS, vTB, vLS)
    # no ponto de condições anulares (v_SG = 30) a equação do filme não tem raiz em (0, H_LS):
    # o resolvedor devolve o extremo com menor resíduo (filme = pistão), com peso ~0 na mistura
    assert np.max(np.abs(film[[0, 1, 3]])) < 1e-6
    assert abs(res["intermittent"].level[2] - HLS[2]) < 1e-6 * HLS[2]
    for p in PATTERNS:
        r = res[p]
        assert np.all((r.H_L > 0) & (r.H_L < 1)) and np.all(r.F_friction > 0) and np.all(np.isfinite(r.F))
        assert np.allclose(r.v_L * r.H_L, inp.v_SL, rtol=1e-9) or p in ("annular", "intermittent")


def test_bubble_without_root_falls_to_zero_holdup():
    # lambda_G > 1/C_0: (79) não tem raiz em (0, 1); o extremo sensato é H_L -> 0
    inp = inputs([0.05], [5.0], theta=0.0)
    r = pattern_results(inp)["bubble"]
    assert r.H_L[0] < 1e-5


def test_bubble_horizontal_closed_form():
    inp = inputs([1.0, 3.0], [0.5, 2.0], theta=0.0)
    r = pattern_results(inp)["bubble"]
    lam_G = inp.v_SG / inp.v_M
    assert np.allclose(r.H_L, 1.0 - lam_G / 1.15, atol=1e-9)


def test_stratified_gravity_zero_when_horizontal():
    inp = inputs([0.1], [1.0], theta=0.0)
    r = pattern_results(inp)["stratified"]
    assert r.F_gravity[0] == 0.0 and r.F[0] == r.F_friction[0]


def test_horizontal_map_structure():
    # Taitel & Dukler (1976), ar–água 2,5 cm: estratificado a baixa v_SL, bolhas dispersas a alta v_SL,
    # anular a alta v_SG, intermitente entre eles
    opts = ClosureOptions(k_switch=None)
    VSG = np.logspace(-1, np.log10(500), 20)

    def dominant(v_SL):
        _, w = flow_closure(inputs(np.full(VSG.size, v_SL), VSG, theta=0.0), opts)
        return [PATTERNS[i] for i in np.argmax(w, axis=1)]

    low = dominant(0.01)
    assert low[0] == "stratified" and low[-1] == "annular"
    mid = dominant(0.5)
    assert mid[0] == "intermittent" and mid[-1] == "annular"
    high = dominant(20.0)
    assert high[0] == "bubble"


def test_vertical_map_structure():
    opts = ClosureOptions(k_switch=None, dispersed_max_void=0.52)
    VSG = np.logspace(-1, np.log10(500), 20)
    _, w = flow_closure(inputs(np.full(VSG.size, 0.1), VSG, theta=np.pi / 2), opts)
    dom = [PATTERNS[i] for i in np.argmax(w, axis=1)]
    v_ann = 3.1 * (0.072 * 9.80665 * (1000 - 1.65)) ** 0.25 / 1.65**0.5
    assert all(d == "annular" for d, v in zip(dom, VSG) if v > 1.2 * v_ann)
    assert all(d != "annular" for d, v in zip(dom, VSG) if v < 0.8 * v_ann)
    assert "stratified" not in dom
    _, w = flow_closure(inputs(np.full(VSG.size, 20.0), VSG, theta=np.pi / 2), opts)
    assert PATTERNS[int(np.argmax(w[0]))] == "bubble"


def test_weights_sum_to_one_and_smooth():
    opts = ClosureOptions(k_switch=50.0)
    inp = inputs(np.logspace(-2, 1.5, 30), 3.0, theta=0.0)
    _, w = flow_closure(inp, opts)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.all((w >= 0) & (w <= 1))


def test_numpy_jax_parity_and_autodiff():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    JAX = get_backend("jax")
    opts = ClosureOptions(k_switch=50.0)
    v_SL = np.array([0.05, 0.3, 1.0, 5.0])
    v_SG = np.array([2.0, 1.0, 8.0, 0.5])
    inp = inputs(v_SL, v_SG, theta=0.2)
    res, w = flow_closure(inp, opts)

    def f(vsl, vsg):
        inpj = FlowInputs.build(bk=JAX, theta=0.2, v_SL=vsl, v_SG=vsg, **AIR_WATER)
        r, wj = flow_closure(inpj, opts, JAX)
        return r.H_L, r.F, wj

    fj = jax.jit(f)
    HL, F, wj = fj(jnp.asarray(v_SL), jnp.asarray(v_SG))
    assert np.max(np.abs(np.asarray(HL) - res.H_L) / res.H_L) < 1e-9
    assert np.max(np.abs(np.asarray(F) - res.F) / res.F) < 1e-9
    assert np.max(np.abs(np.asarray(wj) - w)) < 1e-9
    # derivada de H_L e F em relação a v_SL: AD contra diferenças centrais
    dHL = np.diag(np.asarray(jax.jacfwd(lambda a, b: f(a, b)[0])(jnp.asarray(v_SL), jnp.asarray(v_SG))))
    dF = np.diag(np.asarray(jax.jacfwd(lambda a, b: f(a, b)[1])(jnp.asarray(v_SL), jnp.asarray(v_SG))))
    h = 1e-5 * v_SL
    rp, _ = flow_closure(inputs(v_SL + h, v_SG, theta=0.2), opts)
    rm, _ = flow_closure(inputs(v_SL - h, v_SG, theta=0.2), opts)
    fd_HL = (rp.H_L - rm.H_L) / (2 * h)
    fd_F = (rp.F - rm.F) / (2 * h)
    assert np.max(np.abs(dHL - fd_HL) / np.max(np.abs(fd_HL))) < 1e-4
    assert np.max(np.abs(dF - fd_F) / np.max(np.abs(fd_F))) < 1e-4
