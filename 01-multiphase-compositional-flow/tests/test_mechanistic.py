"""Modelo mecanicista por diferenças finitas: operador de derivada, Jacobiana colorida = densa, e a
consistência forte com a marcha de não escorregamento (bolhas fixas com C_0 = 1 no horizontal
são, por construção, o modelo de não escorregamento)."""

import json

import numpy as np
import pytest

from cmfsim import Case, simulate
from cmfsim.backend import NUMPY
from cmfsim.models.mechanistic import gradient_matrix
from cmfsim.simulation import build_mechanistic, march_noslip
from cmfsim.solvers.newton import assemble_from_colors, band_colors, colored_jacobian
from generators.condensate import base_case


def test_gradient_matrix_is_second_order():
    x = np.linspace(0.0, 1.0, 11)
    f = np.sin(3.0 * x)
    df = 3.0 * np.cos(3.0 * x)
    err2 = np.max(np.abs(gradient_matrix(x, 2) @ f - df))
    x2 = np.linspace(0.0, 1.0, 21)
    err2_fine = np.max(np.abs(gradient_matrix(x2, 2) @ np.sin(3.0 * x2) - 3.0 * np.cos(3.0 * x2)))
    assert err2 < 0.1 and err2_fine < err2 / 3.0  # ~h^2
    assert np.allclose(gradient_matrix(x, 2) @ x**2, 2 * x, atol=1e-12)  # exato em polinómios de grau 2
    assert np.allclose(gradient_matrix(x, 1) @ x, 1.0, atol=1e-12)
    # equivalente ao numpy.gradient com edge_order=2
    assert np.allclose(gradient_matrix(x, 2) @ f, np.gradient(f, x, edge_order=2), atol=1e-12)


def _condensing_case(n=12, **model):
    d = base_case(dx_out=250.0)
    d["pipe"]["profile"][0].update(U=6.0, T_amb=288.15)
    dm = json.loads(json.dumps(d))
    dm["model"] = {"type": "mechanistic", "mass_split": "algebraic", "regime": {"regime": "bubble", "bubble_C0": 1.0}}
    dm["model"].update(model)
    dm["solver"] = {
        "type": "fd",
        "n": n,
        "initial_guess": "march_noslip",
        "rtol": 1e-7,
        "atol": 1e-7,
        "first_step": 10.0,
        "backend": "numpy",
        "newton": {"max_iter": 30, "tol": 1e-10},
    }
    return d, dm


def test_colored_jacobian_equals_dense():
    _, dm = _condensing_case(n=6)
    case = Case.from_dict(dm)
    model, params = build_mechanistic(case, NUMPY)
    u0 = model.initial_guess(params)
    colors, n_colors, mask = model.sparsity()
    V, seeds = colored_jacobian(lambda u: model.residual(u, params), u0, colors, n_colors, NUMPY, 1e-6)
    Jc = assemble_from_colors(V, seeds, mask, NUMPY)
    n = u0.size
    Jd = np.zeros((n, n))
    F0 = model.residual(u0, params)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1e-6
        Jd[:, j] = (model.residual(u0 + e, params) - F0) / 1e-6
    assert np.max(np.abs(Jc - Jd)) < 1e-9 * max(1.0, np.max(np.abs(Jd)))
    assert np.max(np.abs(np.where(mask, 0.0, Jd))) == 0.0  # nada fora da banda


def test_fd_bubble_c0_one_matches_noslip_march():
    d, dm = _condensing_case(n=16)
    ref = simulate(Case.from_dict(d), backend="numpy", progress=False)
    res = simulate(Case.from_dict(dm), backend="numpy", progress=False)
    assert res.info["converged"] and res.info["iterations"] <= 6
    Pi = np.interp(res.x, ref.x, ref.P)
    Ti = np.interp(res.x, ref.x, ref.T)
    wi = np.interp(res.x, ref.x, ref["w_G"])
    assert np.max(np.abs(res.P - Pi)) < 30.0  # Pa (erro de discretização com 17 nós)
    assert np.max(np.abs(res.T - Ti)) < 0.05
    assert np.max(np.abs(res["w_G"] - wi)) < 5e-4
    assert np.allclose(res["H_L"], res["H_L_noslip"], atol=1e-6)
    assert res["w_G"][-1] < 1.0  # condensa


def test_fd_jax_compiles_and_converges():
    pytest.importorskip("jax")
    _, dm = _condensing_case(n=16)
    res = simulate(Case.from_dict(dm), backend="jax", progress=False)
    assert res.info["converged"] and res.info["backend"] == "jax"
    res_np = simulate(Case.from_dict(dm), backend="numpy", progress=False)
    assert np.max(np.abs(res.P - res_np.P)) < 1.0 and np.max(np.abs(res.T - res_np.T)) < 1e-3
