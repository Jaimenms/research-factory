"""Integrador próprio (Dormand–Prince) e marcha do modelo de não escorregamento: ordem e tolerância
numa EDO conhecida, paridade numpy (diferenças finitas, como a dissertação) × JAX (AD), balanço
de energia e corrida gravada/relida."""

import json

import numpy as np
import pytest

from cmfsim import Case, load_run, simulate
from cmfsim.backend import get_backend
from cmfsim.solvers.rk45 import RK45Options, integrate
from generators.condensate import base_case


def test_rk45_on_known_ode():
    # u' = [-u1, u0]: rotação (solução cos, sin); e u' = -u (decaimento)
    def rhs(x, u, seg):
        return np.array([-u[1], u[0], -u[2]])

    x_out = np.linspace(0.0, 10.0, 11)
    U, info = integrate(
        rhs, np.array([1.0, 0.0, 1.0]), x_out, np.zeros(10, int), RK45Options(rtol=1e-10, atol=1e-12, first_step=0.1)
    )
    assert np.max(np.abs(U[:, 0] - np.cos(x_out))) < 1e-8
    assert np.max(np.abs(U[:, 1] - np.sin(x_out))) < 1e-8
    assert np.max(np.abs(U[:, 2] - np.exp(-x_out))) < 1e-8
    assert info["n_accepted"] > 20 and info["n_rejected"] >= 0


def test_rk45_jax_parity():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    JAX = get_backend("jax")

    def rhs(x, u, seg):
        return jnp.stack([-u[1], u[0]])

    x_out = jnp.linspace(0.0, 6.0, 7)
    f = jax.jit(
        lambda u0: integrate(
            rhs, u0, x_out, jnp.zeros(6, int), RK45Options(rtol=1e-10, atol=1e-12, first_step=0.1), bk=JAX
        )
    )
    U, info = f(jnp.asarray([1.0, 0.0]))
    assert np.max(np.abs(np.asarray(U[:, 0]) - np.cos(np.asarray(x_out)))) < 1e-8


def _case(**over):
    d = base_case(dx_out=1000.0)
    d["solver"].update(rtol=1e-6, atol=1e-6, first_step=10.0)
    d["solver"].update(over.pop("solver", {}))
    d["model"].update(over.pop("model", {}))
    d["pipe"]["profile"][0]["U"] = over.pop("U", 1.0)
    return Case.from_dict(d)


def test_noslip_single_phase_energy_balance(tmp_path):
    # caso 1 (U = 1, 60 °C): gás sem condensação; T(L) deve seguir o balanço térmico com o cp
    # médio do modelo (exp(-U pi D L/(m cp))) a 1 %, e a pressão cair menos de 1,5 bar
    case = _case(U=1.0, model={"derivatives": "finite_difference"})
    res = simulate(case, progress=False)
    assert res.info["n_accepted"] > 5 and np.all(res["w_G"] == 1.0)
    Tin, Tout, Tamb = res.T[0], res.T[-1], 298.15
    cp = (res["h_mix"][0] - res["h_mix"][-1]) / (Tin - Tout)  # cp efetivo (inclui Joule–Thomson)
    T_pred = Tamb + (Tin - Tamb) * np.exp(-1.0 * np.pi * 0.3048 * 8000.0 / (15.0 * cp))
    assert abs(Tout - T_pred) < 0.01 * (Tin - Tamb)
    assert 0.3e5 < res.P[0] - res.P[-1] < 1.5e5
    out = res.save(tmp_path / "run.npz")
    back = load_run(out)
    assert np.allclose(back.T, res.T) and back.case_dict["name"] == case.name and back.info["backend"] == "numpy"


def test_noslip_numpy_fd_vs_jax_ad():
    pytest.importorskip("jax")
    case_fd = _case(U=3.0, model={"derivatives": "finite_difference"})
    case_ad = _case(U=3.0, model={"derivatives": "autodiff"})
    a = simulate(case_fd, backend="numpy", progress=False)
    b = simulate(case_ad, backend="jax", progress=False)
    assert np.max(np.abs(a.P - b.P)) < 2e2  # Pa (diferenças finitas de 1.ª ordem com dP = 1 kPa)
    assert np.max(np.abs(a.T - b.T)) < 5e-3  # K
    assert b.info["backend"] == "jax"
