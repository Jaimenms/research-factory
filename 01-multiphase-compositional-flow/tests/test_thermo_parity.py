"""Paridade numpy × JAX, derivadas por diferenciação automática, e sanidade física da termodinâmica."""

import numpy as np
import pytest

from cmfsim.backend import get_backend
from cmfsim.thermo import FlashOptions, phase_state, pure_cp_ideal, pure_h_ideal
from cmfsim.thermo.ideal import T_REF

T5 = np.array([300.0, 350.0, 400.0, 450.0, 300.0])
P5 = np.array([20e5, 50e5, 80e5, 30e5, 150e5])


def test_polish_reaches_machine_precision(fluid_table6):
    cs, z = fluid_table6
    st0 = phase_state(T5, P5, z, cs, FlashOptions(tol=1e-12, polish_iter=0))  # a dissertação tal e qual
    st2 = phase_state(T5, P5, z, cs, FlashOptions())
    assert np.all(st0.residual[st0.two_phase > 0] < 1e-12)
    assert np.all(st2.residual[st2.two_phase > 0] < 1e-24)
    assert np.max(np.abs(st2.beta - st0.beta)) < 1e-6
    assert np.all(st2.iterations[st2.two_phase > 0] <= 30) and np.all(st2.iterations <= 200)
    assert st2.two_phase.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_numpy_jax_parity(fluid_table6):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    cs, z = fluid_table6
    JAX = get_backend("jax")
    st = phase_state(T5, P5, z, cs, FlashOptions())
    zz = jnp.asarray(np.broadcast_to(z, (5, z.size)))
    f = jax.jit(lambda T, P, zz: phase_state(T, P, zz, cs, FlashOptions(), JAX))
    stj = f(jnp.asarray(T5), jnp.asarray(P5), zz)
    for nm in ("beta", "w_G", "rho_L", "rho_V", "h_L", "h_V", "mu_L", "mu_V", "sigma", "x", "y"):
        a, b = np.asarray(getattr(stj, nm)), getattr(st, nm)
        assert np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)) < 1e-10, nm


def test_autodiff_matches_finite_differences(fluid_table6):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    cs, z = fluid_table6
    JAX = get_backend("jax")
    zz = jnp.asarray(np.broadcast_to(z, (5, z.size)))

    def props(T, P):
        st = phase_state(T, P, zz, cs, FlashOptions(), JAX)
        return jnp.stack([st.w_G, st.rho_L, st.rho_V, st.h_L, st.h_V], axis=0)

    dP = np.asarray(jax.jacfwd(props, argnums=1)(jnp.asarray(T5), jnp.asarray(P5)))
    dT = np.asarray(jax.jacfwd(props, argnums=0)(jnp.asarray(T5), jnp.asarray(P5)))
    dP = np.stack([np.diag(dP[i]) for i in range(5)])
    dT = np.stack([np.diag(dT[i]) for i in range(5)])

    def props_np(T, P):
        st = phase_state(T, P, z, cs, FlashOptions())
        return np.stack([st.w_G, st.rho_L, st.rho_V, st.h_L, st.h_V], axis=0)

    h, hT = 10.0, 1e-3
    fdP = (props_np(T5, P5 + h) - props_np(T5, P5 - h)) / (2 * h)
    fdT = (props_np(T5 + hT, P5) - props_np(T5 - hT, P5)) / (2 * hT)
    scale = np.max(np.abs(fdP), axis=1, keepdims=True)
    assert np.max(np.abs(dP - fdP) / scale) < 1e-5
    scale = np.max(np.abs(fdT), axis=1, keepdims=True)
    assert np.max(np.abs(dT - fdT) / scale) < 1e-5


def test_ideal_gas_enthalpy_is_integral_of_cp(fluid_table6):
    cs, _ = fluid_table6
    T = np.linspace(T_REF, 500.0, 2001)
    cp = pure_cp_ideal(T, cs)
    h = pure_h_ideal(T, cs)
    integ = np.concatenate([np.zeros((1, cs.n)), np.cumsum(0.5 * (cp[1:] + cp[:-1]) * np.diff(T)[:, None], axis=0)])
    assert np.max(np.abs(h - h[0] - integ)) < 2e-5 * np.max(np.abs(h - h[0]))
    assert np.allclose(h[0], cs.Hf)


def test_physical_sanity(fluid_table6):
    cs, z = fluid_table6
    T = np.array([300.0, 300.0])
    P = np.array([1e5, 300e5])
    names = cs.names
    # metano gás a 300 K e 1 bar: Z ~ 1, viscosidade ~1.1e-5 Pa s; octano líquido a 300 K: ~0.5 mPa s, sigma ~21 mN/m
    zC1 = np.array([1.0 if n == "C1" else 0.0 for n in names])
    zC8 = np.array([1.0 if n == "nC8" else 0.0 for n in names])
    stC1 = phase_state(T[:1], P[:1], zC1, cs)
    assert stC1.beta[0] == 1.0 and 0.9e-5 < stC1.mu_V[0] < 1.3e-5
    assert abs(stC1.rho_V[0] - 1e5 * 16.0 / (8314.46 * 300.0)) < 0.02
    stC8 = phase_state(T[:1], P[:1], zC8, cs)
    assert stC8.beta[0] == 0.0 and 4e-4 < stC8.mu_L[0] < 6e-4 and 0.018 < stC8.sigma[0] < 0.024
    assert 640 < stC8.rho_L[0] < 720  # PR subestima (673 contra 698 medidos)
    # mistura: líquido mais denso que o vapor, titulo entre 0 e 1 na região bifásica, entalpia do vapor > líquido
    st = phase_state(T5, P5, z, cs)
    assert np.all(st.rho_L[st.two_phase > 0] > st.rho_V[st.two_phase > 0])
    assert np.all((st.w_G[st.two_phase > 0] > 0) & (st.w_G[st.two_phase > 0] < 1))
    # entalpias por fase incluem a formação: só comparáveis a composição igual; no octano puro a
    # fase vapor (incipiente) tem h_V - h_L = calor latente ~ 363 kJ/kg a 25 °C
    assert 250e3 < stC8.h_V[0] - stC8.h_L[0] < 450e3
