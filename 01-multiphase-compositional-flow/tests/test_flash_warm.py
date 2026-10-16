"""Arranque quente do flash (issue 007): Rachford–Rice nos extremos do intervalo, solução trivial,
identidade frio × quente e derivadas com arranque quente."""

import numpy as np
import pytest

from cmfsim.backend import NUMPY, get_backend
from cmfsim.thermo import FlashOptions, phase_state
from cmfsim.thermo.flash import flash_pt, rachford_rice, rr_bounds

T4 = np.array([300.0, 320.0, 340.0, 360.0])
P4 = np.array([40e5, 50e5, 60e5, 70e5])


def _rr_residual(z, K, beta):
    return np.sum(z * (K - 1.0) / (1.0 + beta[:, None] * (K - 1.0)), axis=1)


def test_rachford_rice_random_and_edge_cases():
    # raízes genuínas em sistemas aleatórios (negativo incluído) a 1e-12, numpy e JAX iguais
    rng = np.random.default_rng(7)
    n, nc = 200, 8
    z = rng.dirichlet(np.ones(nc), size=n)
    K = np.exp(rng.normal(0.0, 2.0, size=(n, nc)))
    beta = rachford_rice(z, K, NUMPY)
    lo, hi = rr_bounds(K, NUMPY)
    assert np.all((beta > lo) & (beta < hi))
    # raiz genuína só quando há K > 1 e K < 1; sem raiz a bisseção encosta a um limite
    has_root = (K.max(axis=1) > 1.0) & (K.min(axis=1) < 1.0)
    assert has_root.sum() > 150
    assert np.max(np.abs(_rr_residual(z, K, beta)[has_root])) < 1e-12
    edge = ~has_root
    assert np.all((beta[edge] < lo[edge] + 1e-6 * (hi - lo)[edge]) | (beta[edge] > hi[edge] - 1e-6 * (hi - lo)[edge]))
    jax = pytest.importorskip("jax")
    JAX = get_backend("jax")  # ativa float64 antes de criar os arrays
    import jax.numpy as jnp

    bj = np.asarray(jax.jit(lambda z, K: rachford_rice(z, K, JAX))(jnp.asarray(z), jnp.asarray(K)))
    assert np.max(np.abs(bj - beta)) < 1e-13
    # raiz num extremo do intervalo de bisseção: Newton inclusivo não salta para o ponto médio
    z1 = np.array([[0.5, 0.5]])
    K1 = np.array([[2.0, 0.5]])  # raiz exata beta = 0,5
    b = rachford_rice(z1, K1, NUMPY)
    assert abs(b[0] - 0.5) < 1e-14
    # solução trivial (K = 1): f = 0 em todo o beta; fica no ponto de partida sem NaN
    b = rachford_rice(np.array([[0.3, 0.7]]), np.ones((1, 2)), NUMPY)
    assert np.isfinite(b[0]) and b[0] == 0.5


def test_trivial_solution_is_single_phase(fluid_table6):
    # puros e misturas supercríticas colapsam em K = 1: nunca bifásico, rótulo pela T pseudocrítica
    cs, z = fluid_table6
    names = cs.names
    for comp, T, want in (("C1", 300.0, 1.0), ("nC8", 300.0, 0.0), ("C1", 150.0, 0.0)):
        zz = np.array([1.0 if n == comp else 0.0 for n in names])
        st = phase_state(np.array([T]), np.array([1e5 if comp == "nC8" or T > 200 else 50e5]), zz, cs)
        assert st.two_phase[0] == 0.0 and st.beta[0] == want, (comp, T)


def test_warm_start_identity_and_iterations(fluid_table19):
    cs, z = fluid_table19
    cold = flash_pt(T4, P4, np.broadcast_to(z, (4, z.size)), cs)
    assert cold["two_phase"].tolist() == [1.0, 1.0, 1.0, 1.0]
    # a partir do próprio K convergido: uma substituição e o mesmo resultado a precisão de máquina
    same = flash_pt(T4, P4, np.broadcast_to(z, (4, z.size)), cs, lnK0=cold["lnK"])
    assert np.all(same["iterations"] <= 2) and np.all(same["iterations"] < cold["iterations"])
    assert np.max(np.abs(same["beta"] - cold["beta"])) < 1e-13
    assert np.max(np.abs(np.log(same["K"]) - np.log(cold["K"]))) < 1e-12
    # a partir do K do estado vizinho (como ao longo de x): menos iterações, mesma solução
    nb = flash_pt(T4 + 2.0, P4 - 1e5, np.broadcast_to(z, (4, z.size)), cs, lnK0=cold["lnK"])
    ref = flash_pt(T4 + 2.0, P4 - 1e5, np.broadcast_to(z, (4, z.size)), cs)
    assert np.all(nb["iterations"] <= ref["iterations"])
    assert np.max(np.abs(nb["beta"] - ref["beta"])) < 1e-12
    # lnK0 inutilizável (não finito ou trivial) volta a Wilson: mesmo resultado do arranque frio
    bad = np.full((4, z.size), np.nan)
    bad[1] = 0.0
    fb = flash_pt(T4, P4, np.broadcast_to(z, (4, z.size)), cs, lnK0=bad)
    assert np.max(np.abs(fb["beta"] - cold["beta"])) < 1e-13
    # fase única: o K guardado é Wilson (reentrar na região bifásica a partir dele é o arranque frio)
    one = flash_pt(np.array([280.0]), np.array([500e5]), z[None, :], cs)
    assert one["two_phase"][0] == 0.0
    from cmfsim.thermo.flash import wilson_K

    assert np.allclose(one["lnK"][0], np.log(wilson_K(np.array([280.0]), np.array([500e5]), cs, NUMPY))[0])


def test_warm_start_keeps_exact_derivatives(fluid_table19):
    # o passo final de Newton incondicional transporta a derivada (TFI) mesmo quando o arranque
    # quente já cai sobre a solução: jacfwd igual ao do arranque frio
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    cs, z = fluid_table19
    JAX = get_backend("jax")
    zz = jnp.asarray(np.broadcast_to(z, (4, z.size)))
    cold = phase_state(T4, P4, np.broadcast_to(z, (4, z.size)), cs)
    lnK0 = jnp.asarray(cold.lnK)

    def props(T, P, lnK0):
        st = phase_state(T, P, zz, cs, FlashOptions(), JAX, lnK0=lnK0)
        return jnp.stack([st.w_G, st.rho_L, st.rho_V, st.h_L, st.h_V], axis=0)

    jac = jax.jit(jax.jacfwd(props, argnums=(0, 1)))
    dT_c, dP_c = jac(jnp.asarray(T4), jnp.asarray(P4), None)
    dT_w, dP_w = jac(jnp.asarray(T4), jnp.asarray(P4), lnK0)
    for a, b in ((dT_c, dT_w), (dP_c, dP_w)):
        a, b = np.asarray(a), np.asarray(b)
        assert np.max(np.abs(a - b)) < 1e-8 * np.max(np.abs(a))
        assert np.max(np.abs(a)) > 0.0
