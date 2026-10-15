"""Termodinâmica contra um oráculo independente: a biblioteca `thermo` (Caleb Bell) com Peng–Robinson.
Com as constantes exatas de PR (`with_exact_pr_constants`) a concordância é de 1e-9; com as da
dissertação a diferença é ~1e-5 (documentado na issue 001)."""

import numpy as np
import pytest

pytestmark = pytest.mark.reference

thermo = pytest.importorskip("thermo")

from cmfsim.thermo import FlashOptions, ln_phi, phase_state, residual_enthalpy  # noqa: E402
from cmfsim.thermo.pr import phase_Z  # noqa: E402


def _flasher(cs):
    from thermo import (
        CEOSGas,
        CEOSLiquid,
        ChemicalConstantsPackage,
        FlashVL,
        HeatCapacityGas,
        PRMIX,
        PropertyCorrelationsPackage,
    )

    kw = dict(Tcs=list(cs.Tc), Pcs=list(cs.Pc), omegas=list(cs.omega))
    constants = ChemicalConstantsPackage(MWs=list(cs.MW), names=list(cs.names), **kw)
    hcs = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [0.0] * 7 + [30.0])) for _ in cs.names]
    props = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=hcs)
    gas = CEOSGas(PRMIX, eos_kwargs=kw, HeatCapacityGases=hcs)
    liq = CEOSLiquid(PRMIX, eos_kwargs=kw, HeatCapacityGases=hcs)
    return FlashVL(constants, props, liquid=liq, gas=gas)


def test_pure_methane_and_octane_against_thermo(fluid_table6):
    from thermo.eos import PR

    cs = fluid_table6[0].with_exact_pr_constants()
    for name, T, P in (("C1", 300.0, 50e5), ("nC8", 400.0, 5e5), ("nC8", 300.0, 1e5)):
        i = cs.names.index(name)
        ref = PR(Tc=cs.Tc[i], Pc=cs.Pc[i], omega=cs.omega[i], T=T, P=P)
        x = np.zeros((1, cs.n))
        x[0, i] = 1.0
        for phase, suf in (("vapor", "g"), ("liquid", "l")):
            Zref = getattr(ref, f"Z_{suf}", None)
            if Zref is None:
                continue
            Z = phase_Z(np.array([T]), np.array([P]), x, cs, phase)[0][0]
            assert abs(Z - Zref) < 1e-9 * Zref
            lnphi = ln_phi(np.array([T]), np.array([P]), x, cs, phase)[0, i]
            assert abs(lnphi - getattr(ref, f"lnphi_{suf}")) < 1e-9
            hres = residual_enthalpy(np.array([T]), np.array([P]), x, cs, phase)[0]
            assert abs(hres / 1e3 - getattr(ref, f"H_dep_{suf}")) < 1e-7 * abs(getattr(ref, f"H_dep_{suf}"))


def test_mixture_lnphi_against_thermo(fluid_table6):
    from thermo.eos_mix import PRMIX

    cs, z = fluid_table6
    cs = cs.with_exact_pr_constants()
    T, P = 350.0, 50e5
    ref = PRMIX(Tcs=list(cs.Tc), Pcs=list(cs.Pc), omegas=list(cs.omega), zs=list(z), T=T, P=P)
    for phase, suf in (("vapor", "g"), ("liquid", "l")):
        lref = getattr(ref, f"lnphis_{suf}", None)
        if lref is None:
            continue
        mine = ln_phi(np.array([T]), np.array([P]), z[None, :], cs, phase)[0]
        assert np.max(np.abs(mine - np.array(lref))) < 1e-9
        hres = residual_enthalpy(np.array([T]), np.array([P]), z[None, :], cs, phase)[0] / 1e3
        assert abs(hres - getattr(ref, f"H_dep_{suf}")) < 1e-7 * abs(getattr(ref, f"H_dep_{suf}"))


GRIDS = {
    "fluid_table6": (
        [280.0, 300.0, 330.0, 360.0, 400.0, 450.0, 300.0, 350.0],
        [5e5, 20e5, 40e5, 60e5, 80e5, 30e5, 150e5, 100e5],
    ),
    "fluid_table19": (
        [280.0, 300.0, 330.0, 360.0, 400.0, 450.0, 300.0, 350.0],
        [5e5, 20e5, 40e5, 60e5, 80e5, 30e5, 150e5, 100e5],
    ),
    # gás rico: a região bifásica (retrógrada) fica abaixo de ~300 K entre 30 e 80 bar
    "fluid_table23": (
        [250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 330.0, 300.0],
        [30e5, 40e5, 50e5, 60e5, 70e5, 70e5, 40e5, 150e5],
    ),
}


def _tpd(T, P, z, y, cs):
    """Distância ao plano tangente de Michelsen da fase única z (raiz líquida) com a tentativa y."""
    lz = ln_phi(np.array([T]), np.array([P]), z[None, :], cs, "liquid")[0]
    ly = ln_phi(np.array([T]), np.array([P]), y[None, :], cs, "vapor")[0]
    m = (y > 0) & (z > 0)
    return float(np.sum(y[m] * (np.log(y[m]) + ly[m] - np.log(z[m]) - lz[m])))


@pytest.mark.parametrize("fluid_name", ["fluid_table6", "fluid_table23", "fluid_table19"])
def test_flash_against_thermo(fluid_name, request):
    cs, z = request.getfixturevalue(fluid_name)
    cs = cs.with_exact_pr_constants()
    flasher = _flasher(cs)
    Ts, Ps = (np.array(v) for v in GRIDS[fluid_name])
    st = phase_state(Ts, Ps, z, cs, FlashOptions())
    n_two = 0
    for k, (T, P) in enumerate(zip(Ts, Ps)):
        r = flasher.flash(T=float(T), P=float(P), zs=list(z))
        gas = getattr(r, "gas", None)
        liqs = getattr(r, "liquids", None) or []
        liq = liqs[0] if liqs else None
        VF = 0.0 if r.VF is None else float(r.VF)
        if gas is not None and liq is not None and 0.0 < VF < 1.0:
            n_two += 1
            assert abs(st.beta[k] - VF) < 1e-7, (fluid_name, T, P, st.beta[k], VF)
            assert np.max(np.abs(st.x[k] - np.array(liq.zs))) < 1e-7
            assert np.max(np.abs(st.y[k] - np.array(gas.zs))) < 1e-7
            assert abs(st.rho_L[k] - liq.rho_mass()) < 1e-6 * liq.rho_mass()
            assert abs(st.rho_V[k] - gas.rho_mass()) < 1e-6 * gas.rho_mass()
        elif st.two_phase[k] == 1.0:
            # a thermo diz fase única e nós duas: arbitra a estabilidade com a nossa EoS (o
            # fluido com C7+ a 300 K e 150 bar: a thermo falha, TPD = -0.17 e G/RT menor)
            assert _tpd(T, P, z, st.y[k], cs) < -1e-6, (fluid_name, T, P, st.beta[k], VF)
        else:
            # fase única: o rótulo líquido/vapor de um fluido supercrítico é convenção (a thermo
            # decide por critérios próprios); o que tem de coincidir é a massa específica da
            # fase presente
            ref = (gas or liq).rho_mass()
            mine = st.rho_V[k] if st.beta[k] == 1.0 else st.rho_L[k]
            assert abs(mine - ref) < 1e-6 * ref, (fluid_name, T, P, mine, ref)
    assert n_two >= 2, "a grelha de teste deve ter pontos bifásicos"
