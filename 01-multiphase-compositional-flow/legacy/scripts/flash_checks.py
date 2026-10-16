"""Verificações da termodinâmica do código original (L02).

1. ln φ do código contra a forma padrão de Peng–Robinson no mesmo estado (L02.1);
2. flash da §5.5 nos estados de 200, 4000 e 7000 m: original, com 2√2 e 1 ± √2 exatos, e com ln φ
   padrão (a_ij = √(a_i a_j) e sinal −), injetado em memória (L02.1);
3. varrimento do flash do gás rico da §5.6 a 69,1 bar perto do orvalho: solução trivial (L02.2).

Uso: .venv-legacy/bin/python legacy/scripts/flash_checks.py
"""

import inspect
import textwrap

import _bootstrap as B  # noqa: F401  (caminhos e substitutos)
import numpy as np

import app.thermodynamic_v2.equation_of_state.equation_of_state as eosm
import app.thermodynamic_v2.flash.flash2_via_ss_normal as fm
from app.thermodynamic_v2.components_properties import ComponentsProperties
from app.thermodynamic_v2.stream import Stream

EOS_SRC = inspect.getsource(eosm.EquationOfState.__call__)
AIJ_ORIG = eosm.EquationOfState.build_aij

Z19 = {
    "CO2": 0.007,
    "N2": 0.011,
    "C1": 0.696,
    "C2": 0.103,
    "C3": 0.049,
    "nC4": 0.032,
    "nC5": 0.022,
    "nC6": 0.007,
    "C20pseud": 0.073,
}
MW19 = {
    "CO2": 44.01,
    "N2": 28.014,
    "C1": 16.043,
    "C2": 30.07,
    "C3": 44.097,
    "nC4": 58.123,
    "nC5": 72.15,
    "nC6": 86.177,
    "C20pseud": 560.0,
}


def set_lnphi(mode):
    """mode: 'original' | 'exact_constants' | 'standard'."""
    src = EOS_SRC
    if mode == "exact_constants":
        src = (
            src.replace("(2.82*self.B[i])", "(2*np.sqrt(2)*self.B[i])")
            .replace("2.414*self.B[i]", "(1+np.sqrt(2))*self.B[i]")
            .replace("0.414*self.B[i]", "(np.sqrt(2)-1)*self.B[i]")
        )
    if mode == "standard":
        src = src.replace("+ (self.A[i]/(2.82*self.B[i]))", "- (self.A[i]/(2*np.sqrt(2)*self.B[i]))")
    ns = dict(eosm.__dict__)
    exec(textwrap.dedent(src), ns)
    eosm.EquationOfState.__call__ = ns["__call__"]
    if mode == "standard":
        eosm.EquationOfState.build_aij = staticmethod(lambda ap, kij: (1 - kij) * np.outer(np.sqrt(ap), np.sqrt(ap)))
    else:
        eosm.EquationOfState.build_aij = staticmethod(AIJ_ORIG)


def lnphi_vs_standard(T=328.31, P=116.0):
    set_lnphi("original")
    comps = tuple(Z19)
    x = np.array([Z19[c] for c in comps]).reshape(-1, 1)
    MM = np.array([MW19[c] for c in comps]).reshape(-1, 1)
    st = Stream(ComponentsProperties(comps))
    st.specify(x * MM * 1e-3, np.array([T]), np.array([P]))
    st.calculate_eos(x, np.array([T]), np.array([P]))
    e = st.eos
    ap, bp, am, bm, A, Bb, Z = e.ap[:, 0], e.bp, e.am[0], e.bm[0], e.A[0], e.B[0], e.Z[0]
    sa = np.sqrt(ap)
    std = (
        bp / bm * (Z - 1)
        - np.log(Z - Bb)
        - A
        / (2 * np.sqrt(2) * Bb)
        * (2 * sa * np.sum(x[:, 0] * sa) / am - bp / bm)
        * np.log((Z + (1 + np.sqrt(2)) * Bb) / (Z + (1 - np.sqrt(2)) * Bb))
    )
    print(f"1. ln φ a {P} bar, {T} K (composição da Tabela 19), Z = {Z:.4f}")
    for c, lo, ls in zip(comps, e.lnphi[:, 0], std):
        print(f"   {c:9s} código {lo:+9.3f}   PR padrão {ls:+9.3f}")


def flash_55():
    comps = tuple(Z19)
    w = np.array([Z19[c] * MW19[c] for c in comps])
    mass = (w / w.sum() * 10).reshape(-1, 1)
    print("2. flash da §5.5 (β molar, ρ_L, λ_L no-slip)")
    for mode in ("original", "exact_constants", "standard"):
        set_lnphi(mode)
        for P, TC in [(162.8, 78.35), (116.0, 55.16), (112.0, 43.6)]:
            st, liq, vap, K, v, info = fm.Flash2ViaSS_normal(Stream(ComponentsProperties(comps)))(
                mass, np.array([TC + 273.15]), np.array([P])
            )
            qL = liq.total_flowrate / liq.properties.density
            qV = vap.total_flowrate / vap.properties.density
            print(
                f"   {mode:16s} {P:6.1f} bar {TC:5.2f} °C  β {float(np.ravel(v)[0]):.4f}  ρ_L {float(liq.properties.density[0]):6.1f}  λ_L {float(qL / (qL + qV)):.4f}"
            )
    set_lnphi("original")


def dew_scan_56(P=69.1):
    comps = ("C1", "C2", "C3", "nC4", "nC5")
    mass = (np.array([0.5, 0.2, 0.15, 0.1, 0.05]) * 15).reshape(-1, 1)
    print(f"3. flash original do gás rico da §5.6 a {P} bar (orvalho de PR: 34,3 °C)")
    set_lnphi("original")
    for TC in np.arange(36.0, 30.9, -0.25):
        st, liq, vap, K, v, info = fm.Flash2ViaSS_normal(Stream(ComponentsProperties(comps)))(
            mass, np.array([TC + 273.15]), np.array([P])
        )
        lnK = float(np.max(np.abs(np.log(np.ravel(K)))))
        state = (
            "trivial (K → 1), declarado vapor"
            if lnK < 1e-2
            else ("bifásico" if 0 < float(np.ravel(v)[0]) < 1 else "monofásico")
        )
        print(f"   {TC:5.2f} °C  β {float(np.ravel(v)[0]):.4f}  max|ln K| {lnK:.3g}  {state}")


if __name__ == "__main__":
    lnphi_vs_standard()
    flash_55()
    dew_scan_56()
