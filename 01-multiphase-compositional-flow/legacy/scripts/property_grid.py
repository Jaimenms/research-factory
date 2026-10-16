"""Propriedades «Implementado» das Figuras 13–21 com o código original, como em
tests/validation_thermo_model.py: fluido C1–nC8 sem inertes, grelha P = linspace(1, 140, 20) bar ×
T = linspace(100, 550, 20) K, e os 159 estados de pt_values.csv (L03.4).

Uso: .venv-legacy/bin/python legacy/scripts/property_grid.py
Saída: runs/legacy/property_grid.csv (400 estados) e runs/legacy/property_159.csv.
"""

import contextlib
import io

import _bootstrap as B
import numpy as np
import pandas as pd

from app.edge.pipeline import PipeLine
from app.edge.segment import Segment
from app.hydrodynamic.hydrodynamic import calculate_hydrodynamic
from app.thermodynamic_v2.components_properties import ComponentsProperties
from app.thermodynamic_v2.flash.flash2_via_ss_normal import Flash2ViaSS_normal
from app.thermodynamic_v2.stream import Stream


def main():
    comps = ("C1", "C2", "C3", "nC4", "nC5", "nC6", "nC7", "nC8")
    flow = np.array([0.0968, 0.0075, 0.0183, 0.0269, 0.0290, 0.0323, 0.0538, 0.7355]) * 10
    flash = Flash2ViaSS_normal(Stream(ComponentsProperties(comps)))
    pressures = np.linspace(1.0, 140.0, 20)
    temperatures = np.linspace(100.0, 550.0, 20)
    flow = np.tile(flow.reshape(-1, 1), len(temperatures))
    seg = Segment()
    seg.set_geometry(0.127, 0.00635, 2e-4)
    seg.set_profile(500, 500)
    seg.set_heat(7, 313)
    seg.discretize(3)
    pl = PipeLine()
    pl.append(seg)
    pl.construct()
    pl.pipeline[0].discretize(len(temperatures))
    pl.construct()
    rows = []
    for P in pressures:
        with contextlib.redirect_stdout(io.StringIO()):
            st, liq, vap, K, v, info = flash(flow, temperatures, P * np.ones_like(temperatures))
            res = calculate_hydrodynamic(liq, vap, pl)
        hp = res[5]
        for j, T in enumerate(temperatures):
            rows.append(
                dict(
                    P=P,
                    T=T,
                    beta=float(v[j]),
                    mu_L=float(hp[-2][j]),
                    mu_G=float(hp[-1][j]),
                    sigma=float(hp[-3][j]),
                    rho_L=float(liq.properties.density[j]),
                    rho_V=float(vap.properties.density[j]),
                    h_L=float(liq.properties.enthalpy[j]),
                    h_V=float(vap.properties.enthalpy[j]),
                )
            )
    grid = pd.DataFrame(rows)
    grid.to_csv(B.out("property_grid.csv"), index=False)
    pt = pd.read_csv("pt_values.csv", index_col=0)
    pt.columns = ["P", "T"]
    key = {(round(r["P"], 6), round(r["T"], 6)): i for i, r in grid.iterrows()}
    s = grid.loc[[key[(round(p, 6), round(t, 6))] for p, t in zip(pt["P"], pt["T"])]]
    s.to_csv(B.out("property_159.csv"), index=False)
    two = ((s.beta > 0) & (s.beta < 1)).sum()
    print(f"159 estados: {two} bifásicos no código hoje; μ_L > 1,8e-5 Pa s em {(s.mu_L > 1.8e-5).sum()}")
    for c, f, u in [
        ("mu_L", 1e3, "mPa s"),
        ("mu_G", 1e6, "µPa s"),
        ("sigma", 1e3, "mN/m"),
        ("rho_L", 1, "kg/m3"),
        ("rho_V", 1, "kg/m3"),
    ]:
        q = np.quantile(s[c], [0, 0.1, 0.5, 0.9, 1]) * f
        print(f"  {c:6s} [{u}] min {q[0]:.3g}  q10 {q[1]:.3g}  mediana {q[2]:.3g}  q90 {q[3]:.3g}  max {q[4]:.3g}")


if __name__ == "__main__":
    main()
