"""Apêndice B (linha horizontal de 6000 m) com o código original, como em
tests/test_generate_samples_sim.py (RK45, U no perímetro externo), com e sem a espessura (L01.3).

O H2S (0,06 % da massa) é retirado e as frações renormalizadas: não tem polinómio de entalpia do
HYSYS e a entalpia sai NaN (L02.4).

Uso: .venv-legacy/bin/python legacy/scripts/appendix_b_line.py --rows 0-9 --thickness 0.0635
Saída: runs/legacy/appendix_b_line_t{thickness}.json e resumo no terminal.
"""

import argparse
import contextlib
import io
import json

import _bootstrap as B
import numpy as np
import pandas as pd

COMPS = ("N2", "CO2", "C1", "C2", "C3", "nC4", "nC5", "nC6", "nC7", "nC8")
FRAC = np.array([0.003, 0.0059, 0.0904, 0.007, 0.0171, 0.0251, 0.0271, 0.0301, 0.0502, 0.7433])
FRAC = FRAC / FRAC.sum()


def main():
    from app.edge.pipeline import PipeLine
    from app.edge.segment import Segment
    from app.network.single import Single
    from app.node.sink import Sink
    from app.node.source import Source
    from app.simulator.simulator import Simulator

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="0-9")
    ap.add_argument("--thickness", type=float, default=0.0635)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.rows.split("-"))
    df = pd.read_csv(B.project("dissertation", "data", "appendix_b_line.csv")).iloc[lo : hi + 1]
    out = []
    for _, r in df.iterrows():
        seg = Segment()
        seg.set_geometry(0.1524, a.thickness, 6e-4)
        seg.set_profile(6000, 0)
        seg.set_heat(5, 285)
        seg.discretize(3)
        pl = PipeLine()
        pl.append(seg)
        pl.construct()
        src = Source()
        src.set_temperature(float(r.T_in_K))
        src.set_components(COMPS)
        src.set_flowrate(FRAC * float(r.m_kg_s))
        src.set_pressure(float(r.P_in_bar))
        sim = Simulator()
        sim.set_model(Single(src, Sink(), pl))
        row = dict(case=int(r.case), P_thesis=float(r.P_out_bar), T_thesis=float(r.T_out_K))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = sim.solve_integrate("RK45")
            row.update(P_out=float(res.y[0][-1]), T_out=float(res.y[1][-1]))
            print(
                f"linha {row['case']}: T_out − tese {row['T_out'] - row['T_thesis']:+.2f} K, P_out − tese {row['P_out'] - row['P_thesis']:+.2f} bar"
            )
        except Exception as e:  # o original falha em alguns estados (cúbica sem raiz real)
            row.update(error=repr(e)[:200])
            print(f"linha {row['case']}: falhou ({row['error']})")
        out.append(row)
    json.dump(out, open(B.out(f"appendix_b_line_t{a.thickness}.json"), "w"))


if __name__ == "__main__":
    main()
