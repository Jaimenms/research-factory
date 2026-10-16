"""§5.6 (gasoduto com condensação) com o código original, como em tests/validation_condensate_pipeline.py
(D_ext passado como espessura, L01.2), e comparação com as Figuras 39 e 40 digitalizadas.

Uso: .venv-legacy/bin/python legacy/scripts/condensate_5_6.py [--cases 1,2,3,4,6]
Saída: runs/legacy/condensate_5_6.json e resumo no terminal.
"""

import argparse
import contextlib
import io
import json

import _bootstrap as B
import numpy as np

CASES = {1: (1, 333), 2: (3, 333), 3: (6, 333), 4: (1, 353), 5: (3, 353), 6: (6, 353)}


def run(case, thickness=0.365):
    from app.edge.pipeline import PipeLine
    from app.edge.segment import Segment
    from app.network.single import Single
    from app.node.sink import Sink
    from app.node.source import Source
    from app.simulator.simulator import Simulator

    U, T_in = CASES[case]
    seg = Segment()
    seg.set_geometry(304.8e-3, thickness, 4.57e-5)
    seg.set_profile(8000, 0)
    seg.set_heat(U, 298)
    seg.discretize(3)
    pl = PipeLine()
    pl.append(seg)
    pl.construct()
    src = Source()
    src.set_temperature(T_in)
    src.set_components(("C1", "C2", "C3", "nC4", "nC5"))
    src.set_flowrate(np.array([0.5, 0.2, 0.15, 0.1, 0.05]) * 15)
    src.set_pressure(70)
    sim = Simulator()
    sim.set_model(Single(src, Sink(), pl))
    with contextlib.redirect_stdout(io.StringIO()):
        sim.solve_integrate("LSODA", plot="force")
    x, P, T, vf = sim.output[:4]
    return dict(
        case=case,
        U=U,
        T_in=T_in,
        thickness=thickness,
        x=list(map(float, x)),
        P=list(map(float, P)),
        T_C=list(map(float, T)),
        vap_vol_frac=list(map(float, vf)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2,3,4,6")
    ap.add_argument(
        "--thickness", type=float, default=0.365, help="0.365 = como o script original; 0.0301 = D_ext pretendido"
    )
    a = ap.parse_args()
    res = [run(int(c), a.thickness) for c in a.cases.split(",")]
    json.dump(res, open(B.out("condensate_5_6.json"), "w"))
    figs = {
        n: {
            s["name"]: s for s in json.load(open(B.project("references", "figures", "data", f"fig_{n}.json")))["series"]
        }
        for n in (39, 40)
    }
    for r in res:
        x, T, vf = np.array(r["x"]), np.array(r["T_C"]), np.array(r["vap_vol_frac"])
        s = figs[39][str(r["case"])]
        rms = np.sqrt(np.mean((np.interp(s["x"], x, T) - np.array(s["y"])) ** 2))
        j = np.where(vf < 0.9999)[0]
        jump = f"salto em x = {x[j[0]]:.0f} m para {vf[j[0]]:.4f}" if len(j) else "sem condensação"
        print(
            f"caso {r['case']}: T_out {T[-1]:.2f} °C, P_out {r['P'][-1]:.2f} bar, rms T contra a Figura 39 {rms:.2f} °C, {jump}"
        )


if __name__ == "__main__":
    main()
