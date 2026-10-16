"""§5.5 (sistema offshore, não escorregamento) com o código original: fluido da Tabela 19 com
C7+ = C20pseud, 170 bar, 80 °C, 10 kg/s, poço + linha + riser (Tabela 21, riser de 250 m como nos
scripts), três trechos encadeados como em tests/result_base_case_model.py. Compara com as curvas
«Modelo no-slip» das Figuras 32–34 (L01.4, L02.1).

Uso: .venv-legacy/bin/python legacy/scripts/offshore_5_5.py --perimeter inner|outer
(«outer» usa a espessura da Tabela 21, como o código; com ela a integração LSODA foi interrompida aos 22 min sem terminar)
Saída: runs/legacy/offshore_5_5_{perimeter}.json e resumo no terminal.
"""

import argparse
import contextlib
import io
import json

import _bootstrap as B
import numpy as np

Z = {
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
MW = {
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


def segment(comps, mass, P, T, D, t, rough, L, H, U, Tamb):
    from app.edge.pipeline import PipeLine
    from app.edge.segment import Segment
    from app.network.single import Single
    from app.node.sink import Sink
    from app.node.source import Source
    from app.simulator.simulator import Simulator

    seg = Segment()
    seg.set_geometry(D, t, rough)
    seg.set_profile(L, H)
    seg.set_heat(U, Tamb)
    seg.discretize(3)
    pl = PipeLine()
    pl.append(seg)
    pl.construct()
    src = Source()
    src.set_temperature(T)
    src.set_components(comps)
    src.set_flowrate(mass)
    src.set_pressure(P)
    sim = Simulator()
    sim.set_model(Single(src, Sink(), pl))
    with contextlib.redirect_stdout(io.StringIO()):
        sim.solve_integrate("LSODA", plot="force")
    return [np.asarray(v) for v in sim.output[:4]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perimeter", default="inner", choices=["inner", "outer"])
    ap.add_argument("--riser", type=float, default=250.0)
    a = ap.parse_args()
    comps = tuple(Z)
    w = np.array([Z[c] * MW[c] for c in comps])
    mass = w / w.sum() * 10.0
    tw, tl = (0.00635, 0.0635) if a.perimeter == "outer" else (0.0, 0.0)
    segs = [
        (0.127, tw, 2e-4, 1500, 1500, 7, 313),
        (0.1524, tl, 6e-4, 6000, 0, 5, 285),
        (0.1524, tl, 6e-4, a.riser, a.riser, 5, 285),
    ]
    P, T, x0 = 170.0, 353.0, 0.0
    r = dict(perimeter=a.perimeter, x=[], P=[], T_C=[], lambda_L=[])
    for D, t, e, L, H, U, Ta in segs:
        x, Pp, Tc, vf = segment(comps, mass, P, T, D, t, e, L, H, U, Ta)
        r["x"] += list(x0 + x)
        r["P"] += list(Pp)
        r["T_C"] += list(Tc)
        r["lambda_L"] += list(1 - vf)
        P, T, x0 = float(Pp[-1]), float(Tc[-1] + 273), x0 + float(x[-1])
    r = {k: ([float(v) for v in vals] if isinstance(vals, list) else vals) for k, vals in r.items()}
    json.dump(r, open(B.out(f"offshore_5_5_{a.perimeter}.json"), "w"))
    x = np.array(r["x"])
    o = np.argsort(x, kind="stable")
    for n, key, unit in [(32, "P", "bar"), (33, "T_C", "°C"), (34, "lambda_L", "")]:
        s = {
            q["name"]: q for q in json.load(open(B.project("references", "figures", "data", f"fig_{n}.json")))["series"]
        }["Modelo no-slip"]
        sx, sy = np.array(s["x"]), np.array(s["y"])
        yi = np.interp(sx, x[o], np.array(r[key])[o])
        pts = ", ".join(
            f"{xx:.0f} m: {np.interp(xx, x[o], np.array(r[key])[o]):.3f} / {np.interp(xx, sx, sy):.3f}"
            for xx in (200, 1500, 4000, 7000, 7500)
        )
        print(f"Figura {n}: rms {np.sqrt(np.mean((yi - sy) ** 2)):.3f} {unit}; código / figura em {pts}")


if __name__ == "__main__":
    main()
