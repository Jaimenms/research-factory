"""Linha de escoamento da §5.5 isolada (código original, no-slip), a partir do estado da tese na cabeça do poço
(119,97 bar, 67,3 °C): margem relativa do critério de Kelvin–Helmholtz, (v_SG − v_crit)/v_crit, e o perfil com o
padrão estratificado forçado, contra a curva «Modelo no-slip» da Figura 32 (legacy/issues L04.6).

Uso: .venv-legacy/bin/python legacy/scripts/flowline_5_5_margin.py   (~2 min)
"""

import sys, io, contextlib, json, inspect, textwrap
import numpy as np
import _bootstrap as B
import app.hydrodynamic.hydrodynamic as hy
import app.simulator.simulator as simm

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
SRC = inspect.getsource(hy.calculate_hydrodynamic)
KH = "                stratified_nonstratified_transition = superficial_vapour_velocity[index] > "
assert KH in SRC
LOG = []


def build(force):
    src = SRC.replace(
        KH,
        "                _crit = (1 - liquid_height[index]/pipeline.diameter[index])*np.sqrt((liquid_stream.properties.density[index] - vapour_stream.properties.density[index])*9.81*vapour_area[index]*np.cos(np.arcsin(pipeline.dz[index]/pipeline.dL[index]))/(vapour_stream.properties.density[index]*interface_perimeter[index]))\n"
        "                if index == 0: KHLOG.append((float(liquid_stream.P[0]), float(superficial_vapour_velocity[0]), float(_crit), float(liquid_height[0]/pipeline.diameter[0])))\n"
        + KH,
    )
    if force == "stratified":
        src = src.replace("if stratified_nonstratified_transition == True: # The fluid is non-stratified", "if False:")
    ns = dict(hy.__dict__)
    ns["KHLOG"] = LOG
    exec(textwrap.dedent(src), ns)
    simm.calculate_hydrodynamic = ns["calculate_hydrodynamic"]


def run(force, P0=119.97, T0=67.30 + 273.15):
    build(force)
    LOG.clear()
    from app.edge.pipeline import PipeLine
    from app.edge.segment import Segment
    from app.network.single import Single
    from app.node.sink import Sink
    from app.node.source import Source

    comps = tuple(Z)
    w = np.array([Z[c] * MW[c] for c in comps])
    mass = w / w.sum() * 10.0
    seg = Segment()
    seg.set_geometry(0.1524, 0.0, 6e-4)
    seg.set_profile(6000, 0)
    seg.set_heat(5, 285)
    seg.discretize(3)
    pl = PipeLine()
    pl.append(seg)
    pl.construct()
    src = Source()
    src.set_temperature(T0)
    src.set_components(comps)
    src.set_flowrate(mass)
    src.set_pressure(P0)
    sim = simm.Simulator()
    sim.set_model(Single(src, Sink(), pl))
    with contextlib.redirect_stdout(io.StringIO()):
        res = sim.solve_integrate("LSODA")
    x, P, T = res.t, res.y[0], res.y[1] - 273.15
    fig = {q["name"]: q for q in json.load(open(B.project("references", "figures", "data", "fig_32.json")))["series"]}[
        "Modelo no-slip"
    ]
    sx, sy = np.array(fig["x"]), np.array(fig["y"])
    pts = [(xx, np.interp(xx - 1500, x, P), np.interp(xx, sx, sy)) for xx in (1500, 3000, 4500, 6000, 7500)]
    marg = np.array([(v - c) / c for _, v, c, _ in LOG])
    print(
        f"{force or 'código':10s}: ΔP linha {P[0]-P[-1]:.2f} bar (tese {np.interp(1500, sx, sy) - np.interp(7500, sx, sy):.2f}); P código/tese: "
        + ", ".join(f"{xx} m {a:.2f}/{b:.2f}" for xx, a, b in pts)
    )
    print(
        f"            margem KH (v_SG − v_crit)/v_crit: entrada {marg[0]:+.3f}, mín {marg.min():+.3f}, máx {marg.max():+.3f}; h/D {LOG[0][3]:.3f}"
    )
    return dict(force=force, x=list(map(float, x)), P=list(map(float, P)))


if __name__ == "__main__":
    out = [run(None), run("stratified")]
    for P0 in (119.5, 120.5, 121.0):
        print("entrada P0 =", P0)
        run(None, P0=P0)
    json.dump(out, open(B.out("flowline_55_regime.json"), "w"))
