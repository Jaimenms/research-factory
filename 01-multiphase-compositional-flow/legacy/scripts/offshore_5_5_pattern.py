"""§5.5, modelo no-slip do código original no sistema completo (poço, linha, riser de 250 m, perímetro interno),
com os critérios de padrão como estão no código (comparação «>») e em sigmoide (como no módulo com escorregamento,
`hydrodynamic_slip.py`: decide só com sigmoid(10·(v_SG − v_crit)) ≥ 1 − 1e-6, ou seja v_SG > v_crit + 1,38 m/s).
Compara com as curvas «Modelo no-slip» das Figuras 32 e 33 (legacy/issues L04.6).

Uso: .venv-legacy/bin/python legacy/scripts/offshore_5_5_pattern.py   (~3 min)
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
KH = "stratified_nonstratified_transition = superficial_vapour_velocity[index] > "
KU = "annular_flow = (superficial_vapour_velocity[index]*vapour_stream.properties.density[index]**0.5)/((liquid_stream.properties.density[index] - vapour_stream.properties.density[index])*9.81*surface_tension[index])**(1/4) >= 3.1 "
assert KH in SRC and KU in SRC
KULOG, REG = [], []


def build(sig_kh, sig_ku):
    src = SRC
    if sig_kh:
        src = src.replace(
            KH, "stratified_nonstratified_transition = 1/(1 + np.exp(-10*(superficial_vapour_velocity[index] - "
        )
        # fecha a sigmoide no fim da expressão crítica: a linha termina em ')))' -> acrescenta '))) >= 1 - 1e-6'
        lines = src.split("\n")
        for i, l in enumerate(lines):
            if "stratified_nonstratified_transition = 1/(1 + np.exp" in l:
                lines[i] = l + "))) >= 1 - 1e-6"
        src = "\n".join(lines)
    ku_expr = "(superficial_vapour_velocity[index]*vapour_stream.properties.density[index]**0.5)/((liquid_stream.properties.density[index] - vapour_stream.properties.density[index])*9.81*surface_tension[index])**(1/4)"
    new_ku = f"_ku = {ku_expr}\n                KULOG.append(float(_ku))\n                annular_flow = " + (
        "1/(1 + np.exp(-10*(_ku - 3.1))) >= 1 - 1e-6" if sig_ku else "_ku >= 3.1"
    )
    src = src.replace(KU, new_ku)
    src = src.replace(
        "    return superficial_liquid_velocity, superficial_vapour_velocity, liquid_dissipation",
        "    REG.append(float(flow_regime[0]))\n    return superficial_liquid_velocity, superficial_vapour_velocity, liquid_dissipation",
    )
    ns = dict(hy.__dict__)
    ns["KULOG"] = KULOG
    ns["REG"] = REG
    exec(textwrap.dedent(src), ns)
    simm.calculate_hydrodynamic = ns["calculate_hydrodynamic"]


def segment(comps, mass, P, T, D, L, H, U, Ta, rough):
    from app.edge.pipeline import PipeLine
    from app.edge.segment import Segment
    from app.network.single import Single
    from app.node.sink import Sink
    from app.node.source import Source

    seg = Segment()
    seg.set_geometry(D, 0.0, rough)
    seg.set_profile(L, H)
    seg.set_heat(U, Ta)
    seg.discretize(3)
    pl = PipeLine()
    pl.append(seg)
    pl.construct()
    src = Source()
    src.set_temperature(T)
    src.set_components(comps)
    src.set_flowrate(mass)
    src.set_pressure(P)
    sim = simm.Simulator()
    sim.set_model(Single(src, Sink(), pl))
    with contextlib.redirect_stdout(io.StringIO()):
        res = sim.solve_integrate("LSODA")
    return res.t, res.y[0], res.y[1]


def run(label, sig_kh, sig_ku, riser=250.0):
    build(sig_kh, sig_ku)
    KULOG.clear()
    comps = tuple(Z)
    w = np.array([Z[c] * MW[c] for c in comps])
    mass = w / w.sum() * 10.0
    P, T, x0 = 170.0, 353.0, 0.0
    X, PP, TT, regs = [], [], [], []
    for D, L, H, U, Ta, e in [
        (0.127, 1500, 1500, 7, 313, 2e-4),
        (0.1524, 6000, 0, 5, 285, 6e-4),
        (0.1524, riser, riser, 5, 285, 6e-4),
    ]:
        REG.clear()
        x, p, t = segment(comps, mass, P, T, D, L, H, U, Ta, e)
        regs.append(sorted(set(REG)))
        X += list(x0 + x)
        PP += list(p)
        TT += list(t - 273.15)
        P, T, x0 = float(p[-1]), float(t[-1]), x0 + float(x[-1])
    X, PP, TT = map(np.array, (X, PP, TT))
    o = np.argsort(X, kind="stable")
    f32 = {q["name"]: q for q in json.load(open(B.project("references", "figures", "data", "fig_32.json")))["series"]}[
        "Modelo no-slip"
    ]
    f33 = {q["name"]: q for q in json.load(open(B.project("references", "figures", "data", "fig_33.json")))["series"]}[
        "Modelo no-slip"
    ]
    rmsP = np.sqrt(np.mean((np.interp(f32["x"], X[o], PP[o]) - np.array(f32["y"])) ** 2))
    rmsT = np.sqrt(np.mean((np.interp(f33["x"], X[o], TT[o]) - np.array(f33["y"])) ** 2))
    pts = ", ".join(
        f"{xx} m {np.interp(xx, X[o], PP[o]):.2f}/{np.interp(xx, f32['x'], f32['y']):.2f}"
        for xx in (1500, 4000, 7000, 7500)
    )
    print(
        f"{label:34s} rms P {rmsP:.2f} bar, rms T {rmsT:.2f} °C; P código/tese: {pts}; padrões poço/linha/riser {regs}; Ku no poço {min(KULOG):.2f}–{max(KULOG):.2f}"
    )
    return dict(label=label, x=X[o].tolist(), P=PP[o].tolist(), T=TT[o].tolist())


if __name__ == "__main__":
    out = [
        run("critérios como no código (>)", False, False),
        run("KH em sigmoide", True, False),
        run("KH e Kutateladze em sigmoide", True, True),
    ]
    json.dump(out, open(B.out("offshore_5_5_sigmoid.json"), "w"))
