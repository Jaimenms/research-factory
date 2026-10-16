"""Apêndice A (§5.4) com o código original e variantes aplicadas em memória (issue L04).

Variantes (separadas por vírgulas; combináveis com «_»):
  base     código tal como está
  fanning  fator de Churchill / 4 no modelo com escorregamento (L04.1)
  visc20   Allan & Teja com Nc_eff = 20 para o C20pseud e 1 para o C1 (L03.2)
  smooth   rugosidade 4,5e-5 m em vez de 2e-4 m (L06.1)

Uso (ambiente de requirements-legacy.txt):
  .venv-legacy/bin/python legacy/scripts/appendix_a.py --variants base,fanning,visc20,smooth,fanning_visc20,fanning_visc20_smooth
Saída: runs/legacy/appendix_a_{orientation}.json e um resumo no terminal (contra a dissertação e, se
existir, contra runs/dp_comparison do cmfsim).
"""

import argparse
import contextlib
import io
import json
import multiprocessing as mp

import _bootstrap as B
import numpy as np
import pandas as pd


def allan_nc(composition, temperature, liquid_density, props):
    nc = {"C1": 1.0, "C20pseud": 20.0}
    n = sum(composition[i] * nc.get(c, 0.0) for i, c in enumerate(props.comp))
    A = 145.73 + 99.01 * n + 0.83 * n**2 - 0.125 * n**3
    Bc = 30.48 + 34.01 * n - 1.230 * n**2 + 0.017 * n**3
    C = -3.07 - 1.99 * n
    return np.exp(A * (-1 / Bc + 1 / (temperature + C))) * 1e-3


def setup(variant):
    import app.hydrodynamic.hydrodynamic_slip as hs
    import app.simulator.simulator_slip as ss
    from app.hydrodynamic.friction_factor import churchill_friction
    from app.thermodynamic_v2.equation_of_state.viscosity_correlation import calculate_viscosity_allan

    hs.churchill_friction = churchill_friction
    hs.calculate_viscosity_allan = calculate_viscosity_allan
    if not hasattr(hs, "_orig_calc"):
        hs._orig_calc = hs.calculate_hydrodynamic
    if "fanning" in variant:
        hs.churchill_friction = lambda Re, e, D: churchill_friction(Re, e, D) / 4
    if "visc20" in variant:
        hs.calculate_viscosity_allan = allan_nc
    rec = {}
    orig = hs._orig_calc

    def wrapped(*a, **k):
        res = orig(*a, **k)
        rec["last"] = res
        return res

    ss.calculate_hydrodynamic = wrapped
    return rec


def run(job):
    variant, orient, row = job
    rec = setup(variant)
    from app.edge.pipeline import PipeLine
    from app.edge.segment import Segment
    from app.network.single import Single
    from app.node.sink import Sink
    from app.node.source import Source
    from app.simulator.simulator_slip import Simulator

    vertical = orient == "vertical"
    rough = 4.5e-5 if "smooth" in variant else 2e-4
    D_in, P, m = row["D_in"], row["P_in_bar"], row["m_kg_s"]
    seg = Segment()
    seg.set_geometry(D_in * 0.0254, 0.00635, rough)
    seg.set_profile(100, 100 if vertical else 0)
    seg.set_heat(0, 313)
    seg.discretize(3)
    pl = PipeLine()
    pl.append(seg)
    pl.construct()
    src = Source()
    src.set_temperature(313)
    src.set_components(("C1", "C20pseud"))
    src.set_flowrate(np.array([0.06, 0.94]) * m)
    src.set_pressure(P)
    sim = Simulator()
    sim.set_model(Single(src, Sink(), pl))
    out = dict(
        variant=variant, orient=orient, case=int(row["case"]), D_in=D_in, P=P, m=m, dp_thesis=row["dp_bar_dissertation"]
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            guess = (P - 6, 313 - 4) if vertical else (P - 4, 313 - 1)
            r, ok = sim.solve_single_slip("hybr", guess[0], guess[1], "inlet", activate_plot=False)
        diss, hold, reg, extra = rec["last"]
        out.update(
            dp=float(P - r["pressure"].iloc[-1]),
            ok=bool(ok),
            regime=int(r["flow_regime"].iloc[1]),
            holdup=float(r["holdup"].iloc[1]),
            eq=float(r["equilibrium"].iloc[1]),
            sigma=float(extra[2][1]),
            mu_L=float(extra[3][1]),
            mu_G=float(extra[4][1]),
        )
    except Exception as e:  # registar e seguir: o original falha em alguns estados
        out.update(dp=float("nan"), ok=False, error=repr(e)[:200])
    return out


def summary(res, orient):
    df = pd.DataFrame(res)
    ours = None
    try:
        rows = []
        for c in sorted(df.case.unique()):
            z = np.load(B.project("runs", "dp_comparison", f"dp_{orient}_mechanistic_{c:03d}.npz"))
            rows.append(dict(case=c, dp_ours=(z["P"][0] - z["P"][-1]) / 1e5))
        ours = pd.DataFrame(rows)
    except FileNotFoundError:
        pass
    for v, d in df.groupby("variant", sort=False):
        r = d.dp / d.dp_thesis
        line = f"{v:24s} original/tese mediana {np.nanmedian(r):.3f} [q10 {np.nanquantile(r, .1):.2f}, q90 {np.nanquantile(r, .9):.2f}]"
        if ours is not None:
            ro = d.merge(ours, on="case")
            ro = ro.dp / ro.dp_ours
            line += (
                f"  original/cmfsim {np.nanmedian(ro):.3f} [{np.nanquantile(ro, .1):.2f}, {np.nanquantile(ro, .9):.2f}]"
            )
        line += f"  padrões {dict(d.regime.value_counts())}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base")
    ap.add_argument("--orientation", default="horizontal", choices=["horizontal", "vertical"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    df = pd.read_csv(B.project("dissertation", "data", f"appendix_a_{a.orientation}.csv"))
    jobs = [(v, a.orientation, row) for v in a.variants.split(",") for _, row in df.iterrows()]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(run, jobs, chunksize=1)
    path = B.out(f"appendix_a_{a.orientation}.json")
    json.dump(res, open(path, "w"), indent=0)
    summary(res, a.orientation)
    print("->", path)


if __name__ == "__main__":
    main()
