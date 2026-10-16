"""Figuras e tabelas do artigo 01 (volume I: o simulador), a partir das corridas gravadas em `runs/`, das
figuras digitalizadas e dos dados da dissertação (não simula). As figuras e tabelas dos substitutos estão no
artigo 02 (`papers/02-surrogate-models/figures/make_figures.py`).

uso: uv run python figures/make_figures.py --out figures/generated --tables tables/generated [--only NOME] [--no-png]
     (a partir da pasta do artigo; `make figures` faz o mesmo)

Figuras (PDF vetorial + PNG em `--out`):
  offshore_profiles   §5.5: perfis de P, T e holdup no sistema poço–linha–riser, os dois modelos contra as
                      curvas digitalizadas (Figuras 32–34)
  dp_parity           Apêndice A: paridade da perda de carga nos 200 casos (horizontal e vertical)
  condensate_profiles §5.6: temperatura e fração vaporizada molar nos casos 1–4 e 6 contra as Figuras 39–40

Tabelas (fragmentos LaTeX `tabular` em `--tables`, lidos por \\input no main.tex; a nota em `<nome>_note.tex`):
  tab_benchmarks, tab_solver_matrix, tab_dp_stats, tab_offshore_points, tab_condensate, tab_performance

Corridas necessárias: `batches/dissertation.json`, `batches/dp_comparison.json`, `batches/solver_matrix.json`,
`batches/performance.json` e, para o custo por amostra da marcha vetorizada, `surrogate.pipeline --steps datasets`.
Sem corridas, a figura sai com um aviso e a tabela com «--», para o PDF compilar na mesma.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_common"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figlib as fl  # noqa: E402

SR = fl.RUNS / "surrogate"
MODELS = (
    ("noslip", "Modelo no-slip", "no-slip", fl.COLORS["noslip"]),
    ("slip", "Modelo", "mechanistic", fl.COLORS["mechanistic"]),
)


def _exists(path):
    return Path(path).exists()


# ----------------------------------------------------------------------------------------------- figuras


def offshore_profiles(out, png):
    runs = {}
    for key, *_ in MODELS:
        path = fl.RUNS / "dissertation" / f"offshore_5_5_{key}.npz"
        if path.exists():
            runs[key] = fl.load_run(path)
    fig, axes = plt.subplots(3, 1, figsize=(fl.MID, 0.95 * fl.MID), sharex=True)
    if not runs:
        for ax in axes:
            fl.placeholder(ax, "corrida em falta: uv run cmfsim batch batches/dissertation.json")
        fl.save(fig, out, "offshore_profiles", png)
        return
    digitized = {n: fl.digitized(n) for n in (32, 33, 34)}
    for key, series, label, color in MODELS:
        if key not in runs:
            continue
        r = runs[key]
        axes[0].plot(r.x, r.P / 1e5, color=color, label=f"this work, {label}")
        axes[1].plot(r.x, r.T - 273.15, color=color)
        axes[2].plot(r.x, r["H_L"], color=color)
        for ax, n in zip(axes, (32, 33, 34)):
            if series in digitized[n]:
                x, y = digitized[n][series]
                ax.plot(x, y, ":", color=color, lw=1.4, label=f"Carvalho (2022), {label}" if n == 32 else None)
    for ax, ylabel, letter in zip(axes, ("$P$ (bar)", "$T$ ($^\\circ$C)", "$H_L$ (--)"), "abc"):
        ax.set_ylabel(ylabel)
        ax.grid(True)
        fl.panel_label(ax, f"({letter})")
    axes[2].set_xlabel("$x$ (m)")
    axes[0].legend(ncol=2, loc="upper right")
    fig.align_ylabels(axes)
    fl.save(fig, out, "offshore_profiles", png)


def _dp_runs(orientation, model):
    runs = fl.load_runs(f"dp_comparison/dp_{orientation}_{model}_*.npz")
    if not runs:
        return None, None
    idx = np.array([int(name.rsplit("_", 1)[1]) for name in runs])
    dp = np.array([(r.P[0] - r.P[-1]) / 1e5 for r in runs.values()])
    return idx, dp


def dp_parity(out, png):
    fig, axes = plt.subplots(1, 2, figsize=(fl.DOUBLE, 0.46 * fl.DOUBLE))
    found = False
    for ax, orientation, letter in zip(axes, ("horizontal", "vertical"), "ab"):
        rows = fl.appendix_a(orientation)
        ref = np.array([float(r["dp_bar_dissertation"]) for r in rows])
        for model, marker, color, label in (
            ("mechanistic", "o", fl.COLORS["mechanistic"], "mechanistic"),
            ("noslip", "s", fl.COLORS["noslip"], "no-slip"),
        ):
            idx, dp = _dp_runs(orientation, model)
            if idx is None:
                continue
            found = True
            ax.loglog(ref[idx - 1], dp, marker, color=color, mfc="none", mew=0.7, label=f"this work, {label}")
        lo, hi = np.nanmin(ref) * 0.5, np.nanmax(ref) * 2
        ax.plot([lo, hi], [lo, hi], "-", color="0.2", lw=0.7)
        ax.plot([lo, hi], [lo * 1.25, hi * 1.25], ":", color="0.4", lw=0.6)
        ax.plot([lo, hi], [lo * 0.8, hi * 0.8], ":", color="0.4", lw=0.6)
        ax.set_xlabel(r"$\Delta P$, Carvalho (2022) (bar)")
        ax.set_ylabel(r"$\Delta P$, this work (bar)")
        ax.set_title(orientation)
        ax.grid(True, which="both")
        fl.panel_label(ax, f"({letter})")
        if not found:
            fl.placeholder(ax, "corridas em falta: uv run cmfsim batch batches/dp_comparison.json")
    if found:
        axes[0].legend(loc="upper left")
    fl.save(fig, out, "dp_parity", png)


def _beta_molar(r):
    MW = np.array([c["MW"] for c in r.case_dict["fluid"]["components"]])
    w = r["w_G"]
    nG = w / (r["y_comp"] * MW).sum(1)
    nL = (1.0 - w) / (r["x_comp"] * MW).sum(1)
    return nG / (nG + nL)


def condensate_profiles(out, png):
    runs = {
        k: fl.load_run(fl.RUNS / "dissertation" / f"condensate_5_6_case{k}.npz")
        for k in (1, 2, 3, 4, 6)
        if _exists(fl.RUNS / "dissertation" / f"condensate_5_6_case{k}.npz")
    }
    fig, axes = plt.subplots(2, 1, figsize=(fl.MID, 0.9 * fl.MID), sharex=True)
    if not runs:
        for ax in axes:
            fl.placeholder(ax, "corridas em falta: uv run cmfsim batch batches/dissertation.json")
        fl.save(fig, out, "condensate_profiles", png)
        return
    d39, d40 = fl.digitized(39), fl.digitized(40)
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    for k, r in runs.items():
        c = palette[k - 1]
        U = r.case_dict["meta"].get("U_table25", r.case_dict["pipe"]["profile"][0]["U"])
        T_in = r.case_dict["boundary"]["inlet"]["T"] - 273.15
        axes[0].plot(
            r.x,
            r.T - 273.15,
            color=c,
            label=f"case {k}: $T_\\mathrm{{in}}$ = {T_in:.0f} $^\\circ$C, $U$ = {U:.0f} W m$^{{-2}}$ K$^{{-1}}$",
        )
        axes[1].plot(r.x, _beta_molar(r), color=c)
        if str(k) in d39:
            axes[0].plot(*d39[str(k)], ":", color=c, lw=1.4)
        if str(k) in d40:
            axes[1].plot(*d40[str(k)], ":", color=c, lw=1.4)
    axes[0].plot([], [], "-", color="0.3", label="this work")
    axes[0].plot([], [], ":", color="0.3", lw=1.4, label="Carvalho (2022)")
    axes[0].set_ylabel("$T$ ($^\\circ$C)")
    axes[1].set_ylabel(r"molar vapour fraction $\beta$ (--)")
    axes[1].set_xlabel("$x$ (m)")
    axes[0].legend(ncol=2, fontsize=6)
    for ax, letter in zip(axes, "ab"):
        ax.grid(True)
        fl.panel_label(ax, f"({letter})")
    fl.save(fig, out, "condensate_profiles", png)


# ----------------------------------------------------------------------------------------------- tabelas


def tab_benchmarks(out):
    ct = json.load(open(fl.DATA / "cases_tables.json"))
    g, p21, p24 = ct["table16_dp_grid"], ct["table21_offshore_pipes"], ct["table24_gas_pipeline"]
    rows = [
        [
            "Pressure-drop grid (200 lines)",
            "$L$ = 100 m, horizontal and vertical",
            f"$D$ = {', '.join(str(d) for d in g['D_in'])} in; $P_\\mathrm{{in}}$ = {g['P_in_bar'][0]}--{g['P_in_bar'][-1]} bar; $\\dot m$ = {g['m_kg_s'][0]}--{g['m_kg_s'][-1]} kg/s",
            "CH$_4$/C$_{7+}$ 70/30 mol",
            "adiabatic",
        ],
        [
            "Offshore system",
            f"well {p21['well']['L']:.0f} m vertical, flowline {p21['flowline']['L']:.0f} m horizontal, riser {p21['riser']['L']:.0f} m vertical",
            f"$D$ = {p21['well']['D']*1000:.0f}/{p21['flowline']['D']*1000:.1f} mm; 170 bar, 80 $^\\circ$C, 10 kg/s",
            "9 components with C$_{7+}$",
            f"$U$ = {p21['well']['U']:.0f}/{p21['flowline']['U']:.0f} W m$^{{-2}}$ K$^{{-1}}$",
        ],
        [
            "Gas-condensate pipeline",
            f"$L$ = {p24['L']/1000:.0f} km horizontal, $D$ = {p24['D_in']} in",
            f"{p24['P_in_bar']:.0f} bar, {p24['m_kg_s']:.0f} kg/s; $T_\\mathrm{{in}}$ = 60, 80 $^\\circ$C",
            "C$_1$--nC$_5$ rich gas",
            "$U$ = 1, 3, 6 W m$^{-2}$ K$^{-1}$ (printed), $T_\\mathrm{amb}$ = 25 $^\\circ$C",
        ],
    ]
    return fl.tex_table(
        out,
        "tab_benchmarks",
        ["Problem", "Geometry", "Operating range", "Fluid", "Thermal"],
        rows,
        align="p{2.5cm}p{3.4cm}p{3.8cm}p{2.1cm}p{3.1cm}",
    )


def tab_solver_matrix(out):
    rows = []
    for prefix, pts, label in (("offshore", (1500.0, 7500.0, 7800.0), "offshore"), ("condensate3", (8000.0,), "condensate 3")):
        for model, mlab in (("noslip", "no-slip"), ("slip", "mechanistic")):
            for solver, slab in (("march", "march"), ("fd", "FD"), ("fv", "FV")):
                for backend in ("", "_numpy"):
                    path = fl.RUNS / "solver_matrix" / f"{prefix}_{model}_{solver}{backend}.npz"
                    if not path.exists():
                        continue
                    r = fl.load_run(path)
                    i, b = r.info, r.info.get("balances", {})
                    vals = " / ".join(f"{np.interp(x, r.x, r.P)/1e5:.2f}" for x in pts)
                    tvals = " / ".join(f"{np.interp(x, r.x, r.T)-273.15:.1f}" for x in pts)
                    it = f"{i['iterations']} it." if i.get("iterations") is not None else f"{i.get('n_accepted', '--')} st."
                    rows.append(
                        [
                            label,
                            mlab,
                            slab,
                            "NumPy" if backend else "JAX",
                            vals,
                            tvals,
                            it,
                            fl.fmt(b.get("momentum_Pa"), "{:+.0e}"),
                            fl.fmt(b.get("energy_J_kg"), "{:+.0e}"),
                            fl.fmt(i.get("cpu"), "{:.0f}"),
                        ]
                    )
    if not rows:
        rows = [["--"] * 10]
    return fl.tex_table(
        out,
        "tab_solver_matrix",
        ["Problem", "Model", "Solver", "Backend", "$P$ (bar)", "$T$ ($^\\circ$C)", "Newton / steps", "Mom. (Pa)", "Energy (J/kg)", "Time (s)"],
        rows,
        align="llllllrrrr",
        notes="Solvers: adaptive march (RK45; number of accepted steps), finite differences (FD) and finite volumes (FV) with Newton iterations. $P$ and $T$ at 1500 / 7500 / 7800 m (offshore) or at the outlet (condensate case 3). Mom. and Energy: global balance residuals $(\\Phi_\\mathrm{out}-\\Phi_\\mathrm{in}) - \\int S\\,\\mathrm{d}x$ on the solver's own grid. Wall times include just-in-time compilation on the first run of each structure.",
    )


def tab_dp_stats(out):
    rows = []
    for orientation in ("horizontal", "vertical"):
        ref = np.array([float(r["dp_bar_dissertation"]) for r in fl.appendix_a(orientation)])
        for model, mlab in (("noslip", "no-slip"), ("mechanistic", "mechanistic")):
            idx, dp = _dp_runs(orientation, model)
            if idx is None:
                rows.append([orientation, mlab, "--", "--", "--", "--", "--"])
                continue
            dev = 100 * (dp / ref[idx - 1] - 1)
            rows.append(
                [
                    orientation,
                    mlab,
                    len(dp),
                    f"{np.median(dev):+.1f}",
                    f"{np.percentile(dev, 10):+.1f} / {np.percentile(dev, 90):+.1f}",
                    f"{100*np.mean(np.abs(dev) < 10):.0f}",
                    f"{100*np.mean(np.abs(dev) < 25):.0f}",
                ]
            )
    return fl.tex_table(
        out,
        "tab_dp_stats",
        ["Orientation", "Model", "$n$", "Median deviation (\\%)", "10th / 90th percentile (\\%)", "Within 10\\% (\\%)", "Within 25\\% (\\%)"],
        rows,
        notes="Deviation = $\\Delta P_\\mathrm{this\\ work}/\\Delta P_\\mathrm{Carvalho}-1$ over the 100 lines of each orientation.",
    )


def tab_offshore_points(out):
    rows = []
    dig = {n: fl.digitized(n) for n in (32, 33, 34)}
    for key, series, label, _ in MODELS:
        path = fl.RUNS / "dissertation" / f"offshore_5_5_{key}.npz"
        if not path.exists():
            continue
        r = fl.load_run(path)
        for x in (1500.0, 4500.0, 7500.0):
            ref = [float(np.interp(x, *dig[n][series])) if series in dig[n] else np.nan for n in (32, 33, 34)]
            rows.append(
                [
                    label,
                    f"{x:.0f}",
                    f"{np.interp(x, r.x, r.P)/1e5:.1f} / {fl.fmt(ref[0], '{:.1f}')}",
                    f"{np.interp(x, r.x, r.T)-273.15:.1f} / {fl.fmt(ref[1], '{:.1f}')}",
                    f"{np.interp(x, r.x, r['H_L']):.3f} / {fl.fmt(ref[2], '{:.3f}')}",
                ]
            )
    if not rows:
        rows = [["--"] * 5]
    return fl.tex_table(
        out,
        "tab_offshore_points",
        ["Model", "$x$ (m)", "$P$ (bar): this work / ref.", "$T$ ($^\\circ$C): this work / ref.", "$H_L$: this work / ref."],
        rows,
        align="llrrr",
        notes="Reference values read from the digitised Figs. 32--34 of Carvalho (2022) for the same model.",
    )


def tab_condensate(out):
    d39, d40 = fl.digitized(39), fl.digitized(40)
    rows = []
    for k in range(1, 7):
        path = fl.RUNS / "dissertation" / f"condensate_5_6_case{k}.npz"
        if not path.exists():
            continue
        r = fl.load_run(path)
        s = str(k)
        xt, yt = d39[s]
        T_ref = float(np.interp(7950.0, xt, yt))
        x40, y40 = d40[s]
        xc_ref = float(x40[y40 < 0.9995][0]) if np.any(y40 < 0.9995) else np.nan
        w = r["w_G"]
        xc = float(r.x[w < 0.9995][0]) if np.any(w < 0.9995) else np.nan
        meta = r.case_dict.get("meta", {})
        rows.append(
            [
                k,
                f"{r.case_dict['boundary']['inlet']['T']-273.15:.0f}",
                f"{meta.get('U_table25', np.nan):.0f} / {r.case_dict['pipe']['profile'][0]['U']:.2f}",
                f"{T_ref:.1f} / {r.T[-1]-273.15:.1f}",
                fl.fmt(xc_ref, "{:.0f}") + " / " + fl.fmt(xc, "{:.0f}"),
                f"{_beta_molar(r)[-1]:.3f}",
            ]
        )
    if not rows:
        rows = [["--"] * 6]
    return fl.tex_table(
        out,
        "tab_condensate",
        ["Case", "$T_\\mathrm{in}$ ($^\\circ$C)", "$U$: printed / eff. (W m$^{-2}$ K$^{-1}$)", "$T_\\mathrm{out}$: ref. / this ($^\\circ$C)", "Onset: ref. / this (m)", "$\\beta_\\mathrm{out}$, this"],
        rows,
        align="lrrrrr",
        notes="Reference values from the digitised Figs. 39--40; case 5 is poorly resolved in the source figure.",
    )


def tab_performance(out):
    ct = json.load(open(fl.DATA / "cases_tables.json"))
    t34, t35 = ct["table34_timings_noslip_s"], ct["table35_timings_slip_s"]
    runs = {Path(f).stem: fl.load_run(f) for f in glob.glob(str(fl.RUNS / "performance" / "*.npz"))}
    cpu = lambda k: fl.fmt(runs[k].info.get("cpu"), "{:.1f}") if k in runs else "--"  # noqa: E731
    rows = [
        ["Marching integrator (no-slip)", f"{t34['march']:.0f}", "--", cpu("perf_march_numpy"), cpu("perf_march_jax")],
        ["Nodal system, 12 nodes", f"{t34['fd_12']:.0f}", f"{t35['fd_12']:.0f}", "--", cpu("perf_fd_jax_n12")],
        ["Nodal system, 30 nodes", f"{t34['fd_30']:.0f}", f"{t35['fd_30']:.0f}", "--", cpu("perf_fd_jax_n30")],
        ["Nodal system, 78 nodes", f"{t34['fd_78']:.0f}", f"{t35['fd_78']:.0f}", "--", cpu("perf_fd_jax_n78")],
    ]
    per_sample = []
    for seg in ("well", "line", "riser"):
        p = SR / f"{seg}.npz"
        if p.exists():
            from surrogate.dataset import load_dataset

            d = load_dataset(p)
            per_sample.append(d["meta"]["cpu"] / len(d["X"]))
    rows.append(["Vectorised march, per sample (batch of 300, incl. compilation)", "--", "--", "--", f"{np.mean(per_sample):.2f}" if per_sample else "--"])
    return fl.tex_table(
        out,
        "tab_performance",
        ["Task", "Ref., no-slip (s)", "Ref., mechanistic (s)", "This work, NumPy (s)", "This work, JAX (s)"],
        rows,
        align="p{5.6cm}rrrr",
        notes="Offshore system; JAX times include just-in-time compilation. Ref. = Carvalho (2022), Intel i5 2.5 GHz. This work: Apple silicon laptop, CPU only. The vectorised march is the one used to generate the training data of the companion paper.",
    )


FIGURES = {
    "offshore_profiles": offshore_profiles,
    "dp_parity": dp_parity,
    "condensate_profiles": condensate_profiles,
}
TABLES = {
    "tab_benchmarks": tab_benchmarks,
    "tab_solver_matrix": tab_solver_matrix,
    "tab_dp_stats": tab_dp_stats,
    "tab_offshore_points": tab_offshore_points,
    "tab_condensate": tab_condensate,
    "tab_performance": tab_performance,
}

if __name__ == "__main__":
    fl.run_cli(FIGURES, __doc__.splitlines()[0], tables=TABLES)
