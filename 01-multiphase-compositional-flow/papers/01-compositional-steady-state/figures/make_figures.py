"""Figuras e tabelas do artigo 01 (volume I: o simulador, comparado com o HYSYS), a partir das corridas gravadas
em `runs/`, dos valores do HYSYS (Apêndice A e figuras digitalizadas: envelope, propriedades nos 159 estados,
perfis do sistema offshore) e das tabelas de dados (não simula escoamento; as propriedades termodinâmicas nos
159 estados (os do código original, `dissertation/data/hysys_states.csv`) e o envelope são calculados pelo flash
e guardados em `runs/paper01/thermo_states_legacy.npz`). As figuras e
tabelas dos substitutos estão no artigo 02.

uso: uv run python figures/make_figures.py --out figures/generated --tables tables/generated [--only NOME] [--no-png]
     (a partir da pasta do artigo; `make figures` faz o mesmo)

Figuras (PDF vetorial + PNG em `--out`):
  envelope_hysys       envelope de fases do fluido de 11 componentes: este trabalho contra o HYSYS
  properties_hysys     distribuições (quantis) de ρ_V, ρ_L, h_V, h_L, β, μ_V, μ_L e σ nos 159 estados contra o HYSYS
  dp_hysys             perda de carga nas 200 linhas contra a faixa das correlações do HYSYS, caso a caso
  offshore_hysys       perfis de P, T e holdup no sistema poço–linha–riser contra o HYSYS (Tulsa, Beggs & Brill)
  condensate_profiles  aplicação: temperatura e fração vaporizada molar no gasoduto com condensado

Tabelas (fragmentos LaTeX `tabular` em `--tables`, lidos por \\input no main.tex; a nota em `<nome>_note.tex`),
e dados suplementares (CSV) com os valores do HYSYS usados:
  tab_benchmarks, tab_thermo_reference, tab_properties_hysys, tab_dp_hysys, tab_dp_horizontal_sensitivity,
  tab_offshore_hysys, tab_solver_matrix,
  tab_condensate, tab_performance, tab_components, tab_fluids, supplementary_csv

Corridas necessárias: `batches/dissertation.json` (offshore e condensado), `batches/dp_comparison.json`,
`batches/dp_roughness.json` e `batches/dp_comparison_darcy.json` (sensibilidade do horizontal),
`batches/solver_matrix.json` e `tools/measure_performance.py` (tempos, em `runs/performance_compiled/timings.json`). Sem corridas, a figura sai com um aviso e a
tabela com «--», para o PDF compilar na mesma.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_common"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figlib as fl  # noqa: E402

CACHE = fl.RUNS / "paper01"
X_LABEL = "$x$ (m)"
T_LABEL = "$T$ ($^\\circ$C)"
OUR_MODELS = (("noslip", "no-slip", fl.COLORS["noslip"]), ("slip", "mechanistic", fl.COLORS["mechanistic"]))
HYSYS_OFFSHORE = (
    ("TULSA", "HYSYS, Tulsa", fl.COLORS["hysys_tulsa"], ":"),
    ("BB", "HYSYS, Beggs & Brill", fl.COLORS["hysys_bb"], "--"),
)
DP_MODELS = {
    "horizontal": (
        ("beggs_brill", "Beggs \\& Brill"),
        ("tulsa", "Tulsa"),
        ("htfs", "HTFS"),
        ("gregory", "Gregory--Aziz--Mandhane"),
    ),
    "vertical": (
        ("beggs_brill", "Beggs \\& Brill"),
        ("tulsa", "Tulsa"),
        ("htfs", "HTFS"),
        ("aziz", "Aziz--Govier--Fogarasi"),
    ),
}
PROPS = (
    (14, "rho_V", "$\\rho_V$ (kg m$^{-3}$)", 1.0),
    (15, "rho_L", "$\\rho_L$ (kg m$^{-3}$)", 1.0),
    (16, "h_V", "$h_V$ (MJ kg$^{-1}$)", 1e-6),
    (17, "h_L", "$h_L$ (MJ kg$^{-1}$)", 1e-6),
    (18, "beta", "$\\beta$ (--)", 1.0),
    (19, "mu_V", "$\\mu_V$ ($\\mu$Pa s)", 1e6),
    (20, "mu_L", "$\\mu_L$ (mPa s)", 1e3),
    (21, "sigma", "$\\sigma$ (mN m$^{-1}$)", 1e3),
)
SYSTEM_SEGMENTS = (("well", 0.0, 1500.0), ("flowline", 1500.0, 7500.0), ("riser", 7500.0, 7800.0))


def _exists(path):
    return Path(path).exists()


def _running_median(y, w=9):
    h = w // 2
    yp = np.pad(y, h, mode="edge")
    return np.array([np.median(yp[i : i + w]) for i in range(len(y))])


def _hysys_offshore(n):
    """Perfis do HYSYS digitalizados (Figuras 32–34). Tulsa é pontilhada: em T e holdup a digitalização é densa e
    os picos isolados saem com uma mediana móvel de 9 pontos; em pressão é esparsa e ruidosa depois dos 4,5 km
    (7 pontos entre 6 e 7,5 km), por isso fica crua e é desenhada como pontos. Beggs & Brill fica como está."""
    dig = fl.digitized(n)
    out = {}
    for series, *_ in HYSYS_OFFSHORE:
        if series in dig:
            x, y = dig[series]
            out[series] = (x, _running_median(y) if (series == "TULSA" and n != 32) else y)
    return out


TULSA_P_RELIABLE_TO = 4500.0  # m: até aqui a pressão de Tulsa digitalizada é densa (>= 30 pontos por 1,5 km)


def _shade_segments(axes):
    for ax in np.atleast_1d(axes):
        ax.axvspan(SYSTEM_SEGMENTS[1][1], SYSTEM_SEGMENTS[1][2], color="0.93", lw=0)
    top = np.atleast_1d(axes)[0]
    for name, x0, x1 in SYSTEM_SEGMENTS:
        top.text(0.5 * (x0 + x1), 1.03, name, transform=top.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)


# ------------------------------------------------------------------------------------ dados termodinâmicos


def thermo_data():
    """Propriedades nos 159 estados da figura do HYSYS, envelope de fases pelo nosso flash e, se a biblioteca
    `thermo` estiver instalada, a mesma flash por uma implementação independente de Peng–Robinson. Calculado
    uma vez e guardado em `runs/paper01/thermo.npz`."""
    path = CACHE / "thermo_states_legacy.npz"  # estados do código original (o antigo thermo.npz usava os da Figura 13)
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    from cmfsim.thermo import FlashOptions, phase_state
    from dissertation.notebook_utils import make_fluid

    cs, z = make_fluid("table6_validation")
    cs = cs.with_exact_pr_constants()
    opts = FlashOptions(stability_test=True)
    f13 = fl.digitized(13)
    # os 159 estados da comparação com o HYSYS tal como o código original os gerou (dissertation/data/
    # hysys_states.csv, de pt_values.csv): a Figura 13 desenha-os a T − 10 K (legacy/issues L03.4)
    from dissertation.notebook_utils import hysys_states

    T, P_bar = hysys_states()
    st = phase_state(T, P_bar * 1e5, z, cs, opts)
    out = {"T": T, "P_bar": P_bar, "two_phase": st.two_phase.astype(float)}
    for _, key, _, _ in PROPS:
        out[key] = np.asarray(getattr(st, key), float)

    def band(T_K, P_grid):
        two = phase_state(np.full(P_grid.size, T_K), P_grid, z, cs, opts).two_phase > 0
        if not two.any():
            return np.nan, np.nan
        idx = np.flatnonzero(two)

        def refine(p_in, p_out):
            lo, hi = p_in, p_out
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                b = phase_state(np.array([T_K]), np.array([mid]), z, cs, opts).two_phase[0] > 0
                lo, hi = (mid, hi) if b else (lo, mid)
            return 0.5 * (lo + hi)

        i0, i1 = idx[0], idx[-1]
        p_dew = refine(P_grid[i0], P_grid[i0 - 1]) if i0 > 0 else np.nan  # orvalho abaixo de 0,5 bar: fora da grelha
        p_bub = refine(P_grid[i1], P_grid[i1 + 1]) if i1 < P_grid.size - 1 else np.nan
        return p_dew / 1e5, p_bub / 1e5

    Tb, _ = f13["Curva azul do envelope (bolha)"]
    Td, _ = f13["Curva laranja do envelope (orvalho)"]
    T_grid = np.linspace(min(Tb.min(), Td.min()) + 2.0, max(Tb.max(), Td.max()) - 2.0, 36)
    P_grid = np.logspace(np.log10(0.5e5), np.log10(300e5), 120)
    env = np.array([band(t, P_grid) for t in T_grid])
    out.update(env_T=T_grid, env_P_dew=env[:, 0], env_P_bub=env[:, 1])
    try:
        sys.path.insert(0, str(fl.PROJECT))
        from tests.test_thermo_reference import _flasher

        flasher = _flasher(cs)
        ref = {"rho_V": [], "rho_L": [], "beta": []}
        for Tk, Pk in zip(T, P_bar * 1e5):
            r = flasher.flash(T=float(Tk), P=float(Pk), zs=list(z))
            gas, liqs = getattr(r, "gas", None), getattr(r, "liquids", None) or []
            ref["beta"].append(np.nan if r.VF is None else float(r.VF))
            ref["rho_V"].append(gas.rho_mass() if gas else np.nan)
            ref["rho_L"].append(liqs[0].rho_mass() if liqs else np.nan)
        for k, v in ref.items():
            out[f"ref_{k}"] = np.array(v, float)
    except Exception as e:  # noqa: BLE001
        print("biblioteca thermo indisponível:", e)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def _merged_quantiles(values, n):
    """Funde os nossos valores como a digitalização funde os marcadores sobrepostos (ver notebook 5.1)."""
    from dissertation.notebook_utils import figure, merge_like_figure

    return merge_like_figure(values, figure(n))


# ----------------------------------------------------------------------------------------------- figuras


def envelope_hysys(out, png):
    fig, ax = plt.subplots(figsize=(fl.MID, 0.62 * fl.MID))
    d13 = fl.digitized(13)
    th = thermo_data()
    Tb, Pb = d13["Curva azul do envelope (bolha)"]
    Td, Pd = d13["Curva laranja do envelope (orvalho)"]
    ax.plot(Tb, Pb, ":", color=fl.COLORS["hysys_tulsa"], lw=1.6, label="bubble line, HYSYS")
    ax.plot(Td, Pd, ":", color=fl.COLORS["hysys_bb"], lw=1.6, label="dew line, HYSYS")
    ax.plot(th["env_T"], th["env_P_bub"], "-", color=fl.COLORS["noslip"], lw=1.0, label="bubble line, this work")
    ax.plot(th["env_T"], th["env_P_dew"], "-", color=fl.COLORS["mechanistic"], lw=1.0, label="dew line, this work")
    ax.plot(th["T"], th["P_bar"], "k.", ms=2.5, label="159 states compared")
    ax.set_xlabel("$T$ (K)")
    ax.set_ylabel("$P$ (bar)")
    ax.grid(True)
    ax.legend(fontsize=6.5)
    fl.save(fig, out, "envelope_hysys", png)


def properties_hysys(out, png):
    th = thermo_data()
    fig, axes = plt.subplots(2, 4, figsize=(fl.DOUBLE, 0.52 * fl.DOUBLE))
    qs = np.linspace(0.0, 1.0, 41)
    for ax, (n, key, lab, scale), letter in zip(axes.ravel(), PROPS, "abcdefgh"):
        dig = fl.digitized(n)
        _, y_hysys = next(iter(dig.values()))
        mine = th[key][th["two_phase"] > 0]
        merged = _merged_quantiles(mine, n)
        qh = np.quantile(np.sort(y_hysys), qs) * scale
        ax.plot(qh, np.quantile(merged, qs) * scale, "o-", ms=2.2, lw=0.8, color=fl.COLORS["noslip"])
        lo, hi = min(qh.min(), np.min(merged) * scale), max(qh.max(), np.max(merged) * scale)
        ax.plot([lo, hi], [lo, hi], "-", color="0.3", lw=0.6)
        ax.set_title(lab, fontsize=7.5)
        ax.set_xlabel("HYSYS quantile", fontsize=7)
        ax.set_ylabel("this work quantile", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True)
        fl.panel_label(ax, f"({letter})")
    fig.tight_layout()
    fl.save(fig, out, "properties_hysys", png)


def _dp_data(orientation):
    rows = fl.appendix_a(orientation)
    H = np.array([[float(r[f"dp_bar_{m}"]) for m, _ in DP_MODELS[orientation]] for r in rows])
    ours = {}
    for model in ("mechanistic", "noslip"):
        runs = fl.load_runs(f"dp_comparison/dp_{orientation}_{model}_*.npz")
        if runs:
            idx = np.array([int(name.rsplit("_", 1)[1]) for name in runs]) - 1
            dp = np.full(len(rows), np.nan)
            dp[idx] = [(r.P[0] - r.P[-1]) / 1e5 for r in runs.values()]
            ours[model] = dp
    return rows, H, ours


def dp_hysys(out, png):
    fig, axes = plt.subplots(2, 1, figsize=(fl.DOUBLE, 0.62 * fl.DOUBLE))
    for ax, orientation, letter in zip(axes, ("horizontal", "vertical"), "ab"):
        rows, H, ours = _dp_data(orientation)
        k = np.arange(1, len(rows) + 1)
        ax.fill_between(
            k, H.min(1), H.max(1), step="mid", color="0.82", lw=0, label="range of the four HYSYS correlations"
        )
        ax.plot(k, np.exp(np.log(H).mean(1)), "_", color="0.35", ms=4, label="HYSYS, geometric mean")
        for model, marker, color, label in (
            ("mechanistic", "o", fl.COLORS["mechanistic"], "this work, mechanistic"),
            ("noslip", "s", fl.COLORS["noslip"], "this work, no-slip"),
        ):
            if model in ours:
                ax.plot(k, ours[model], marker, ms=2.6, mfc="none", mew=0.6, color=color, label=label)
        if not ours:
            fl.placeholder(ax, "corridas em falta: uv run cmfsim batch batches/dp_comparison.json")
        ax.set_yscale("log")
        ax.set_xlim(0.5, len(rows) + 0.5)
        for j, D in enumerate((4, 6, 8, 10)):
            ax.axvline(25 * j + 0.5, color="0.5", lw=0.5)
            ax.text(
                25 * j + 13,
                1.02,
                f"$D$ = {D} in",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
        ax.set_ylabel(f"$\\Delta P$ (bar), {orientation}")
        ax.grid(True, which="major")
        fl.panel_label(ax, f"({letter})")
    axes[1].set_xlabel("case (diameter, then inlet pressure 20--160 bar, then mass flow rate 2--10 kg s$^{-1}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fl.save(fig, out, "dp_hysys", png)


def offshore_hysys(out, png):
    runs = {
        key: fl.load_run(fl.RUNS / "dissertation" / f"offshore_5_5_{key}.npz")
        for key, *_ in OUR_MODELS
        if _exists(fl.RUNS / "dissertation" / f"offshore_5_5_{key}.npz")
    }
    fig, axes = plt.subplots(3, 1, figsize=(fl.MID, 1.0 * fl.MID), sharex=True)
    if not runs:
        for ax in axes:
            fl.placeholder(ax, "corrida em falta: uv run cmfsim batch batches/dissertation.json")
        fl.save(fig, out, "offshore_hysys", png)
        return
    dig = {n: _hysys_offshore(n) for n in (32, 33, 34)}
    for key, label, color in OUR_MODELS:
        if key in runs:
            r = runs[key]
            axes[0].plot(r.x, r.P / 1e5, color=color, lw=1.1, label=f"this work, {label}")
            axes[1].plot(r.x, r.T - 273.15, color=color, lw=1.1)
            axes[2].plot(r.x, r["H_L"], color=color, lw=1.1)
    for series, label, color, ls in HYSYS_OFFSHORE:
        for ax, n in zip(axes, (32, 33, 34)):
            if series not in dig[n]:
                continue
            if series == "TULSA" and n == 32:
                ax.plot(*dig[n][series], ".", color=color, ms=2.5, label=f"{label} (digitised points)")
            else:
                ax.plot(*dig[n][series], ls, color=color, lw=1.4, label=label if n == 32 else None)
    for ax, ylabel, letter in zip(axes, ("$P$ (bar)", T_LABEL, "$H_L$ (--)"), "abc"):
        ax.set_ylabel(ylabel)
        ax.grid(True)
        fl.panel_label(ax, f"({letter})")
    axes[2].set_xlabel(X_LABEL)
    _shade_segments(axes)
    axes[0].legend(ncol=1, fontsize=6.5, loc="upper right")
    fig.align_ylabels(axes)
    fl.save(fig, out, "offshore_hysys", png)


def _beta_molar(r):
    MW = np.array([c["MW"] for c in r.case_dict["fluid"]["components"]])
    w = r["w_G"]
    n_g = w / (r["y_comp"] * MW).sum(1)
    n_l = (1.0 - w) / (r["x_comp"] * MW).sum(1)
    return n_g / (n_g + n_l)


def _condensate_runs():
    return {
        k: fl.load_run(fl.RUNS / "dissertation" / f"condensate_5_6_case{k}.npz")
        for k in range(1, 7)
        if _exists(fl.RUNS / "dissertation" / f"condensate_5_6_case{k}.npz")
    }


def condensate_profiles(out, png):
    runs = _condensate_runs()
    fig, axes = plt.subplots(2, 1, figsize=(fl.MID, 0.9 * fl.MID), sharex=True)
    if not runs:
        for ax in axes:
            fl.placeholder(ax, "corridas em falta: uv run cmfsim batch batches/dissertation.json")
        fl.save(fig, out, "condensate_profiles", png)
        return
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    for k, r in runs.items():
        c = palette[k - 1]
        U = r.case_dict["pipe"]["profile"][0]["U"]
        T_in = r.case_dict["boundary"]["inlet"]["T"] - 273.15
        beta = _beta_molar(r)
        ls = "-" if T_in < 70 else "--"
        axes[0].plot(
            r.x,
            r.T - 273.15,
            ls,
            color=c,
            lw=1.1,
            label=f"case {k}: $T_\\mathrm{{in}}$ = {T_in:.0f} $^\\circ$C, $U$ = {U:.1f} W m$^{{-2}}$ K$^{{-1}}$",
        )
        axes[1].plot(r.x, beta, ls, color=c, lw=1.1)
        m = r["w_G"] < 0.9995
        if np.any(m):
            i = int(np.flatnonzero(m)[0])
            axes[0].plot(r.x[i], r.T[i] - 273.15, "o", color=c, ms=4)
            axes[1].plot(r.x[i], beta[i], "o", color=c, ms=4)
    axes[0].set_ylabel(T_LABEL)
    axes[1].set_ylabel(r"molar vapour fraction $\beta$ (--)")
    axes[1].set_xlabel(X_LABEL)
    axes[0].legend(fontsize=6, ncol=2)
    for ax, letter in zip(axes, "ab"):
        ax.grid(True)
        fl.panel_label(ax, f"({letter})")
    fl.save(fig, out, "condensate_profiles", png)


# ----------------------------------------------------------------------------------------------- tabelas


def tab_benchmarks(out):
    ct = json.load(open(fl.DATA / "cases_tables.json"))
    g, p21, p24 = ct["table16_dp_grid"], ct["table21_offshore_pipes"], ct["table24_gas_pipeline"]
    th = thermo_data()
    rows = [
        [
            "Thermodynamic states",
            "11-component fluid (CO$_2$, N$_2$, H$_2$S, C$_1$--nC$_8$)",
            f"159 $(P, T)$ states inside the phase envelope, {th['P_bar'].min():.0f}--{th['P_bar'].max():.0f} bar, {th['T'].min():.0f}--{th['T'].max():.0f} K",
            "HYSYS, Peng--Robinson",
        ],
        [
            "Pressure-drop lines",
            "CH$_4$/C$_{7+}$ 70/30 mol; horizontal and vertical, $L$ = 100 m, adiabatic, 40 $^\\circ$C",
            f"$D$ = {', '.join(str(d) for d in g['D_in'])} in; $P_\\mathrm{{in}}$ = {g['P_in_bar'][0]}--{g['P_in_bar'][-1]} bar; $\\dot m$ = {g['m_kg_s'][0]}--{g['m_kg_s'][-1]} kg/s (200 lines)",
            "HYSYS: Beggs \\& Brill, Tulsa, HTFS, Gregory--Aziz--Mandhane (horizontal) or Aziz--Govier--Fogarasi (vertical)",
        ],
        [
            "Offshore system",
            "9 components with C$_{7+}$",
            f"well {p21['well']['L']:.0f} m vertical, flowline {p21['flowline']['L']:.0f} m horizontal, riser {p21['riser']['L']:.0f} m vertical; 170 bar, 80 $^\\circ$C, 10 kg/s",
            "HYSYS: Tulsa, Beggs \\& Brill",
        ],
        [
            "Gas-condensate pipeline",
            "C$_1$--nC$_5$ rich gas",
            f"$L$ = {p24['L']/1000:.0f} km horizontal, $D$ = {p24['D_in']} in; {p24['P_in_bar']:.0f} bar, {p24['m_kg_s']:.0f} kg/s; $T_\\mathrm{{in}}$ = 60, 80 $^\\circ$C",
            "none (application)",
        ],
    ]
    return fl.tex_table(
        out,
        "tab_benchmarks",
        ["Problem", "Fluid and geometry", "Conditions", "Reference"],
        rows,
        align="p{2.6cm}p{4.2cm}p{5.0cm}p{3.2cm}",
    )


def tab_thermo_reference(out):
    th = thermo_data()
    rows = []
    m = th["two_phase"] > 0
    for key, lab in (("rho_V", "vapour density"), ("rho_L", "liquid density"), ("beta", "molar vapour fraction")):
        if f"ref_{key}" not in th:
            rows.append([lab, "--", "--", "--"])
            continue
        a, b = th[key][m], th[f"ref_{key}"][m]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        r2 = 1.0 - np.sum((a - b) ** 2) / np.sum((b - b.mean()) ** 2)
        rel = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))
        rows.append([lab, int(ok.sum()), f"{r2:.8f}", f"{rel:.1e}"])
    return fl.tex_table(
        out,
        "tab_thermo_reference",
        ["Property", "States", "$R^2$", "Max. relative deviation"],
        rows,
        notes="Same flash computed by the open-source thermo library (independent Peng--Robinson implementation, exact $\\Omega_a$ and $\\Omega_b$, zero binary interaction parameters) at the 159 states of the HYSYS comparison.",
    )


def tab_properties_hysys(out):
    th = thermo_data()
    d13 = fl.digitized(13)
    rows = []
    Tb, Pb = d13["Curva azul do envelope (bolha)"]
    Td, Pd = d13["Curva laranja do envelope (orvalho)"]
    ok = np.isfinite(th["env_P_bub"])
    dev_b = np.abs(th["env_P_bub"][ok] - np.interp(th["env_T"][ok], Tb, Pb))
    ok = np.isfinite(th["env_P_dew"])
    dev_d = np.abs(th["env_P_dew"][ok] - np.interp(th["env_T"][ok], Td, Pd))
    rows.append(
        ["bubble pressure (bar)", "--", f"median $|\\Delta|$ = {np.median(dev_b):.2f}, max {dev_b.max():.1f}", "--"]
    )
    rows.append(
        ["dew pressure (bar)", "--", f"median $|\\Delta|$ = {np.median(dev_d):.2f}, max {dev_d.max():.1f}", "--"]
    )
    for n, key, lab, scale in PROPS:
        _, y_hysys = next(iter(fl.digitized(n).values()))
        mine = th[key][th["two_phase"] > 0]
        merged = _merged_quantiles(mine, n)
        q = lambda v, p: np.quantile(v, p)  # noqa: E731
        rows.append(
            [
                lab,
                f"{len(y_hysys)} / {merged.size}",
                f"{q(merged, 0.5) / q(y_hysys, 0.5):.2f}",
                f"{q(merged, 0.1) / q(y_hysys, 0.1):.2f} / {q(merged, 0.9) / q(y_hysys, 0.9):.2f}",
            ]
        )
    return fl.tex_table(
        out,
        "tab_properties_hysys",
        ["Property", "Points: HYSYS / this work", "Median ratio (this work / HYSYS)", "10th / 90th percentile ratio"],
        rows,
        align="lrrr",
        notes="Envelope: deviation from the HYSYS bubble and dew lines at the same temperature. Properties: the HYSYS values are available only as unpaired markers of parity plots, so distributions are compared; overlapping markers merge in the digitisation and our 159 values are merged in the same way before computing quantiles.",
    )


def tab_dp_hysys(out):
    rows = []
    for orientation in ("horizontal", "vertical"):
        _, H, ours = _dp_data(orientation)
        lo, hi = H.min(1), H.max(1)
        gmean = np.exp(np.log(H).mean(1))
        for model, mlab in (("noslip", "no-slip"), ("mechanistic", "mechanistic")):
            if model not in ours:
                rows.append([orientation, mlab] + ["--"] * 7)
                continue
            dp = ours[model]
            ok = np.isfinite(dp)
            inside = 100 * np.mean((dp[ok] >= lo[ok]) & (dp[ok] <= hi[ok]))
            below = 100 * np.mean(dp[ok] < lo[ok])
            devs = [np.median(100 * (dp[ok] / H[ok, j] - 1)) for j in range(H.shape[1])]
            rows.append(
                [
                    orientation,
                    mlab,
                    f"{inside:.0f}",
                    f"{below:.0f}",
                    f"{np.median(100 * (dp[ok] / gmean[ok] - 1)):+.0f}",
                ]
                + [f"{v:+.0f}" for v in devs]
            )
    names_h = " / ".join(lab for _, lab in DP_MODELS["horizontal"])
    names_v = " / ".join(lab for _, lab in DP_MODELS["vertical"])
    return fl.tex_table(
        out,
        "tab_dp_hysys",
        [
            "Orientation",
            "Model",
            "Inside HYSYS range (\\%)",
            "Below (\\%)",
            "vs. geom. mean (\\%)",
            "vs. B\\&B (\\%)",
            "vs. Tulsa (\\%)",
            "vs. HTFS (\\%)",
            "vs. Gregory / Aziz (\\%)",
        ],
        rows,
        align="llrrrrrrr",
        notes=f"Median deviation $\\Delta P_\\mathrm{{this\\ work}}/\\Delta P_\\mathrm{{HYSYS}} - 1$ over 100 lines per orientation. HYSYS correlations: {names_h} (horizontal); {names_v} (vertical). The spread of the four correlations themselves (max/min) has a median of 3.0 in the horizontal and 1.2 in the vertical lines.",
    )


def _dp_runs(folder, orientation, model, n):
    """ΔP (bar) e código do padrão na entrada das corridas `runs/<folder>/dp_<orientação>_<modelo>_NNN.npz`."""
    dp, pat = np.full(n, np.nan), np.full(n, np.nan)
    runs = fl.load_runs(f"{folder}/dp_{orientation}_{model}_*.npz")
    for name, r in runs.items():
        i = int(name.rsplit("_", 1)[1]) - 1
        dp[i] = (r.P[0] - r.P[-1]) / 1e5
        if "pattern_code" in r.profiles:
            pat[i] = r["pattern_code"][0]
    return dp, pat


def tab_dp_horizontal_sensitivity(out):
    """Linhas horizontais: o que move o ΔP do modelo face à faixa do HYSYS (rugosidade, convenção do fator de
    atrito, padrão de escoamento). Lotes: dp_comparison, dp_roughness, dp_comparison_darcy (generators/dp_comparison.py).
    """
    rows_a, H, _ = _dp_data("horizontal")
    n = len(rows_a)
    lo, hi, gm = H.min(1), H.max(1), np.exp(np.log(H).mean(1))
    base_m, pat = _dp_runs("dp_comparison", "horizontal", "mechanistic", n)
    base_n, _ = _dp_runs("dp_comparison", "horizontal", "noslip", n)
    variants = [
        ("mechanistic", "base: $\\varepsilon$ = 0.045 mm, Fanning wall friction", base_m, base_m, None),
        (
            "mechanistic",
            "$\\varepsilon$ = 0.2 mm",
            _dp_runs("dp_roughness", "horizontal", "mechanistic", n)[0],
            base_m,
            None,
        ),
        (
            "mechanistic",
            "wall friction $\\times 4$ (Darcy factor in Fanning-form shear)",
            _dp_runs("dp_comparison_darcy", "horizontal", "mechanistic", n)[0],
            base_m,
            None,
        ),
        ("mechanistic", "base, lines in stratified flow", base_m, base_m, pat == 0),
        ("mechanistic", "base, lines in intermittent flow", base_m, base_m, pat == 3),
        ("no-slip", "base: $\\varepsilon$ = 0.045 mm", base_n, base_n, None),
        ("no-slip", "$\\varepsilon$ = 0.2 mm", _dp_runs("dp_roughness", "horizontal", "noslip", n)[0], base_n, None),
    ]
    rows = []
    for model, lab, dp, ref, mask in variants:
        ok = np.isfinite(dp) & np.isfinite(ref) & (np.ones(n, bool) if mask is None else mask)
        if not ok.any():
            rows.append([model, lab] + ["--"] * 6)
            continue
        d, r = dp[ok], ref[ok]
        rows.append(
            [
                model,
                lab,
                f"{ok.sum()}",
                f"{100 * np.mean((d >= lo[ok]) & (d <= hi[ok])):.0f}",
                f"{100 * np.mean(d > hi[ok]):.0f}",
                f"{np.median(d / gm[ok]):.2f}",
                f"{np.median(d / H[ok, 1]):.2f}",
                "--" if dp is ref else f"{np.median(d / r):.2f}",
            ]
        )
    return fl.tex_table(
        out,
        "tab_dp_horizontal_sensitivity",
        [
            "Model",
            "Variant",
            "Lines",
            "Inside (\\%)",
            "Above (\\%)",
            "$\\Delta P$ / HYSYS geom. mean",
            "$\\Delta P$ / Tulsa",
            "$\\Delta P$ / base",
        ],
        rows,
        align="llrrrrrr",
        notes="Medians over the 100 horizontal lines (or over the lines of each flow pattern, as classified by the mechanistic model at the inlet). The roughness of the HYSYS lines is not reported; 0.045 mm is close to the HYSYS default for steel pipe and 0.2 mm is the value of the original case definitions. The factor 4 on the wall friction is the ratio of the Darcy to the Fanning friction factor.",
    )


def tab_offshore_hysys(out):
    dig = {n: _hysys_offshore(n) for n in (32, 33, 34)}
    runs = {
        key: fl.load_run(fl.RUNS / "dissertation" / f"offshore_5_5_{key}.npz")
        for key, *_ in OUR_MODELS
        if _exists(fl.RUNS / "dissertation" / f"offshore_5_5_{key}.npz")
    }
    rows = []
    for x, lab in ((1500.0, "wellhead"), (4500.0, "mid-flowline"), (7500.0, "riser base")):
        for qty, n, unit in (("$P$ (bar)", 32, 1), ("$T$ ($^\\circ$C)", 33, 2), ("$H_L$ (--)", 34, 3)):
            vals = []
            for key, *_ in OUR_MODELS:
                if key not in runs:
                    vals.append("--")
                    continue
                r = runs[key]
                v = {
                    1: np.interp(x, r.x, r.P) / 1e5,
                    2: np.interp(x, r.x, r.T) - 273.15,
                    3: np.interp(x, r.x, r["H_L"]),
                }[unit]
                vals.append(f"{v:.3f}" if unit == 3 else f"{v:.1f}")
            for series, *_ in HYSYS_OFFSHORE:
                if series not in dig[n] or (series == "TULSA" and n == 32 and x > TULSA_P_RELIABLE_TO):
                    vals.append("--")
                    continue
                v = np.interp(x, *dig[n][series])
                vals.append(f"{v:.3f}" if unit == 3 else f"{v:.1f}")
            rows.append([f"{lab} ({x:.0f} m)", qty] + vals)
    return fl.tex_table(
        out,
        "tab_offshore_hysys",
        [
            "Position",
            "Quantity",
            "This work, no-slip",
            "This work, mechanistic",
            "HYSYS, Tulsa",
            "HYSYS, Beggs \\& Brill",
        ],
        rows,
        align="llrrrr",
        notes="HYSYS values read from the digitised profiles; digitisation uncertainty about 0.3 bar, 0.2 K and 0.005 in holdup. The dotted Tulsa curve is smoothed by a 9-point running median in temperature and holdup; its pressure is too sparsely digitised beyond 4.5 km to be read at the riser base.",
    )


def tab_solver_matrix(out):
    rows = []
    for prefix, pts, label in (
        ("offshore", (1500.0, 7500.0, 7800.0), "offshore"),
        ("condensate3", (8000.0,), "condensate 3"),
    ):
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
                    it = (
                        f"{i['iterations']} it."
                        if i.get("iterations") is not None
                        else f"{i.get('n_accepted', '--')} st."
                    )
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
                        ]
                    )
    if not rows:
        rows = [["--"] * 9]
    return fl.tex_table(
        out,
        "tab_solver_matrix",
        [
            "Problem",
            "Model",
            "Solver",
            "Backend",
            "$P$ (bar)",
            "$T$ ($^\\circ$C)",
            "Newton / steps",
            "Mom. (Pa)",
            "Energy (J/kg)",
        ],
        rows,
        align="llllllrrr",
        notes="Solvers: adaptive march (RK45; number of accepted steps), finite differences (FD) and finite volumes (FV) with Newton iterations; 81 nodes (offshore) and 64 nodes (condensate case 3). $P$ and $T$ at 1500 / 7500 / 7800 m (offshore) or at the outlet (condensate). Mom. and Energy: global balance residuals $(\\Phi_\\mathrm{out}-\\Phi_\\mathrm{in}) - \\int S\\,\\mathrm{d}x$ on the solver's own grid.",
    )


def tab_condensate(out):
    runs = _condensate_runs()
    rows = []
    for k, r in runs.items():
        w = r["w_G"]
        m = w < 0.9995
        i = int(np.flatnonzero(m)[0]) if np.any(m) else None
        rows.append(
            [
                k,
                f"{r.case_dict['boundary']['inlet']['T']-273.15:.0f}",
                f"{r.case_dict['pipe']['profile'][0]['U']:.1f}",
                f"{r.T[-1]-273.15:.1f}",
                f"{r.P[-1]/1e5:.2f}",
                "--" if i is None else f"{r.x[i]:.0f}",
                "--" if i is None else f"{r.T[i]-273.15:.1f}",
                f"{_beta_molar(r)[-1]:.3f}",
            ]
        )
    if not rows:
        rows = [["--"] * 8]
    return fl.tex_table(
        out,
        "tab_condensate",
        [
            "Case",
            "$T_\\mathrm{in}$ ($^\\circ$C)",
            "$U$ (W m$^{-2}$ K$^{-1}$)",
            "$T_\\mathrm{out}$ ($^\\circ$C)",
            "$P_\\mathrm{out}$ (bar)",
            "Onset of condensation (m)",
            "$T$ at onset ($^\\circ$C)",
            "$\\beta_\\mathrm{out}$",
        ],
        rows,
        align="lrrrrrrr",
        notes="No-slip model, adaptive march; 70 bar and 15 kg/s at the inlet, ambient at 25 $^\\circ$C.",
    )


def tab_performance(out):
    path = fl.RUNS / "performance_compiled" / "timings.json"
    t = json.load(open(path)) if path.exists() else {}
    rows = []
    for model, mlab in (("noslip", "no-slip"), ("slip", "mechanistic")):
        for solver, slab in (("march", "march (RK45)"), ("fd", "finite differences"), ("fv", "finite volumes")):
            j = t.get(f"offshore_{model}_{solver}", {})
            n = t.get(f"offshore_{model}_{solver}_numpy", {})
            it = (
                f"{j['iterations']} it."
                if j.get("iterations") is not None
                else (f"{j['n_accepted']} st." if j.get("n_accepted") else "--")
            )
            rows.append(
                [
                    mlab,
                    slab,
                    it,
                    fl.fmt(j.get("first_wall"), "{:.1f}"),
                    fl.fmt(j.get("second_wall"), "{:.2f}"),
                    fl.fmt(n.get("first_wall"), "{:.1f}"),
                ]
            )
    s1 = t.get("single_well_march")
    if s1:
        rows.append(
            [
                "no-slip",
                "march, well segment, one case",
                "--",
                f"{s1['first_wall']:.1f}",
                f"{s1['compiled_mean_wall']:.2f}",
                "--",
            ]
        )
    v = t.get("vmap_well_300")
    if v:
        rows.append(
            [
                "no-slip",
                "vectorised march, well segment, 300 cases",
                "--",
                f"{v['first_wall']:.1f}",
                f"{v['second_wall']:.1f} ({v['second_wall'] / v['n']:.2f} per case)",
                "--",
            ]
        )
    return fl.tex_table(
        out,
        "tab_performance",
        ["Model", "Solver", "Newton / steps", "JAX, first run (s)", "JAX, compiled (s)", "NumPy (s)"],
        rows,
        align="llrrrr",
        notes="Offshore system (81 nodes for the nodal solvers); wall times on an Apple silicon laptop, CPU only. The first JAX run includes just-in-time compilation, which is paid once per model structure (fluid, options and grid) and reused for any other inlet condition, pipe parameter or flow rate.",
    )


def tab_components(out):
    comps = json.load(open(fl.DATA / "components.json"))["components"]
    names = {
        "C7plus": "C$_{7+}$",
        "nC4": "$n$-C$_4$",
        "nC5": "$n$-C$_5$",
        "nC6": "$n$-C$_6$",
        "nC7": "$n$-C$_7$",
        "nC8": "$n$-C$_8$",
        "CO2": "CO$_2$",
        "N2": "N$_2$",
        "H2S": "H$_2$S",
        "C1": "C$_1$",
        "C2": "C$_2$",
        "C3": "C$_3$",
    }
    rows = []
    for cid, c in comps.items():
        A, B, C, D = c["cp"]
        rows.append(
            [
                names.get(cid, cid),
                f"{c['MW']:.1f}",
                f"{c['Tc']:.1f}",
                f"{c['Pc']/1e5:.2f}",
                f"{c['omega']:.4f}",
                f"{c['Tb']:.1f}",
                f"{c['Hf']/1e6:.2f}",
                f"{A:.3f}",
                f"{B*1e3:.3f}",
                f"{C*1e6:.3f}",
                f"{D*1e-5:.3f}",
            ]
        )
    return fl.tex_table(
        out,
        "tab_components",
        [
            "Component",
            "$M$ (g/mol)",
            "$T_c$ (K)",
            "$P_c$ (bar)",
            "$\\omega$",
            "$T_b$ (K)",
            "$\\Delta h_f$ (kJ/mol)",
            "$A$",
            "$10^3 B$",
            "$10^6 C$",
            "$10^{-5} D$",
        ],
        rows,
        align="lrrrrrrrrrr",
        notes="Ideal-gas heat capacity $c_p^\\mathrm{ig}/R = A + BT + CT^2 + DT^{-2}$ ($T$ in K). The C$_{7+}$ pseudo-component was characterised in HYSYS (molar mass, density 950 kg/m$^3$, $T_b$, $T_c$, $P_c$, $\\omega$ and enthalpy of formation); its ideal-gas heat capacity is extrapolated from the normal alkanes.",
    )


def tab_fluids(out):
    fl_ = json.load(open(fl.DATA / "fluids.json"))["fluids"]
    order = ("CO2", "N2", "H2S", "C1", "C2", "C3", "nC4", "nC5", "nC6", "nC7", "nC8", "C7plus")
    names = {
        "C7plus": "C$_{7+}$",
        "nC4": "$n$-C$_4$",
        "nC5": "$n$-C$_5$",
        "nC6": "$n$-C$_6$",
        "nC7": "$n$-C$_7$",
        "nC8": "$n$-C$_8$",
        "CO2": "CO$_2$",
        "N2": "N$_2$",
        "H2S": "H$_2$S",
        "C1": "C$_1$",
        "C2": "C$_2$",
        "C3": "C$_3$",
    }
    cols = (
        ("table6_validation", "Thermodynamic states"),
        ("table17_dp_comparison", "Pressure-drop lines"),
        ("table19_offshore", "Offshore system"),
        ("table23_rich_gas", "Gas-condensate pipeline"),
    )
    rows = [[names[c]] + [f"{100 * fl_[k][c]:.1f}" if c in fl_[k] else "--" for k, _ in cols] for c in order]
    return fl.tex_table(
        out, "tab_fluids", ["Component"] + [lab for _, lab in cols], rows, align="lrrrr", notes="Mole percent."
    )


def supplementary_csv(out):
    """Valores do HYSYS usados no artigo, como material suplementar: as 200 perdas de carga (com as nossas) e
    os perfis digitalizados do sistema offshore."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "supplementary_pressure_drop.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "orientation",
                "case",
                "D_in",
                "P_in_bar",
                "m_kg_s",
                "dp_bar_hysys_beggs_brill",
                "dp_bar_hysys_tulsa",
                "dp_bar_hysys_htfs",
                "dp_bar_hysys_gregory_or_aziz",
                "dp_bar_this_work_mechanistic",
                "dp_bar_this_work_noslip",
            ]
        )
        for orientation in ("horizontal", "vertical"):
            rows, H, ours = _dp_data(orientation)
            for i, r in enumerate(rows):
                w.writerow(
                    [orientation, r["case"], r["D_in"], r["P_in_bar"], r["m_kg_s"]]
                    + [f"{v:.4g}" for v in H[i]]
                    + [fl.fmt(ours.get(m, np.full(len(rows), np.nan))[i], "{:.4g}") for m in ("mechanistic", "noslip")]
                )
    with open(out / "supplementary_offshore_hysys.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "correlation", "x_m", "value"])
        for n, qty in ((32, "P_bar"), (33, "T_C"), (34, "H_L")):
            dig = fl.digitized(n)
            for series, lab in (("TULSA", "Tulsa"), ("BB", "Beggs_Brill")):
                if series in dig:
                    for x, y in zip(*dig[series]):
                        w.writerow([qty, lab, f"{x:.1f}", f"{y:.4g}"])
    print("suplementar:", out / "supplementary_pressure_drop.csv", out / "supplementary_offshore_hysys.csv")
    return out


FIGURES = {
    "envelope_hysys": envelope_hysys,
    "properties_hysys": properties_hysys,
    "dp_hysys": dp_hysys,
    "offshore_hysys": offshore_hysys,
    "condensate_profiles": condensate_profiles,
}
TABLES = {
    "tab_benchmarks": tab_benchmarks,
    "tab_thermo_reference": tab_thermo_reference,
    "tab_properties_hysys": tab_properties_hysys,
    "tab_dp_hysys": tab_dp_hysys,
    "tab_dp_horizontal_sensitivity": tab_dp_horizontal_sensitivity,
    "tab_offshore_hysys": tab_offshore_hysys,
    "tab_solver_matrix": tab_solver_matrix,
    "tab_condensate": tab_condensate,
    "tab_performance": tab_performance,
    "tab_components": tab_components,
    "tab_fluids": tab_fluids,
    "supplementary_csv": supplementary_csv,
}

if __name__ == "__main__":
    fl.run_cli(FIGURES, __doc__.splitlines()[0], tables=TABLES)
