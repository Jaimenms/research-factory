"""Figuras e tabelas do artigo 01, a partir das corridas gravadas em `runs/`, dos modelos treinados em
`models/`, das figuras digitalizadas e dos dados da dissertação (não simula).

uso: uv run python figures/make_figures.py --out figures/generated --tables tables/generated [--only NOME] [--no-png]
     (a partir da pasta do artigo; `make figures` faz o mesmo)

Figuras (PDF vetorial + PNG em `--out`):
  offshore_profiles   §5.5: perfis de P, T e holdup no sistema poço–linha–riser, os dois modelos contra as
                      curvas digitalizadas (Figuras 32–34)
  dp_parity           Apêndice A: paridade da perda de carga nos 200 casos (horizontal e vertical)
  condensate_profiles §5.6: temperatura e fração vaporizada molar nos casos 1–4 e 6 contra as Figuras 39–40
  surrogate_training  §5.7: erro quadrático médio de treino e teste por época nos três trechos
  surrogate_parity    §5.7: paridade de teste de P_out e T_out por trecho
  surrogate_grid      §5.7: R² da grelha de 135 redes, este trabalho contra o Apêndice C
  surrogate_chain     §5.7: substituto encadeado contra o modelo físico (Figuras 57–58)
  surrogate_profile   extensão: substituto de perfil (s = x/L como entrada) contra o modelo físico
  pinn_study          extensão: dados escassos com e sem o resíduo físico (PINN), ao longo do sistema
  surrogate_validation extensão: curvas de aprendizagem (sobreajuste, variância) e estratégias de divisão

Tabelas (fragmentos LaTeX `tabular` em `--tables`, lidos por \\input no main.tex):
  tab_benchmarks, tab_solver_matrix, tab_dp_stats, tab_offshore_points, tab_condensate, tab_performance,
  tab_appendix_b, tab_surrogate_grid, tab_surrogate_test, tab_chain, tab_profile_pinn, tab_validation, tab_residual

Corridas necessárias: `batches/dissertation.json`, `batches/dp_comparison.json`, `batches/solver_matrix.json`,
`batches/performance.json` e `surrogate.pipeline` (todos os passos). Sem corridas, a figura sai com um aviso e a
tabela com «--», para o PDF compilar na mesma.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_common"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figlib as fl  # noqa: E402

MODELS_DIR = fl.PROJECT / "models" / "surrogate"
SR = fl.RUNS / "surrogate"
SEG = (("well", "well"), ("line", "flowline"), ("riser", "riser"))
MODELS = (
    ("noslip", "Modelo no-slip", "no-slip", fl.COLORS["noslip"]),
    ("slip", "Modelo", "mechanistic", fl.COLORS["mechanistic"]),
)
C_PINN = "#CC79A7"
C_SPARSE = "#E69F00"
C_PHYS = fl.COLORS["reference"]


def _exists(path):
    return Path(path).exists()


def _surrogate():
    from surrogate.model import MLP

    return MLP


def _profile_cls():
    from surrogate.profile import ProfileMLP

    return ProfileMLP


def _dataset(path):
    from surrogate.dataset import load_dataset

    return load_dataset(path)


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
    runs = {k: fl.load_run(fl.RUNS / "dissertation" / f"condensate_5_6_case{k}.npz") for k in (1, 2, 3, 4, 6) if _exists(fl.RUNS / "dissertation" / f"condensate_5_6_case{k}.npz")}
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
        axes[0].plot(r.x, r.T - 273.15, color=c, label=f"case {k}: $T_\\mathrm{{in}}$ = {T_in:.0f} $^\\circ$C, $U$ = {U:.0f} W m$^{{-2}}$ K$^{{-1}}$")
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


def surrogate_training(out, png):
    fig, axes = plt.subplots(1, 3, figsize=(fl.DOUBLE, 0.33 * fl.DOUBLE))
    for ax, (s, name), letter in zip(axes, SEG, "abc"):
        path = SR / f"history_{s}.npz"
        if not path.exists():
            fl.placeholder(ax, "uv run python -m surrogate.pipeline")
            continue
        h = np.load(path)["history"]
        ep = np.arange(1, len(h) + 1)
        ax.semilogy(ep, h[:, 0], color=fl.COLORS["noslip"], label="training")
        ax.semilogy(ep, h[:, 1], color=fl.COLORS["mechanistic"], label="test")
        ax.set_xlabel("epoch")
        ax.set_title(name)
        ax.grid(True, which="both")
        fl.panel_label(ax, f"({letter})")
    axes[0].set_ylabel("MSE (standardised outputs)")
    axes[0].legend()
    fl.save(fig, out, "surrogate_training", png)


def surrogate_parity(out, png):
    fig, axes = plt.subplots(2, 3, figsize=(fl.DOUBLE, 0.62 * fl.DOUBLE))
    MLP = _surrogate()
    for j, (s, name) in enumerate(SEG):
        dpath, mpath, hpath = SR / f"{s}.npz", MODELS_DIR / f"{s}.json", SR / f"history_{s}.npz"
        if not (dpath.exists() and mpath.exists() and hpath.exists()):
            for ax in axes[:, j]:
                fl.placeholder(ax, "uv run python -m surrogate.pipeline")
            continue
        d, m, te = _dataset(dpath), MLP.load(mpath), np.load(hpath)["test_idx"]
        Y, Yp = d["Y"][te], m.predict(d["X"][te])
        for i, (ax, lab, unit) in enumerate(zip(axes[:, j], ("$P_\\mathrm{out}$", "$T_\\mathrm{out}$"), ("bar", "K"))):
            ax.plot(Y[:, i], Yp[:, i], "o", ms=2.5, color=fl.COLORS["noslip"], mfc="none", mew=0.6)
            lo, hi = Y[:, i].min(), Y[:, i].max()
            ax.plot([lo, hi], [lo, hi], "-", color="0.2", lw=0.7)
            ax.set_xlabel(f"{lab}, physical model ({unit})")
            ax.set_ylabel(f"{lab}, surrogate ({unit})")
            ax.grid(True)
            if i == 0:
                ax.set_title(name)
            fl.panel_label(ax, f"({'abcdef'[i * 3 + j]})")
    fl.save(fig, out, "surrogate_parity", png)


def surrogate_grid(out, png):
    fig, axes = plt.subplots(2, 3, figsize=(fl.DOUBLE, 0.6 * fl.DOUBLE))
    for j, (s, name) in enumerate(SEG):
        gpath, cpath = SR / f"grid_{s}.csv", fl.DATA / f"appendix_c_{s}.csv"
        if not (gpath.exists() and cpath.exists()):
            for ax in axes[:, j]:
                fl.placeholder(ax, "uv run python -m surrogate.pipeline --steps grid")
            continue
        ours = {r["case"]: r for r in csv.DictReader(open(gpath))}
        theirs = {r["case"]: r for r in csv.DictReader(open(cpath))}
        cases = sorted(set(ours) & set(theirs), key=int)
        for i, (ax, col, lab) in enumerate(zip(axes[:, j], ("r2_P", "r2_T"), ("$R^2$ of $P_\\mathrm{out}$", "$R^2$ of $T_\\mathrm{out}$"))):
            x = np.clip([float(theirs[c][col]) for c in cases], -1, 1)
            y = np.clip([float(ours[c][col]) for c in cases], -1, 1)
            ax.plot(x, y, "o", ms=2.5, color=fl.COLORS["noslip"], mfc="none", mew=0.6)
            ax.plot([-1, 1], [-1, 1], "-", color="0.2", lw=0.7)
            ax.axhline(0.9, color="0.5", lw=0.5, ls=":")
            ax.axvline(0.9, color="0.5", lw=0.5, ls=":")
            ax.set_xlabel(f"{lab}, Carvalho (2022)")
            ax.set_ylabel(f"{lab}, this work")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True)
            if i == 0:
                ax.set_title(name)
            fl.panel_label(ax, f"({'abcdef'[i * 3 + j]})")
    fl.save(fig, out, "surrogate_grid", png)


def surrogate_chain(out, png):
    fig, axes = plt.subplots(2, 1, figsize=(fl.MID, 0.85 * fl.MID), sharex=True)
    ref_path = SR / "offshore_table6_noslip.npz"
    models_ok = all(_exists(MODELS_DIR / f"{s}.json") for s, _ in SEG)
    if not (ref_path.exists() and models_ok):
        for ax in axes:
            fl.placeholder(ax, "uv run python -m surrogate.pipeline")
        fl.save(fig, out, "surrogate_chain", png)
        return
    from surrogate.predict import chain_predict

    MLP = _surrogate()
    ref = fl.load_run(ref_path)
    inlet = ref.case_dict["boundary"]["inlet"]
    P0, T0, m0 = inlet["P"] / 1e5, inlet["T"], inlet["m"]
    chain = chain_predict([MLP.load(MODELS_DIR / f"{s}.json") for s, _ in SEG], P0, T0, m0)[:, 0, :]
    x_pts = [0.0, 1500.0, 7500.0, 7800.0]
    d57, d58 = fl.digitized(57), fl.digitized(58)
    for ax, n, dref, y_ref, y_chain, ylabel in (
        (axes[0], 57, d57, ref.T - 273.15, np.r_[T0, chain[:, 1]] - 273.15, "$T$ ($^\\circ$C)"),
        (axes[1], 58, d58, ref.P / 1e5, np.r_[P0, chain[:, 0]], "$P$ (bar)"),
    ):
        ax.plot(ref.x, y_ref, color=fl.COLORS["noslip"], label="physical model, this work")
        if "Modelo físico" in dref:
            ax.plot(*dref["Modelo físico"], ":", color=C_PHYS, lw=1.4, label="physical model, Carvalho (2022)")
        ax.plot(x_pts, y_chain, "o", ms=5, color=fl.COLORS["mechanistic"], label="chained surrogate, this work")
        if "RNA" in dref:
            ax.plot(*dref["RNA"], "s", ms=5, mfc="none", color=C_PHYS, label="chained surrogate, Carvalho (2022)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
    axes[0].legend(fontsize=6)
    axes[1].set_xlabel("$x$ (m)")
    for ax, letter in zip(axes, "ab"):
        fl.panel_label(ax, f"({letter})")
    fl.save(fig, out, "surrogate_chain", png)


def _profile_assets(s):
    dpath, mpath, hpath = SR / f"{s}_profile.npz", MODELS_DIR / f"{s}_profile.json", SR / f"history_{s}_profile.npz"
    if not (dpath.exists() and mpath.exists() and hpath.exists()):
        return None
    return _dataset(dpath), _profile_cls().load(mpath), np.load(hpath)["test_idx"]


SYSTEM_SEGMENTS = (("well", 0.0, 1500.0), ("line", 1500.0, 7500.0), ("riser", 7500.0, 7800.0))
SEG_LABEL = {"well": "well", "line": "flowline", "riser": "riser"}


def _system_profiles(models, ref, chained):
    """Perfil ao longo do sistema (x, P bar, T K) com os substitutos de perfil por trecho: a entrada de cada
    trecho é a do modelo físico (`chained=False`, mede o ajuste) ou a saída do substituto anterior
    (`chained=True`, mede o uso real)."""
    xs, Ps, Ts = [], [], []
    P_in, T_in, m = ref.case_dict["boundary"]["inlet"]["P"] / 1e5, ref.case_dict["boundary"]["inlet"]["T"], ref.case_dict["boundary"]["inlet"]["m"]
    for seg, x0, x1 in SYSTEM_SEGMENTS:
        if not chained:
            P_in, T_in = np.interp(x0, ref.x, ref.P) / 1e5, np.interp(x0, ref.x, ref.T)
        x = ref.x[(ref.x >= x0) & (ref.x <= x1)]
        prof = models[seg].predict(P_in, T_in, m, (x - x0) / (x1 - x0))
        xs.append(x)
        Ps.append(prof[:, 0])
        Ts.append(prof[:, 1])
        P_in, T_in = prof[-1, 0], prof[-1, 1]
    return np.concatenate(xs), np.concatenate(Ps), np.concatenate(Ts)


def _shade_segments(axes):
    for ax in np.atleast_1d(axes):
        for k, (seg, x0, x1) in enumerate(SYSTEM_SEGMENTS):
            if k % 2 == 1:
                ax.axvspan(x0, x1, color="0.92", lw=0)
    top = np.atleast_1d(axes)[0]
    for seg, x0, x1 in SYSTEM_SEGMENTS:
        top.text(0.5 * (x0 + x1), 1.03, SEG_LABEL[seg], transform=top.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)


def _profile_models(suffix):
    paths = {seg: (MODELS_DIR if suffix in ("_profile", "_pinn") else SR) / f"{seg}{suffix}.json" for seg, *_ in SYSTEM_SEGMENTS}
    if not all(p.exists() for p in paths.values()):
        return None
    return {seg: _profile_cls().load(p) for seg, p in paths.items()}


def surrogate_profile(out, png):
    """Perfil ao longo de todo o sistema: físico contra o substituto de perfil, com painéis de erro."""
    fig, axes = plt.subplots(4, 1, figsize=(fl.MID, 1.25 * fl.MID), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.3, 1, 1]})
    ref_path = SR / "offshore_table6_noslip.npz"
    models = _profile_models("_profile")
    if models is None or not ref_path.exists():
        for ax in axes:
            fl.placeholder(ax, "uv run python -m surrogate.pipeline --steps reference,profiles,profile_models")
        fl.save(fig, out, "surrogate_profile", png)
        return
    ref = fl.load_run(ref_path)
    axes[0].plot(ref.x, ref.P / 1e5, color=C_PHYS, label="physical model")
    axes[1].plot(ref.x, ref.T - 273.15, color=C_PHYS)
    for chained, color, ls, lab in ((False, fl.COLORS["noslip"], "--", "profile surrogate, segment inlets from the physical model"), (True, fl.COLORS["mechanistic"], "-.", "profile surrogate, chained (each inlet from the previous surrogate)")):
        x, Pn, Tn = _system_profiles(models, ref, chained)
        axes[0].plot(x, Pn, ls, color=color, lw=1.0, label=lab)
        axes[1].plot(x, Tn - 273.15, ls, color=color, lw=1.0)
        axes[2].plot(x, Pn - np.interp(x, ref.x, ref.P) / 1e5, ls, color=color, lw=1.0)
        axes[3].plot(x, Tn - np.interp(x, ref.x, ref.T), ls, color=color, lw=1.0)
    for ax in axes[2:]:
        ax.axhline(0.0, color="0.3", lw=0.5)
    axes[0].set_ylabel("$P$ (bar)")
    axes[1].set_ylabel("$T$ ($^\\circ$C)")
    axes[2].set_ylabel(r"$P_\mathrm{NN} - P$ (bar)")
    axes[3].set_ylabel(r"$T_\mathrm{NN} - T$ (K)")
    axes[3].set_xlabel("$x$ (m)")
    _shade_segments(axes)
    axes[0].legend(fontsize=6, loc="upper right")
    for ax, letter in zip(axes, "abcd"):
        ax.grid(True)
        fl.panel_label(ax, f"({letter})")
    fig.align_ylabels(axes)
    fl.save(fig, out, "surrogate_profile", png)


def pinn_study(out, png):
    """Dados escassos ao longo do sistema: erro dos substitutos de perfil (20 perfis sem e com o resíduo,
    210 perfis) contra o modelo físico, cada trecho alimentado pela entrada física."""
    fig, axes = plt.subplots(2, 1, figsize=(fl.MID, 0.8 * fl.MID), sharex=True)
    ref_path = SR / "offshore_table6_noslip.npz"
    variants = (("_profile_n20", "data only, 20 profiles", C_SPARSE, "--"), ("_pinn", "PINN, 20 profiles + residual", C_PINN, "-"), ("_profile", "data only, 210 profiles", fl.COLORS["noslip"], ":"))
    models = {suf: _profile_models(suf) for suf, *_ in variants}
    if not ref_path.exists() or any(m is None for m in models.values()):
        for ax in axes:
            fl.placeholder(ax, "uv run python -m surrogate.pipeline --steps profiles,profile_models,pinn")
        fl.save(fig, out, "pinn_study", png)
        return
    ref = fl.load_run(ref_path)
    for suf, lab, color, ls in variants:
        x, Pn, Tn = _system_profiles(models[suf], ref, chained=False)
        axes[0].plot(x, Pn - np.interp(x, ref.x, ref.P) / 1e5, ls, color=color, lw=1.0, label=lab)
        axes[1].plot(x, Tn - np.interp(x, ref.x, ref.T), ls, color=color, lw=1.0)
    for ax in axes:
        ax.axhline(0.0, color="0.3", lw=0.5)
        ax.grid(True)
    axes[0].set_ylabel(r"$P_\mathrm{NN} - P$ (bar)")
    axes[1].set_ylabel(r"$T_\mathrm{NN} - T$ (K)")
    axes[1].set_xlabel("$x$ (m)")
    _shade_segments(axes)
    axes[0].legend(fontsize=6, loc="lower left")
    for ax, letter in zip(axes, "ab"):
        fl.panel_label(ax, f"({letter})")
    fig.align_ylabels(axes)
    fl.save(fig, out, "pinn_study", png)


def surrogate_validation(out, png):
    """Curvas de aprendizagem (teste e treino contra o número de perfis de treino) do substituto de saída
    e do de perfil nas três estratégias de divisão; o PINN com 20 perfis marcado."""
    vpath = SR / "validation.json"
    study = json.load(open(SR / "pinn_study.json")) if (SR / "pinn_study.json").exists() else {}
    fig, axes = plt.subplots(2, 3, figsize=(fl.DOUBLE, 0.72 * fl.DOUBLE))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.2, wspace=0.32, hspace=0.42)
    if not vpath.exists():
        for ax in axes.ravel():
            fl.placeholder(ax, "uv run python -m surrogate.validation")
        fl.save(fig, out, "surrogate_validation", png)
        return
    val = json.load(open(vpath))
    modes = (("profile", "split by profile", fl.COLORS["noslip"], "o"), ("position", "split by axial position", fl.COLORS["mechanistic"], "s"), ("block", "block split (top 30\\% of $P_\\mathrm{in}$)", "#009E73", "^"))
    floor = 1e-3  # erros de treino nulos (interpolação exata) ficam no chão do eixo logarítmico

    def curve(ax, lc, i, fmt, color, label, train=False):
        n = np.array([c["n_train"] for c in lc], float)
        if train:
            ax.plot(n, np.maximum([c["train_rmse_mean"][i] for c in lc], floor), fmt, color=color, ms=3, lw=0.8, label=label)
            return
        m = np.array([c["test_rmse_mean"][i] for c in lc])
        sd = np.array([c["test_rmse_std"][i] for c in lc])
        ax.errorbar(n, m, yerr=[np.minimum(sd, 0.9 * m), sd], fmt=fmt, color=color, ms=3, lw=0.8, capsize=2, label=label)

    for j, (s, name) in enumerate(SEG):
        for i, (ax, col, unit) in enumerate(zip(axes[:, j], ("P", "T"), ("bar", "K"))):
            lc = val.get(s, {}).get("outlet", {}).get("learning_curve", {}).get("curve", [])
            if lc:
                curve(ax, lc, i, "d-", "0.3", "outlet surrogate, test")
                curve(ax, lc, i, "d:", "0.3", "outlet surrogate, train", train=True)
            for mode, lab, color, marker in modes:
                lc = val.get(s, {}).get("profile", {}).get(mode, {}).get("curve", [])
                if not lc:
                    continue
                curve(ax, lc, i, marker + "-", color, f"profile surrogate, test, {lab}")
                if mode == "profile":
                    curve(ax, lc, i, marker + ":", color, "profile surrogate, train, split by profile", train=True)
            if s in study:
                r = study[s]["models"]["pinn_from_sparse"]["test_profile"]["rmse"][i]
                ax.plot([20], [r], "*", color=C_PINN, ms=9, label="PINN, 20 profiles + residual")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(floor, 20.0)
            ax.set_xlabel("training profiles")
            ax.set_ylabel(f"RMSE {col} ({unit})")
            ax.grid(True, which="both")
            if i == 0:
                ax.set_title(name)
            fl.panel_label(ax, f"({'abcdef'[i * 3 + j]})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=5.5, bbox_to_anchor=(0.5, 0.0))
    fl.save(fig, out, "surrogate_validation", png)


# ----------------------------------------------------------------------------------------------- tabelas


def _tex(out_dir, name, header, rows, align=None, notes=None):
    """Grava `out_dir/name.tex` com um `tabular` booktabs; `rows` são listas de células já formatadas."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    align = align or "l" + "r" * (len(header) - 1)
    lines = ["% gerado por figures/make_figures.py; não editar à mão", f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path = out_dir / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    # a nota vai num ficheiro à parte (`<nome>_note.tex`, vazio se não houver), para ficar fora do \\resizebox
    (out_dir / f"{name}_note.tex").write_text(("\\par\\smallskip\\footnotesize " + notes + "\n") if notes else "")
    print("tabela:", path)
    return path


def _f(v, fmt="{:.2f}"):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def tab_benchmarks(out):
    ct = json.load(open(fl.DATA / "cases_tables.json"))
    g, p21, p24 = ct["table16_dp_grid"], ct["table21_offshore_pipes"], ct["table24_gas_pipeline"]
    rows = [
        ["Pressure-drop grid (200 lines)", "$L$ = 100 m, horizontal and vertical", f"$D$ = {', '.join(str(d) for d in g['D_in'])} in; $P_\\mathrm{{in}}$ = {g['P_in_bar'][0]}--{g['P_in_bar'][-1]} bar; $\\dot m$ = {g['m_kg_s'][0]}--{g['m_kg_s'][-1]} kg/s", "CH$_4$/C$_{7+}$ 70/30 mol", "adiabatic"],
        ["Offshore system", f"well {p21['well']['L']:.0f} m vertical, flowline {p21['flowline']['L']:.0f} m horizontal, riser {p21['riser']['L']:.0f} m vertical", f"$D$ = {p21['well']['D']*1000:.0f}/{p21['flowline']['D']*1000:.1f} mm; 170 bar, 80 $^\\circ$C, 10 kg/s", "9 components with C$_{7+}$", f"$U$ = {p21['well']['U']:.0f}/{p21['flowline']['U']:.0f} W m$^{{-2}}$ K$^{{-1}}$"],
        ["Gas-condensate pipeline", f"$L$ = {p24['L']/1000:.0f} km horizontal, $D$ = {p24['D_in']} in", f"{p24['P_in_bar']:.0f} bar, {p24['m_kg_s']:.0f} kg/s; $T_\\mathrm{{in}}$ = 60, 80 $^\\circ$C", "C$_1$--nC$_5$ rich gas", "$U$ = 1, 3, 6 W m$^{-2}$ K$^{-1}$ (printed), $T_\\mathrm{amb}$ = 25 $^\\circ$C"],
    ]
    return _tex(out, "tab_benchmarks", ["Problem", "Geometry", "Operating range", "Fluid", "Thermal"], rows, align="p{2.5cm}p{3.4cm}p{3.8cm}p{2.1cm}p{3.1cm}")


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
                    rows.append([label, mlab, slab, "NumPy" if backend else "JAX", vals, tvals, it, _f(b.get("momentum_Pa"), "{:+.0e}"), _f(b.get("energy_J_kg"), "{:+.0e}"), _f(i.get("cpu"), "{:.0f}")])
    if not rows:
        rows = [["--"] * 10]
    return _tex(out, "tab_solver_matrix", ["Problem", "Model", "Solver", "Backend", "$P$ (bar)", "$T$ ($^\\circ$C)", "Newton / steps", "Mom. (Pa)", "Energy (J/kg)", "Time (s)"], rows, align="llllllrrrr", notes="Solvers: adaptive march (RK45; number of accepted steps), finite differences (FD) and finite volumes (FV) with Newton iterations. $P$ and $T$ at 1500 / 7500 / 7800 m (offshore) or at the outlet (condensate case 3). Mom. and Energy: global balance residuals $(\\Phi_\\mathrm{out}-\\Phi_\\mathrm{in}) - \\int S\\,\\mathrm{d}x$ on the solver's own grid. Wall times include just-in-time compilation on the first run of each structure.")


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
            rows.append([orientation, mlab, len(dp), f"{np.median(dev):+.1f}", f"{np.percentile(dev, 10):+.1f} / {np.percentile(dev, 90):+.1f}", f"{100*np.mean(np.abs(dev) < 10):.0f}", f"{100*np.mean(np.abs(dev) < 25):.0f}"])
    return _tex(out, "tab_dp_stats", ["Orientation", "Model", "$n$", "Median deviation (\\%)", "10th / 90th percentile (\\%)", "Within 10\\% (\\%)", "Within 25\\% (\\%)"], rows, notes="Deviation = $\\Delta P_\\mathrm{this\\ work}/\\Delta P_\\mathrm{Carvalho}-1$ over the 100 lines of each orientation.")


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
            rows.append([label, f"{x:.0f}", f"{np.interp(x, r.x, r.P)/1e5:.1f} / {_f(ref[0], '{:.1f}')}", f"{np.interp(x, r.x, r.T)-273.15:.1f} / {_f(ref[1], '{:.1f}')}", f"{np.interp(x, r.x, r['H_L']):.3f} / {_f(ref[2], '{:.3f}')}"])
    if not rows:
        rows = [["--"] * 5]
    return _tex(out, "tab_offshore_points", ["Model", "$x$ (m)", "$P$ (bar): this work / ref.", "$T$ ($^\\circ$C): this work / ref.", "$H_L$: this work / ref."], rows, align="llrrr", notes="Reference values read from the digitised Figs. 32--34 of Carvalho (2022) for the same model.")


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
        rows.append([k, f"{r.case_dict['boundary']['inlet']['T']-273.15:.0f}", f"{meta.get('U_table25', np.nan):.0f} / {r.case_dict['pipe']['profile'][0]['U']:.2f}", f"{T_ref:.1f} / {r.T[-1]-273.15:.1f}", _f(xc_ref, "{:.0f}") + " / " + _f(xc, "{:.0f}"), f"{_beta_molar(r)[-1]:.3f}"])
    if not rows:
        rows = [["--"] * 6]
    return _tex(out, "tab_condensate", ["Case", "$T_\\mathrm{in}$ ($^\\circ$C)", "$U$: printed / eff. (W m$^{-2}$ K$^{-1}$)", "$T_\\mathrm{out}$: ref. / this ($^\\circ$C)", "Onset: ref. / this (m)", "$\\beta_\\mathrm{out}$, this"], rows, align="lrrrrr", notes="Reference values from the digitised Figs. 39--40; case 5 is poorly resolved in the source figure.")


def tab_performance(out):
    ct = json.load(open(fl.DATA / "cases_tables.json"))
    t34, t35 = ct["table34_timings_noslip_s"], ct["table35_timings_slip_s"]
    runs = {Path(f).stem: fl.load_run(f) for f in glob.glob(str(fl.RUNS / "performance" / "*.npz"))}
    cpu = lambda k: _f(runs[k].info.get("cpu"), "{:.1f}") if k in runs else "--"  # noqa: E731
    rows = [
        ["Marching integrator (no-slip)", f"{t34['march']:.0f}", "--", cpu("perf_march_numpy"), cpu("perf_march_jax")],
        ["Nodal system, 12 nodes", f"{t34['fd_12']:.0f}", f"{t35['fd_12']:.0f}", "--", cpu("perf_fd_jax_n12")],
        ["Nodal system, 30 nodes", f"{t34['fd_30']:.0f}", f"{t35['fd_30']:.0f}", "--", cpu("perf_fd_jax_n30")],
        ["Nodal system, 78 nodes", f"{t34['fd_78']:.0f}", f"{t35['fd_78']:.0f}", "--", cpu("perf_fd_jax_n78")],
    ]
    # custo por amostra do march_many (massa de dados do substituto) e da rede
    per_sample = []
    for s, _ in SEG:
        p = SR / f"{s}.npz"
        if p.exists():
            d = _dataset(p)
            per_sample.append(d["meta"]["cpu"] / len(d["X"]))
    MLP = _surrogate()
    nn = "--"
    if all(_exists(MODELS_DIR / f"{s}.json") for s, _ in SEG):
        from surrogate.predict import chain_predict

        ms = [MLP.load(MODELS_DIR / f"{s}.json") for s, _ in SEG]
        X = np.random.default_rng(0).uniform([145, 333, 2], [185, 373, 25], (10000, 3))
        t0 = time.perf_counter()
        for _ in range(200):
            chain_predict(ms, X[:1, 0], X[:1, 1], X[:1, 2])
        t_one = (time.perf_counter() - t0) / 200
        t0 = time.perf_counter()
        chain_predict(ms, X[:, 0], X[:, 1], X[:, 2])
        t_batch = (time.perf_counter() - t0) / 1e4
        nn = f"{t_one*1e6:.0f} / {t_batch*1e6:.2f} $\\mu$s"
    rows.append(["Vectorised march, per sample (batch of 300, incl. compilation)", "--", "--", "--", f"{np.mean(per_sample):.2f}" if per_sample else "--"])
    rows.append(["Neural surrogate chain, one call / per case batched", f"{t34['neural_network']:.2f}", "--", "--", nn])
    return _tex(out, "tab_performance", ["Task", "Ref., no-slip (s)", "Ref., mechanistic (s)", "This work, NumPy (s)", "This work, JAX (s)"], rows, align="p{5.6cm}rrrr", notes="Offshore system; JAX times include just-in-time compilation. Ref. = Carvalho (2022), Intel i5 2.5 GHz. This work: Apple silicon laptop, CPU only; the surrogate chain runs in plain NumPy.")


def tab_appendix_b(out):
    from surrogate.train import r2_score

    rows = []
    for s, name in SEG:
        p = SR / f"{s}_appendix_b.npz"
        if not p.exists():
            rows.append([name, "--", "--", "--", "--", "--", "--"])
            continue
        d = _dataset(p)
        B = np.loadtxt(fl.DATA / f"appendix_b_{s}.csv", delimiter=",", skiprows=1)
        ok = d["ok"]
        Yc, Yb = d["Y"][ok], B[ok, 4:6]
        r2 = r2_score(Yb, Yc)
        dev = Yc - Yb
        rows.append([name, int(ok.sum()), f"{r2[0]:.3f} / {r2[1]:.3f}", f"{dev[:, 0].mean():+.2f} / {dev[:, 1].mean():+.2f}", f"{np.abs(dev[:, 0]).max():.2f} / {np.abs(dev[:, 1]).max():.2f}", f"{(B[ok, 1]-B[ok, 4]).mean():.1f} / {(B[ok, 1]-Yc[:, 0]).mean():.1f}", f"{(B[ok, 2]-B[ok, 5]).mean():.1f} / {(B[ok, 2]-Yc[:, 1]).mean():.1f}"])
    return _tex(out, "tab_appendix_b", ["Segment", "$n$", "$R^2$ ($P$ / $T$)", "Mean dev. (bar / K)", "Max. dev. (bar / K)", "$\\Delta P$: ref. / this (bar)", "$\\Delta T$: ref. / this (K)"], rows, align="lrrrrrr", notes="Same inlet samples as Appendix B of Carvalho (2022); deviation = this work $-$ reference.")


def tab_surrogate_grid(out):
    st = json.load(open(fl.DATA / "surrogate_tables.json"))
    rows = []
    for s, name in SEG:
        gpath, cpath = SR / f"grid_{s}.csv", fl.DATA / f"appendix_c_{s}.csv"
        if not (gpath.exists() and cpath.exists()):
            rows.append([name, "--", "--", "--", "--"])
            continue
        ours = list(csv.DictReader(open(gpath)))
        theirs = list(csv.DictReader(open(cpath)))
        n_ours = sum(r["both_above_0_9"] == "True" for r in ours)
        n_theirs = sum(r["both_above_0_9"] == "True" for r in theirs)
        best = max(ours, key=lambda r: float(r["r2_sum"]))
        rows.append([name, f"{n_ours}/135", f"{n_theirs}/135", f"{best['neurons']} neurons, {best['layers']} layer(s), {best['activation']}, $\\eta$ = {best['learning_rate']}", f"{float(best['r2_P']):.4f} / {float(best['r2_T']):.4f}"])
    hp = st["table31_hyperparameters"]
    return _tex(out, "tab_surrogate_grid", ["Segment", "Networks with $R^2 > 0.9$ (this work)", "Idem, Carvalho (2022)", "Best network, this work", "$R^2$ ($P$ / $T$)"], rows, align="lrrlr", notes=f"Grid of 135 networks; final models use the configuration adopted by Carvalho (2022): {hp['layers']} hidden layer, {hp['neurons']} neurons, {'ReLU' if hp['activation'] == 'relu' else hp['activation']}, learning rate {hp['learning_rate']}.")


def tab_surrogate_test(out):
    st = json.load(open(fl.DATA / "surrogate_tables.json"))
    MLP = _surrogate()
    rows = []
    for s, name in SEG:
        p = MODELS_DIR / f"{s}.json"
        if not p.exists():
            rows.append([name, "--", "--", "--", "--", "--"])
            continue
        t = MLP.load(p).meta["test"]
        ref = st["table32_r2_test"][s]
        rows.append([name, f"{ref['P']:.3f} / {ref['T']:.3f}", f"{t['r2'][0]:.4f} / {t['r2'][1]:.4f}", f"{t['rmse'][0]:.3f} / {t['rmse'][1]:.3f}", f"{t['mean_percent_error'][0]:.2f} / {t['mean_percent_error'][1]:.2f}", f"{t['max_abs_error'][0]:.2f} / {t['max_abs_error'][1]:.2f}"])
    return _tex(out, "tab_surrogate_test", ["Segment", "$R^2$ Carvalho (2022) ($P$ / $T$)", "$R^2$ this work ($P$ / $T$)", "RMSE (bar / K)", "Mean error (\\%) ($P$ / $T$)", "Max. error (bar / K)"], rows, align="lrrrrr", notes="90 test samples per segment (30\\% of 300); network of Table 31 of Carvalho (2022).")


def tab_chain(out):
    st = json.load(open(fl.DATA / "surrogate_tables.json"))
    t33 = st["table33_mean_percent_error_chain"]
    ref_path = SR / "offshore_table6_noslip.npz"
    rows = []
    if ref_path.exists() and all(_exists(MODELS_DIR / f"{s}.json") for s, _ in SEG):
        from surrogate.predict import chain_predict

        MLP = _surrogate()
        ref = fl.load_run(ref_path)
        inlet = ref.case_dict["boundary"]["inlet"]
        chain = chain_predict([MLP.load(MODELS_DIR / f"{s}.json") for s, _ in SEG], inlet["P"] / 1e5, inlet["T"], inlet["m"])[:, 0, :]
        for k, (x, lab, key) in enumerate(((1500.0, "wellhead", "1500"), (7500.0, "riser base", "7500"), (7800.0, "topside arrival", "7750"))):
            Pf, Tf = np.interp(x, ref.x, ref.P) / 1e5, np.interp(x, ref.x, ref.T)
            Ps, Ts = chain[k]
            rows.append([lab, f"{x:.0f}", f"{Pf:.2f} / {Ps:.2f}", f"{100*abs(Ps-Pf)/Pf:.2f} / {t33['P'][key]:.1f}", f"{Tf-273.15:.2f} / {Ts-273.15:.2f}", f"{100*abs(Ts-Tf)/Tf:.2f} / {t33['T'][key]:.1f}"])
    else:
        rows = [["--"] * 6]
    return _tex(out, "tab_chain", ["Point", "$x$ (m)", "$P$ phys. / NN (bar)", "$P$ error this / ref. (\\%)", "$T$ phys. / NN ($^\\circ$C)", "$T$ error this / ref. (\\%)"], rows, align="lrrrrr", notes="Physical model (phys.) and chained surrogate (NN); inlet 170 bar, 80 $^\\circ$C, 20 kg/s; ref. = Table 33 of Carvalho (2022); temperature errors on the kelvin scale.")


def tab_profile_pinn(out):
    Profile = _profile_cls()
    study_path = SR / "pinn_study.json"
    study = json.load(open(study_path)) if study_path.exists() else {}
    labels = {"data_full": "data only", "data_sparse": "data only", "pinn_from_sparse": "PINN, warm start", "pinn_from_scratch": "PINN, random init."}
    rows = []
    for s, name in SEG:
        if s in study:
            for key, lab in labels.items():
                r = study[s]["models"][key]
                t = r["test_profile"]
                rows.append([name, lab, r["n_train_profiles"], f"{t['rmse'][0]:.3f} / {t['rmse'][1]:.3f}", f"{t['max_abs_error'][0]:.2f} / {t['max_abs_error'][1]:.2f}", f"{r['residual_rms']:.3f}"])
        elif _exists(MODELS_DIR / f"{s}_profile.json"):
            meta = Profile.load(MODELS_DIR / f"{s}_profile.json").mlp.meta
            t = meta["test_profile"]
            rows.append([name, labels["data_full"], meta.get("n_train_profiles", "--"), f"{t['rmse'][0]:.3f} / {t['rmse'][1]:.3f}", f"{t['max_abs_error'][0]:.2f} / {t['max_abs_error'][1]:.2f}", "--"])
        else:
            rows.append([name, "--", "--", "--", "--", "--"])
    return _tex(out, "tab_profile_pinn", ["Segment", "Model", "Train. profiles", "RMSE (bar / K)", "Max. error (bar / K)", "RMS residual"], rows, align="llrrrr", notes="Errors over the 90 test profiles (51 axial points each). The residual is $(\\mathrm{d}u/\\mathrm{d}x - f)/\\mathrm{scale}$ on the collocation set; RMS $\\approx 1$ means deviations of the order of the typical slope.")


def tab_validation(out):
    vpath = SR / "validation.json"
    val = json.load(open(vpath)) if vpath.exists() else {}
    rows = []
    for s, name in SEG:
        kf = val.get(s, {}).get("outlet", {}).get("kfold", {}).get("summary")
        lc = val.get(s, {}).get("outlet", {}).get("learning_curve", {}).get("curve", [])
        if kf:
            rows.append([name, "outlet, 5-fold CV (210 train)", f"{kf['test_rmse_mean'][0]:.3f} $\\pm$ {kf['test_rmse_std'][0]:.3f} / {kf['test_rmse_mean'][1]:.3f} $\\pm$ {kf['test_rmse_std'][1]:.3f}", f"{kf['gap_mean'][0]:+.3f} / {kf['gap_mean'][1]:+.3f}", f"{kf['bias_mean'][0]:+.3f} / {kf['bias_mean'][1]:+.3f}"])
        for c in lc:
            if c["n_train"] in (20, 210):
                rows.append([name, f"outlet, {c['n_train']} train, 3 splits", f"{c['test_rmse_mean'][0]:.3f} $\\pm$ {c['test_rmse_std'][0]:.3f} / {c['test_rmse_mean'][1]:.3f} $\\pm$ {c['test_rmse_std'][1]:.3f}", f"{c['gap_mean'][0]:+.3f} / {c['gap_mean'][1]:+.3f}", f"{c['bias_mean'][0]:+.3f} / {c['bias_mean'][1]:+.3f}"])
        for mode, lab in (("profile", "by profile"), ("position", "by position"), ("block", "block")):
            for c in val.get(s, {}).get("profile", {}).get(mode, {}).get("curve", []):
                if c["n_train"] in (20, 210):
                    rows.append([name, f"profile, {c['n_train']} train, split {lab}", f"{c['test_rmse_mean'][0]:.3f} $\\pm$ {c['test_rmse_std'][0]:.3f} / {c['test_rmse_mean'][1]:.3f} $\\pm$ {c['test_rmse_std'][1]:.3f}", f"{c['gap_mean'][0]:+.3f} / {c['gap_mean'][1]:+.3f}", f"{c['bias_mean'][0]:+.3f} / {c['bias_mean'][1]:+.3f}"])
    if not rows:
        rows = [["--"] * 5]
    return _tex(out, "tab_validation", ["Segment", "Model, training set", "Test RMSE, mean $\\pm$ std ($P$ bar / $T$ K)", "Test $-$ train RMSE", "Bias ($P$ / $T$)"], rows, align="llrrr", notes="Repeated random splits (seeds 0--2) or 5-fold cross-validation; the test--train gap measures overfitting, the bias is the mean signed error on the test set. Block split: test profiles are those with $P_\\mathrm{in}$ in the top 30\\% of the range (extrapolation).")


def tab_residual(out):
    vpath, spath = SR / "validation.json", SR / "pinn_study.json"
    val = json.load(open(vpath)) if vpath.exists() else {}
    study = json.load(open(spath)) if spath.exists() else {}
    labels = {"data_full": "data only", "data_sparse": "data only", "pinn_from_sparse": "PINN, warm start", "pinn_from_scratch": "PINN, random init."}
    rows = []
    for s, name in SEG:
        held = val.get(s, {}).get("profile", {}).get("heldout_residual", {})
        for key, lab in labels.items():
            if s in study and key in study[s]["models"]:
                r = study[s]["models"][key]
                rows.append([name, lab, r["n_train_profiles"], f"{r['residual_rms']:.3f}", f"{held[key]:.3f}" if key in held else "--"])
    if not rows:
        rows = [["--"] * 5]
    return _tex(out, "tab_residual", ["Segment", "Model", "Train. profiles", "RMS residual, training set", "RMS residual, held-out set"], rows, align="llrrr", notes="Scaled residual of the governing equations, Eq.~(\\ref{eq:residual}), at 60 collocation points: the set used in the PINN loss and a set never used in training.")


FIGURES = {
    "offshore_profiles": offshore_profiles,
    "dp_parity": dp_parity,
    "condensate_profiles": condensate_profiles,
    "surrogate_training": surrogate_training,
    "surrogate_parity": surrogate_parity,
    "surrogate_grid": surrogate_grid,
    "surrogate_chain": surrogate_chain,
    "surrogate_profile": surrogate_profile,
    "pinn_study": pinn_study,
    "surrogate_validation": surrogate_validation,
}
TABLES = {
    "tab_benchmarks": tab_benchmarks,
    "tab_solver_matrix": tab_solver_matrix,
    "tab_dp_stats": tab_dp_stats,
    "tab_offshore_points": tab_offshore_points,
    "tab_condensate": tab_condensate,
    "tab_performance": tab_performance,
    "tab_appendix_b": tab_appendix_b,
    "tab_surrogate_grid": tab_surrogate_grid,
    "tab_surrogate_test": tab_surrogate_test,
    "tab_chain": tab_chain,
    "tab_profile_pinn": tab_profile_pinn,
    "tab_validation": tab_validation,
    "tab_residual": tab_residual,
}

if __name__ == "__main__":
    fl.run_cli(FIGURES, __doc__.splitlines()[0], tables=TABLES)
