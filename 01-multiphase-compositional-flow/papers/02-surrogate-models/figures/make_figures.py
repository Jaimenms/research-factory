"""Figuras e tabelas do artigo 02 (volume II: os substitutos), a partir das massas de dados e dos modelos
treinados por `surrogate.pipeline` e `surrogate.validation` (`runs/surrogate/`, `models/surrogate/`) e dos tempos de
`tools/measure_performance.py` (não simula). Compara só com o modelo físico do volume I (decisão do dono, 2026-10-16).

uso: uv run python figures/make_figures.py --out figures/generated --tables tables/generated [--only NOME] [--no-png]
     (a partir da pasta do artigo; `make figures` faz o mesmo)

Figuras (PDF vetorial + PNG em `--out`):
  surrogate_training   erro quadrático médio de treino e teste por época nos três trechos
  surrogate_parity     paridade de teste de P_out e T_out por trecho
  surrogate_grid       mapa da grelha de 135 redes por trecho (1 − R² no teste)
  surrogate_chain      substituto encadeado contra o modelo físico do volume I
  surrogate_profile    substituto de perfil (s = x/L como entrada) ao longo do sistema, com painéis de erro
  pinn_study           dados escassos com e sem o resíduo físico (PINN), ao longo do sistema
  surrogate_validation curvas de aprendizagem (sobreajuste, variância) e estratégias de divisão

Tabelas (fragmentos LaTeX `tabular` em `--tables`, lidos por \\input no main.tex; a nota em `<nome>_note.tex`):
  tab_ranges, tab_surrogate_grid, tab_surrogate_test, tab_chain, tab_speed, tab_profile_pinn,
  tab_validation, tab_residual

Corridas necessárias: `uv run python -m surrogate.pipeline` (todos os passos) e `uv run python -m surrogate.validation`.
Sem corridas, a figura sai com um aviso e a tabela com «--», para o PDF compilar na mesma.
"""

from __future__ import annotations

import csv
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
SYSTEM_SEGMENTS = (("well", 0.0, 1500.0), ("line", 1500.0, 7500.0), ("riser", 7500.0, 7800.0))
SEG_LABEL = {"well": "well", "line": "flowline", "riser": "riser"}
C_PINN = "#CC79A7"
C_SPARSE = "#E69F00"
C_PHYS = fl.COLORS["reference"]
LABELS = {
    "data_full": "data only",
    "data_sparse": "data only",
    "pinn_from_sparse": "PINN, warm start",
    "pinn_from_scratch": "PINN, random init.",
}


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


def _study():
    p = SR / "pinn_study.json"
    return json.load(open(p)) if p.exists() else {}


def _validation():
    p = SR / "validation.json"
    return json.load(open(p)) if p.exists() else {}


# ----------------------------------------------------------------------------------------------- figuras


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


GRID_ACTS = ("relu", "sigmoid", "tanh")
GRID_LRS = (0.1, 0.01, 0.001)
GRID_NEURONS = (5, 10, 15, 20, 25)
CHOSEN = {"neurons": 20, "layers": 1, "activation": "relu", "learning_rate": 0.01}


def _grid_rows(s):
    p = SR / f"grid_{s}.csv"
    return list(csv.DictReader(open(p))) if p.exists() else None


def surrogate_grid(out, png):
    """Mapa da grelha de 135 redes por trecho: 1 − min(R² de P, R² de T) no teste, em escala logarítmica, por
    ativação × taxa (linhas) e neurónios × camadas (colunas); a configuração escolhida enquadrada."""
    fig, axes = plt.subplots(1, 3, figsize=(fl.DOUBLE, 0.4 * fl.DOUBLE), sharey=True)
    im = None
    for ax, (s, name), letter in zip(axes, SEG, "abc"):
        rows = _grid_rows(s)
        if rows is None:
            fl.placeholder(ax, "uv run python -m surrogate.pipeline --steps grid")
            continue
        M = np.full((9, 15), np.nan)
        for r in rows:
            i = GRID_ACTS.index(r["activation"]) * 3 + GRID_LRS.index(float(r["learning_rate"]))
            j = GRID_NEURONS.index(int(r["neurons"])) * 3 + int(r["layers"]) - 1
            M[i, j] = min(float(r["r2_P"]), float(r["r2_T"]))
        V = np.log10(np.clip(1.0 - M, 1e-5, 1.0))
        im = ax.imshow(V, cmap="viridis_r", vmin=-4, vmax=0, aspect="auto", interpolation="nearest")
        ci = GRID_ACTS.index(CHOSEN["activation"]) * 3 + GRID_LRS.index(CHOSEN["learning_rate"])
        cj = GRID_NEURONS.index(CHOSEN["neurons"]) * 3 + CHOSEN["layers"] - 1
        ax.add_patch(plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False, ec="white", lw=1.2))
        ax.set_xticks([3 * k + 1 for k in range(5)], [str(n) for n in GRID_NEURONS])
        for k in range(1, 5):
            ax.axvline(3 * k - 0.5, color="white", lw=0.5)
        for k in range(1, 3):
            ax.axhline(3 * k - 0.5, color="white", lw=0.5)
        ax.set_xlabel("neurons per layer (1, 2, 3 layers)", fontsize=7)
        ax.set_title(name)
        fl.panel_label(ax, f"({letter})")
    axes[0].set_yticks(
        range(9), [f"{a if a != 'relu' else 'ReLU'}, {lr:g}" for a in GRID_ACTS for lr in GRID_LRS], fontsize=6.5
    )
    axes[0].set_ylabel("activation, learning rate", fontsize=7)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
        cb.set_label(r"$\log_{10}\,[1 - \min(R^2_P, R^2_T)]$", fontsize=7)
    fl.save(fig, out, "surrogate_grid", png)


def surrogate_chain(out, png):
    """Substituto encadeado (três redes) contra o modelo físico do volume I ao longo do sistema, com os erros nas
    fronteiras dos trechos em painéis próprios."""
    fig, axes = plt.subplots(
        4, 1, figsize=(fl.MID, 1.15 * fl.MID), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.3, 1, 1]}
    )
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
    x_pts = np.array([1500.0, 7500.0, 7800.0])
    c = fl.COLORS["mechanistic"]
    axes[0].plot(ref.x, ref.T - 273.15, color=C_PHYS, lw=1.0, label="physical model (Part I)")
    axes[0].plot(
        np.r_[0.0, x_pts], np.r_[T0, chain[:, 1]] - 273.15, "o", ms=5, color=c, label="chained outlet surrogates"
    )
    axes[1].plot(ref.x, ref.P / 1e5, color=C_PHYS, lw=1.0)
    axes[1].plot(np.r_[0.0, x_pts], np.r_[P0, chain[:, 0]], "o", ms=5, color=c)
    dT = chain[:, 1] - np.interp(x_pts, ref.x, ref.T)
    dP = chain[:, 0] - np.interp(x_pts, ref.x, ref.P) / 1e5
    for ax, d in ((axes[2], dT), (axes[3], dP)):
        ax.vlines(x_pts, 0.0, d, color=c, lw=1.2)
        ax.plot(x_pts, d, "o", ms=5, color=c)
        ax.axhline(0.0, color="0.3", lw=0.5)
        lim = 1.3 * np.max(np.abs(d))
        ax.set_ylim(-lim, lim)
    axes[0].set_ylabel("$T$ ($^\\circ$C)")
    axes[1].set_ylabel("$P$ (bar)")
    axes[2].set_ylabel(r"$T_\mathrm{NN} - T$ (K)")
    axes[3].set_ylabel(r"$P_\mathrm{NN} - P$ (bar)")
    axes[3].set_xlabel("$x$ (m)")
    _shade_segments(axes)
    axes[0].legend(fontsize=6.5)
    for ax, letter in zip(axes, "abcd"):
        ax.grid(True)
        fl.panel_label(ax, f"({letter})")
    fig.align_ylabels(axes)
    fl.save(fig, out, "surrogate_chain", png)


def _system_profiles(models, ref, chained):
    """Perfil ao longo do sistema (x, P bar, T K) com os substitutos de perfil por trecho: a entrada de cada
    trecho é a do modelo físico (`chained=False`, mede o ajuste) ou a saída do substituto anterior
    (`chained=True`, mede o uso real)."""
    xs, Ps, Ts = [], [], []
    inlet = ref.case_dict["boundary"]["inlet"]
    P_in, T_in, m = inlet["P"] / 1e5, inlet["T"], inlet["m"]
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
        top.text(
            0.5 * (x0 + x1),
            1.03,
            SEG_LABEL[seg],
            transform=top.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def _profile_models(suffix):
    paths = {
        seg: (MODELS_DIR if suffix in ("_profile", "_pinn") else SR) / f"{seg}{suffix}.json"
        for seg, *_ in SYSTEM_SEGMENTS
    }
    if not all(p.exists() for p in paths.values()):
        return None
    return {seg: _profile_cls().load(p) for seg, p in paths.items()}


def surrogate_profile(out, png):
    """Perfil ao longo de todo o sistema: físico contra o substituto de perfil, com painéis de erro."""
    fig, axes = plt.subplots(
        4, 1, figsize=(fl.MID, 1.25 * fl.MID), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.3, 1, 1]}
    )
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
    for chained, color, ls, lab in (
        (False, fl.COLORS["noslip"], "--", "profile surrogate, segment inlets from the physical model"),
        (True, fl.COLORS["mechanistic"], "-.", "profile surrogate, chained (each inlet from the previous surrogate)"),
    ):
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
    variants = (
        ("_profile_n20", "data only, 20 profiles", C_SPARSE, "--"),
        ("_pinn", "PINN, 20 profiles + residual", C_PINN, "-"),
        ("_profile", "data only, 210 profiles", fl.COLORS["noslip"], ":"),
    )
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
    study, val = _study(), _validation()
    fig, axes = plt.subplots(2, 3, figsize=(fl.DOUBLE, 0.72 * fl.DOUBLE))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.2, wspace=0.32, hspace=0.42)
    if not val:
        for ax in axes.ravel():
            fl.placeholder(ax, "uv run python -m surrogate.validation")
        fl.save(fig, out, "surrogate_validation", png)
        return
    modes = (
        ("profile", "split by profile", fl.COLORS["noslip"], "o"),
        ("position", "split by axial position", fl.COLORS["mechanistic"], "s"),
        ("block", "block split (top 30\\% of $P_\\mathrm{in}$)", "#009E73", "^"),
    )
    floor = 1e-3  # erros de treino nulos (interpolação exata) ficam no chão do eixo logarítmico

    def curve(ax, lc, i, fmt, color, label, train=False):
        n = np.array([c["n_train"] for c in lc], float)
        if train:
            ax.plot(
                n, np.maximum([c["train_rmse_mean"][i] for c in lc], floor), fmt, color=color, ms=3, lw=0.8, label=label
            )
            return
        m = np.array([c["test_rmse_mean"][i] for c in lc])
        sd = np.array([c["test_rmse_std"][i] for c in lc])
        ax.errorbar(
            n, m, yerr=[np.minimum(sd, 0.9 * m), sd], fmt=fmt, color=color, ms=3, lw=0.8, capsize=2, label=label
        )

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


def tab_ranges(out):
    st = json.load(open(fl.DATA / "surrogate_tables.json"))
    rows = []
    for s, name in SEG:
        r = st["ranges"][s]
        rows.append(
            [
                name,
                f"{r['P_in_bar'][0]:.0f}--{r['P_in_bar'][1]:.0f}",
                f"{r['T_in_C'][0]:.0f}--{r['T_in_C'][1]:.0f}",
                f"{r['m_kg_s'][0]:.0f}--{r['m_kg_s'][1]:.0f}",
                st["n_samples"],
            ]
        )
    return fl.tex_table(
        out,
        "tab_ranges",
        ["Segment", "$P_\\mathrm{in}$ (bar)", "$T_\\mathrm{in}$ ($^\\circ$C)", "$\\dot m$ (kg/s)", "Samples"],
        rows,
        notes="Latin hypercube sampling, 300 samples per segment, 70\\,\\% for training and 30\\,\\% for testing. The ranges contain the inlet conditions of each segment when the system operates at 170 bar and 80 $^\\circ$C.",
    )


def tab_surrogate_grid(out):
    rows = []
    for s, name in SEG:
        g = _grid_rows(s)
        if g is None:
            rows.append([name, "--", "--", "--", "--", "--"])
            continue
        both = [min(float(r["r2_P"]), float(r["r2_T"])) for r in g]
        best = max(g, key=lambda r: float(r["r2_sum"]))
        ch = next(
            r
            for r in g
            if int(r["neurons"]) == CHOSEN["neurons"]
            and int(r["layers"]) == CHOSEN["layers"]
            and r["activation"] == CHOSEN["activation"]
            and float(r["learning_rate"]) == CHOSEN["learning_rate"]
        )
        rows.append(
            [
                name,
                f"{sum(v > 0.9 for v in both)}/135",
                f"{sum(v > 0.99 for v in both)}/135",
                f"{best['neurons']} neurons, {best['layers']} layer(s), {'ReLU' if best['activation'] == 'relu' else best['activation']}, $\\eta$ = {float(best['learning_rate']):g}",
                f"{float(best['r2_P']):.4f} / {float(best['r2_T']):.4f}",
                f"{float(ch['r2_P']):.4f} / {float(ch['r2_T']):.4f}",
            ]
        )
    return fl.tex_table(
        out,
        "tab_surrogate_grid",
        [
            "Segment",
            "$R^2 > 0.9$",
            "$R^2 > 0.99$",
            "Best network",
            "$R^2$ best ($P$ / $T$)",
            "$R^2$ chosen ($P$ / $T$)",
        ],
        rows,
        align="lrrlrr",
        notes="Grid of 135 networks per segment (1--3 hidden layers, ReLU/sigmoid/tanh, 5--25 neurons, learning rate 0.1/0.01/0.001), 2000 epochs of full-batch Adam on the same split; counts of networks with both $R^2$ above the threshold. Chosen network: 1 hidden layer of 20 ReLU units, learning rate 0.01 (122 parameters).",
    )


def tab_surrogate_test(out):
    MLP = _surrogate()
    rows = []
    for s, name in SEG:
        p = MODELS_DIR / f"{s}.json"
        if not p.exists():
            rows.append([name, "--", "--", "--", "--"])
            continue
        t = MLP.load(p).meta["test"]
        rows.append(
            [
                name,
                f"{t['r2'][0]:.4f} / {t['r2'][1]:.4f}",
                f"{t['rmse'][0]:.3f} / {t['rmse'][1]:.3f}",
                f"{t['mean_percent_error'][0]:.2f} / {t['mean_percent_error'][1]:.2f}",
                f"{t['max_abs_error'][0]:.2f} / {t['max_abs_error'][1]:.2f}",
            ]
        )
    return fl.tex_table(
        out,
        "tab_surrogate_test",
        ["Segment", "$R^2$ ($P$ / $T$)", "RMSE (bar / K)", "Mean error (\\%) ($P$ / $T$)", "Max. error (bar / K)"],
        rows,
        align="lrrrr",
        notes="90 test samples per segment (30\\% of 300), against the physical model of Part I.",
    )


def tab_chain(out):
    ref_path = SR / "offshore_table6_noslip.npz"
    rows = []
    if ref_path.exists() and all(_exists(MODELS_DIR / f"{s}.json") for s, _ in SEG):
        from surrogate.predict import chain_predict

        MLP = _surrogate()
        ref = fl.load_run(ref_path)
        inlet = ref.case_dict["boundary"]["inlet"]
        chain = chain_predict(
            [MLP.load(MODELS_DIR / f"{s}.json") for s, _ in SEG], inlet["P"] / 1e5, inlet["T"], inlet["m"]
        )[:, 0, :]
        for k, (x, lab) in enumerate(((1500.0, "wellhead"), (7500.0, "riser base"), (7800.0, "topside arrival"))):
            Pf, Tf = np.interp(x, ref.x, ref.P) / 1e5, np.interp(x, ref.x, ref.T)
            Ps, Ts = chain[k]
            rows.append(
                [
                    lab,
                    f"{x:.0f}",
                    f"{Pf:.2f} / {Ps:.2f}",
                    f"{Ps - Pf:+.2f} ({100*abs(Ps-Pf)/Pf:.2f}\\,\\%)",
                    f"{Tf-273.15:.2f} / {Ts-273.15:.2f}",
                    f"{Ts - Tf:+.2f}",
                ]
            )
    else:
        rows = [["--"] * 6]
    return fl.tex_table(
        out,
        "tab_chain",
        ["Point", "$x$ (m)", "$P$ phys. / NN (bar)", "$P$ error (bar)", "$T$ phys. / NN ($^\\circ$C)", "$T$ error (K)"],
        rows,
        align="lrrrrr",
        notes="Physical model of Part I (phys.) and chained outlet surrogates (NN); inlet 170 bar, 80 $^\\circ$C, 20 kg/s.",
    )


def _sci(v):
    """Número em notação científica LaTeX, p. ex. 1.2 × 10^5."""
    if v is None or not np.isfinite(v) or v <= 0:
        return "--"
    e = int(np.floor(np.log10(v)))
    return f"${v / 10**e:.1f}\\times10^{{{e}}}$"


def tab_speed(out):
    """Custo por avaliação: o modelo físico do volume I (compilado) contra o substituto encadeado."""
    path = fl.RUNS / "performance_compiled" / "timings.json"
    t = json.load(open(path)) if path.exists() else {}
    sysm, well, vm = t.get("offshore_table6_march", {}), t.get("single_well_march", {}), t.get("vmap_well_300", {})
    base = sysm.get("compiled_mean_wall")
    rows = [
        ["Physical model (Part I): march of the whole system, compiled", fl.fmt(base, "{:.2f} s"), "1"],
        [
            "Physical model (Part I): march of the well segment, compiled",
            fl.fmt(well.get("compiled_mean_wall"), "{:.2f} s"),
            "--",
        ],
        [
            "Physical model (Part I): vectorised march, per case of the well segment",
            fl.fmt(vm["second_wall"] / vm["n"] if vm else None, "{:.2f} s"),
            "--",
        ],
    ]
    if all(_exists(MODELS_DIR / f"{s}.json") for s, _ in SEG):
        from surrogate.predict import chain_predict

        MLP = _surrogate()
        ms = [MLP.load(MODELS_DIR / f"{s}.json") for s, _ in SEG]
        X = np.random.default_rng(0).uniform([145, 333, 2], [185, 373, 25], (10000, 3))
        t0 = time.perf_counter()
        for _ in range(200):
            chain_predict(ms, X[:1, 0], X[:1, 1], X[:1, 2])
        t_one = (time.perf_counter() - t0) / 200
        t0 = time.perf_counter()
        chain_predict(ms, X[:, 0], X[:, 1], X[:, 2])
        t_batch = (time.perf_counter() - t0) / 1e4
        rows.append(
            [
                "Surrogate chain (three segments), one call",
                f"{t_one*1e6:.0f} $\\mu$s",
                _sci(base / t_one if base else None),
            ]
        )
        rows.append(
            [
                "Surrogate chain, per case in a batch of 10\\,000",
                f"{t_batch*1e6:.2f} $\\mu$s",
                _sci(base / t_batch if base else None),
            ]
        )
    else:
        rows += [["Surrogate chain", "--", "--"]] * 2
    return fl.tex_table(
        out,
        "tab_speed",
        ["Evaluation", "Wall time", "Speed-up"],
        rows,
        align="p{8.5cm}rr",
        notes="Apple silicon laptop, CPU only; physical-model times exclude the one-off just-in-time compilation; the surrogate runs in plain NumPy. Speed-up relative to the compiled march of the whole system.",
    )


def tab_profile_pinn(out):
    study = _study()
    rows = []
    for s, name in SEG:
        if s in study:
            for key, lab in LABELS.items():
                r = study[s]["models"][key]
                t = r["test_profile"]
                rows.append(
                    [
                        name,
                        lab,
                        r["n_train_profiles"],
                        f"{t['rmse'][0]:.3f} / {t['rmse'][1]:.3f}",
                        f"{t['max_abs_error'][0]:.2f} / {t['max_abs_error'][1]:.2f}",
                        f"{r['residual_rms']:.3f}",
                    ]
                )
        elif _exists(MODELS_DIR / f"{s}_profile.json"):
            meta = _profile_cls().load(MODELS_DIR / f"{s}_profile.json").mlp.meta
            t = meta["test_profile"]
            rows.append(
                [
                    name,
                    LABELS["data_full"],
                    meta.get("n_train_profiles", "--"),
                    f"{t['rmse'][0]:.3f} / {t['rmse'][1]:.3f}",
                    f"{t['max_abs_error'][0]:.2f} / {t['max_abs_error'][1]:.2f}",
                    "--",
                ]
            )
        else:
            rows.append([name, "--", "--", "--", "--", "--"])
    return fl.tex_table(
        out,
        "tab_profile_pinn",
        ["Segment", "Model", "Train. profiles", "RMSE (bar / K)", "Max. error (bar / K)", "RMS residual"],
        rows,
        align="llrrrr",
        notes="Errors over the 90 test profiles (51 axial points each). The residual is $(\\mathrm{d}u/\\mathrm{d}x - f)/\\mathrm{scale}$ on the collocation set; RMS $\\approx 1$ means deviations of the order of the typical slope.",
    )


def tab_validation(out):
    val = _validation()
    rows = []
    for s, name in SEG:
        kf = val.get(s, {}).get("outlet", {}).get("kfold", {}).get("summary")
        lc = val.get(s, {}).get("outlet", {}).get("learning_curve", {}).get("curve", [])
        if kf:
            rows.append(
                [
                    name,
                    "outlet, 5-fold CV (210 train)",
                    f"{kf['test_rmse_mean'][0]:.3f} $\\pm$ {kf['test_rmse_std'][0]:.3f} / {kf['test_rmse_mean'][1]:.3f} $\\pm$ {kf['test_rmse_std'][1]:.3f}",
                    f"{kf['gap_mean'][0]:+.3f} / {kf['gap_mean'][1]:+.3f}",
                    f"{kf['bias_mean'][0]:+.3f} / {kf['bias_mean'][1]:+.3f}",
                ]
            )
        for c in lc:
            if c["n_train"] in (20, 210):
                rows.append(
                    [
                        name,
                        f"outlet, {c['n_train']} train, 3 splits",
                        f"{c['test_rmse_mean'][0]:.3f} $\\pm$ {c['test_rmse_std'][0]:.3f} / {c['test_rmse_mean'][1]:.3f} $\\pm$ {c['test_rmse_std'][1]:.3f}",
                        f"{c['gap_mean'][0]:+.3f} / {c['gap_mean'][1]:+.3f}",
                        f"{c['bias_mean'][0]:+.3f} / {c['bias_mean'][1]:+.3f}",
                    ]
                )
        for mode, lab in (("profile", "by profile"), ("position", "by position"), ("block", "block")):
            for c in val.get(s, {}).get("profile", {}).get(mode, {}).get("curve", []):
                if c["n_train"] in (20, 210):
                    rows.append(
                        [
                            name,
                            f"profile, {c['n_train']} train, split {lab}",
                            f"{c['test_rmse_mean'][0]:.3f} $\\pm$ {c['test_rmse_std'][0]:.3f} / {c['test_rmse_mean'][1]:.3f} $\\pm$ {c['test_rmse_std'][1]:.3f}",
                            f"{c['gap_mean'][0]:+.3f} / {c['gap_mean'][1]:+.3f}",
                            f"{c['bias_mean'][0]:+.3f} / {c['bias_mean'][1]:+.3f}",
                        ]
                    )
    if not rows:
        rows = [["--"] * 5]
    return fl.tex_table(
        out,
        "tab_validation",
        [
            "Segment",
            "Model, training set",
            "Test RMSE, mean $\\pm$ std ($P$ bar / $T$ K)",
            "Test $-$ train RMSE",
            "Bias ($P$ / $T$)",
        ],
        rows,
        align="llrrr",
        notes="Repeated random splits (seeds 0--2) or 5-fold cross-validation; the test--train gap measures overfitting, the bias is the mean signed error on the test set. Block split: test profiles are those with $P_\\mathrm{in}$ in the top 30\\% of the range (extrapolation).",
    )


def tab_residual(out):
    val, study = _validation(), _study()
    rows = []
    for s, name in SEG:
        held = val.get(s, {}).get("profile", {}).get("heldout_residual", {})
        for key, lab in LABELS.items():
            if s in study and key in study[s]["models"]:
                r = study[s]["models"][key]
                rows.append(
                    [
                        name,
                        lab,
                        r["n_train_profiles"],
                        f"{r['residual_rms']:.3f}",
                        f"{held[key]:.3f}" if key in held else "--",
                    ]
                )
    if not rows:
        rows = [["--"] * 5]
    return fl.tex_table(
        out,
        "tab_residual",
        ["Segment", "Model", "Train. profiles", "RMS residual, training set", "RMS residual, held-out set"],
        rows,
        align="llrrr",
        notes="Scaled residual of the governing equations at 60 collocation points: the set used in the PINN loss and a set never used in training.",
    )


FIGURES = {
    "surrogate_training": surrogate_training,
    "surrogate_parity": surrogate_parity,
    "surrogate_grid": surrogate_grid,
    "surrogate_chain": surrogate_chain,
    "surrogate_profile": surrogate_profile,
    "pinn_study": pinn_study,
    "surrogate_validation": surrogate_validation,
}
TABLES = {
    "tab_ranges": tab_ranges,
    "tab_surrogate_grid": tab_surrogate_grid,
    "tab_surrogate_test": tab_surrogate_test,
    "tab_chain": tab_chain,
    "tab_speed": tab_speed,
    "tab_profile_pinn": tab_profile_pinn,
    "tab_validation": tab_validation,
    "tab_residual": tab_residual,
}

if __name__ == "__main__":
    fl.run_cli(FIGURES, __doc__.splitlines()[0], tables=TABLES)
