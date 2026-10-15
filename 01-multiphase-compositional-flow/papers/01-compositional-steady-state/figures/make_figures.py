"""Figuras do artigo 01 a partir das corridas gravadas em `runs/` (não simula).

uso: uv run python figures/make_figures.py --out figures/generated [--only NOME] [--no-png]
     (a partir da pasta do artigo; `make figures` faz o mesmo)

Figuras:
  offshore_profiles  §5.5 da dissertação: perfis de P, T e holdup no sistema poço–linha–riser, os dois
                     modelos deste trabalho contra as curvas digitalizadas (Figuras 32–34)
  dp_parity          Apêndice A: paridade da perda de carga nos 200 casos (horizontal e vertical)

Corridas necessárias: `uv run cmfsim batch batches/dissertation.json` (offshore) e
`uv run python -m generators.dp_comparison && uv run cmfsim batch batches/dp_comparison.json` (paridade).
Sem corridas, a figura sai com um aviso no lugar dos dados, para o PDF compilar na mesma.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_common"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figlib as fl  # noqa: E402

# (chave da corrida, nome da série na figura digitalizada, rótulo em inglês, cor por entidade)
MODELS = (
    ("noslip", "Modelo no-slip", "no-slip", fl.COLORS["noslip"]),
    ("slip", "Modelo", "mechanistic", fl.COLORS["mechanistic"]),
)


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
            runs = fl.load_runs(f"dp_comparison/dp_{orientation}_{model}_*.npz")
            if not runs:
                continue
            found = True
            idx = np.array([int(name.rsplit("_", 1)[1]) for name in runs])
            dp = np.array([(r.P[0] - r.P[-1]) / 1e5 for r in runs.values()])
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


FIGURES = {"offshore_profiles": offshore_profiles, "dp_parity": dp_parity}

if __name__ == "__main__":
    fl.run_cli(FIGURES, __doc__.splitlines()[0])
