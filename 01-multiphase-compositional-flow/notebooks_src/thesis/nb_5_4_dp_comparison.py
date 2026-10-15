# %% [markdown]
# # §5.4 — Perda de carga em 100 linhas horizontais e 100 verticais: cmfsim contra o Apêndice A
#
# Lê as corridas do lote `batches/dp_comparison.json` (`runs/dp_comparison/`) e os CSV extraídos do
# PDF (`dissertation/data/appendix_a_*.csv`); não simula. Compara a coluna «Dissertação» (modelo
# com escorregamento, método 3.2) com o modelo mecanicista do cmfsim e, a título de contexto, com
# o modelo de não escorregamento e as colunas dos modelos do HYSYS.
# %%
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))
from cmfsim import load_run  # noqa: E402

RUNS = ROOT / "runs" / "dp_comparison"
DATA = ROOT / "dissertation" / "data"


def appendix(orientation):
    with open(DATA / f"appendix_a_{orientation}.csv") as f:
        return list(csv.DictReader(f))


def dp_runs(orientation, model):
    out = {}
    for k in range(1, 101):
        p = RUNS / f"dp_{orientation}_{model}_{k:03d}.npz"
        if p.exists():
            r = load_run(p)
            out[k] = dict(dp=(r.P[0] - r.P[-1]) / 1e5, ok=r.info.get("converged", True), info=r.info, run=r)
    return out


# %%
summary = {}
for orientation in ("horizontal", "vertical"):
    rows = appendix(orientation)
    ref = np.array([float(r["dp_bar_dissertation"]) for r in rows])
    print(f"== {orientation}")
    for model in ("mechanistic", "noslip"):
        got = dp_runs(orientation, model)
        if not got:
            print(f"   {model}: sem corridas")
            continue
        idx = np.array(sorted(got))
        mine = np.array([got[k]["dp"] for k in idx])
        rel = (mine - ref[idx - 1]) / ref[idx - 1]
        conv = np.array([got[k]["ok"] for k in idx])
        summary[(orientation, model)] = (idx, mine, rel, conv)
        print(
            f"   {model}: {idx.size} corridas, convergidas {conv.sum()}; desvio relativo à coluna Dissertação: "
            f"mediana {np.median(rel):+.1%}, |max| {np.max(np.abs(rel)):.1%}, dentro de 10 %: {np.mean(np.abs(rel) < 0.1):.0%}, de 25 %: {np.mean(np.abs(rel) < 0.25):.0%}"
        )
    # piores casos
    if (orientation, "mechanistic") in summary:
        idx, mine, rel, conv = summary[(orientation, "mechanistic")]
        worst = np.argsort(-np.abs(rel))[:5]
        for j in worst:
            r = rows[idx[j] - 1]
            print(f"      caso {idx[j]:3d} D={r['D_in']} in P={r['P_in_bar']} bar m={r['m_kg_s']} kg/s: tese {ref[idx[j]-1]:.4g} bar, cmfsim {mine[j]:.4g} bar ({rel[j]:+.0%})")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, orientation in zip(axes, ("horizontal", "vertical")):
    rows = appendix(orientation)
    ref = np.array([float(r["dp_bar_dissertation"]) for r in rows])
    for model, mk in (("mechanistic", "o"), ("noslip", "x")):
        if (orientation, model) in summary:
            idx, mine, rel, conv = summary[(orientation, model)]
            ax.loglog(ref[idx - 1], mine, mk, ms=4, label=f"cmfsim {model}")
    lo, hi = ref.min() * 0.5, ref.max() * 2
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.plot([lo, hi], [lo * 1.25, hi * 1.25], "k:", lw=0.6)
    ax.plot([lo, hi], [lo * 0.8, hi * 0.8], "k:", lw=0.6)
    ax.set_xlabel("ΔP dissertação (bar)")
    ax.set_ylabel("ΔP cmfsim (bar)")
    ax.set_title(orientation)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
out = ROOT / "notebooks" / "figs" / "nb_5_4_dp_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print("figura:", out)
