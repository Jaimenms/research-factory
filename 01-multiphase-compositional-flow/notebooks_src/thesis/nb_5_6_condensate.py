# %% [markdown]
# # §5.6 — Gasoduto com formação de condensado: cmfsim contra as Figuras 38–40
#
# Lê as seis corridas do lote `batches/dissertation.json` (`runs/dissertation/condensate_5_6_case*.npz`)
# e as figuras digitalizadas (`references/figures/data/fig_38..40.json`); não simula.
# %%
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))
from cmfsim import load_run  # noqa: E402

FIG = ROOT / "references" / "figures" / "data"
import os

RUNS = Path(os.environ.get("CMFSIM_RUNS", ROOT / "runs" / "dissertation"))


def digitized(n):
    d = json.load(open(FIG / f"fig_{n:02d}.json"))
    out = {}
    for s in d["series"]:
        x, y = np.array(s["x"]), np.array(s["y"])
        o = np.argsort(x)
        out[s["name"]] = (x[o], y[o])
    return out


def at(xs, ys, x):
    return float(np.interp(x, xs, ys))


fig38, fig39, fig40 = digitized(38), digitized(39), digitized(40)
runs = {k: load_run(RUNS / f"condensate_5_6_case{k}.npz") for k in range(1, 7) if (RUNS / f"condensate_5_6_case{k}.npz").exists()}
print(f"{len(runs)} corridas lidas")

# %% [markdown]
# ## Tabela de desvios na saída (x = 8000 m) e início da condensação
# %%
rows = []
for k, r in runs.items():
    s = str(k)
    T_d = at(*fig39[s], 7950.0) if s in fig39 else np.nan
    P_d = at(*fig38[s], 7950.0) if s in fig38 else np.nan
    wG_d = at(*fig40[s], 7950.0) if s in fig40 else np.nan
    # início da condensação: primeiro x com fração vaporizada < 0,9995
    xd, yd = fig40.get(s, (np.array([]), np.array([])))
    x_cond_d = float(xd[yd < 0.9995][0]) if np.any(yd < 0.9995) else np.nan
    T_cond_d = at(*fig39[s], x_cond_d) if np.isfinite(x_cond_d) else np.nan
    m = r["w_G"] < 0.9995
    x_cond = float(r.x[m][0]) if np.any(m) else np.nan
    T_cond = float(np.interp(x_cond, r.x, r.T) - 273.15) if np.isfinite(x_cond) else np.nan
    rows.append(
        dict(
            caso=k,
            T_out_tese=T_d,
            T_out=r.T[-1] - 273.15,
            P_out_tese=P_d,
            P_out=r.P[-1] / 1e5,
            wG_out_tese=wG_d,
            wG_out=float(r["w_G"][-1]),
            x_cond_tese=x_cond_d,
            x_cond=x_cond,
            T_cond_tese=T_cond_d,
            T_cond=T_cond,
            passos=r.info["n_accepted"],
            cpu=r.info["cpu"],
        )
    )
hdr = "caso | T_out tese/cmfsim (°C) | P_out tese/cmfsim (bar) | wG_out tese/cmfsim | x_cond tese/cmfsim (m) | T_cond tese/cmfsim (°C) | passos | cpu (s)"
print(hdr)
for r in rows:
    print(
        f"{r['caso']:4d} | {r['T_out_tese']:6.1f} / {r['T_out']:6.1f} | {r['P_out_tese']:6.2f} / {r['P_out']:6.2f} | "
        f"{r['wG_out_tese']:6.4f} / {r['wG_out']:6.4f} | {r['x_cond_tese']:6.0f} / {r['x_cond']:6.0f} | "
        f"{r['T_cond_tese']:5.1f} / {r['T_cond']:5.1f} | {r['passos']:4d} | {r['cpu']:5.1f}"
    )

# %% [markdown]
# ## Perfis sobrepostos
# %%
fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for k, r in runs.items():
    c = colors[(k - 1) % len(colors)]
    s = str(k)
    axes[0].plot(r.x, r.P / 1e5, color=c, label=f"caso {k} cmfsim")
    axes[1].plot(r.x, r.T - 273.15, color=c)
    axes[2].plot(r.x, r["w_G"], color=c)
    for ax, fg in zip(axes, (fig38, fig39, fig40)):
        if s in fg:
            ax.plot(fg[s][0], fg[s][1], ":", color=c, lw=2, label=f"caso {k} dissertação" if ax is axes[0] else None)
axes[0].set_ylabel("P (bar)")
axes[1].set_ylabel("T (°C)")
axes[2].set_ylabel("fração vaporizada mássica w_G")
axes[2].set_xlabel("x (m)")
axes[0].legend(fontsize=7, ncol=3)
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle("§5.6 gasoduto: cmfsim (linhas) contra as Figuras 38–40 digitalizadas (pontilhado)")
out = ROOT / "notebooks" / "figs" / "nb_5_6_condensate.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print("figura:", out)
