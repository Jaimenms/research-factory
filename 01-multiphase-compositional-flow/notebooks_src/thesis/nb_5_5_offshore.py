# %% [markdown]
# # §5.5 — Sistema de produção offshore: cmfsim contra as Figuras 32–37
#
# Lê `runs/dissertation/offshore_5_5_{noslip,slip}.npz` e as figuras digitalizadas; não simula.
# %%
import json
import os
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
RUNS = Path(os.environ.get("CMFSIM_RUNS", ROOT / "runs" / "dissertation"))


def digitized(n):
    d = json.load(open(FIG / f"fig_{n:02d}.json"))
    out = {}
    for s in d["series"]:
        x, y = np.array(s["x"]), np.array(s["y"])
        o = np.argsort(x)
        out[s["name"]] = (x[o], y[o])
    return out


figs = {n: digitized(n) for n in (32, 33, 34, 37)}
runs = {k: load_run(RUNS / f"offshore_5_5_{k}.npz") for k in ("noslip", "slip") if (RUNS / f"offshore_5_5_{k}.npz").exists()}
pair = {"noslip": "Modelo no-slip", "slip": "Modelo"}
print("corridas:", list(runs))

# %% [markdown]
# ## Desvios nos pontos de quebra (topo do poço, base e topo do riser)
# %%
for k, r in runs.items():
    nm = pair[k]
    print(f"== {k} (dissertação: «{nm}»)")
    for xq in (1500.0, 7500.0, 7799.0):
        i = np.argmin(np.abs(r.x - xq))
        line = f"  x={xq:5.0f}: P {r.P[i]/1e5:7.2f}"
        x, y = figs[32][nm]
        if x.min() <= xq <= x.max():
            line += f" / {np.interp(xq, x, y):7.2f} bar"
        line += f" | T {r.T[i]-273.15:5.1f}"
        x, y = figs[33][nm]
        if x.min() <= xq <= x.max():
            line += f" / {np.interp(xq, x, y):5.1f} °C"
        line += f" | H_L {r['H_L'][i]:.3f}"
        x, y = figs[34][nm]
        if x.min() <= xq <= x.max():
            line += f" / {np.interp(xq, x, y):.3f}"
        line += f" | padrão {r['pattern_code'][i]}"
        print(line)

# %%
fig, axes = plt.subplots(4, 1, figsize=(9, 13), sharex=True)
for k, r in runs.items():
    c = "C0" if k == "noslip" else "C3"
    nm = pair[k]
    axes[0].plot(r.x, r.P / 1e5, color=c, label=f"cmfsim {k}")
    axes[1].plot(r.x, r.T - 273.15, color=c)
    axes[2].plot(r.x, r["H_L"], color=c)
    axes[3].plot(r.x, r["pattern_code"], color=c, drawstyle="steps-post")
    for ax, n in zip(axes, (32, 33, 34, 37)):
        if nm in figs[n]:
            ax.plot(figs[n][nm][0], figs[n][nm][1], ":", color=c, lw=2, label=f"dissertação «{nm}»" if ax is axes[0] else None)
for n, nm in ((32, "TULSA"), (32, "BB")):
    if nm in figs[n]:
        axes[0].plot(figs[n][nm][0], figs[n][nm][1], "--", lw=1, label=f"HYSYS {nm}")
axes[0].set_ylabel("P (bar)")
axes[1].set_ylabel("T (°C)")
axes[2].set_ylabel("holdup H_L")
axes[3].set_ylabel("padrão (0 estrat., 2 anular, 3 interm., 4 bolhas)")
axes[3].set_xlabel("x (m)")
axes[0].legend(fontsize=7, ncol=2)
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle("§5.5 sistema offshore: cmfsim (linhas) contra as figuras digitalizadas (pontilhado)")
out = ROOT / "notebooks" / "figs" / "nb_5_5_offshore.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("figura:", out)
