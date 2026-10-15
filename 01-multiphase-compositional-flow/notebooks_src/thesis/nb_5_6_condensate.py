# %% [markdown]
# # §5.6 — Gasoduto com formação de condensado (Figuras 38–40, Tabelas 23–26)
#
# Seis casos (T_in 60/80 °C × U 1/3/6 W/m²K) do gasoduto de 8 km e 12 in com o gás rico da Tabela 23,
# modelo de não escorregamento por marcha em x. Corridas do lote `batches/dissertation.json`
# (`runs/dissertation/condensate_5_6_case*.npz`); variante com a hipótese do Cp em °C (issue 004)
# em `runs/dissertation_celsius/`. Figuras 38–40 digitalizadas sobrepostas. Não simula.
# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np

from dissertation.notebook_utils import RUNS, figure, load_runs, make_fluid, plt, save, table, tables

from cmfsim.thermo import FlashOptions, phase_state

figs = {n: figure(n) for n in (38, 39, 40)}
runs = load_runs("dissertation/condensate_5_6_case*.npz")
runs_c = load_runs("dissertation_celsius/condensate_5_6_case*.npz")
print("corridas:", list(runs), "| hipótese °C:", len(runs_c))
t25 = tables()["table25_condensate_cases"]


def at(fig, name, x):
    xs, ys = fig["series"][name]
    return float(np.interp(x, xs, ys)) if xs.min() <= x <= xs.max() else np.nan


def x_cond(xs, w):
    m = w < 0.9995
    return float(xs[m][0]) if np.any(m) else np.nan


# %% [markdown]
# ## Tabelas 23–25 (entradas) e saída de cada caso
# %%
rows = []
for c in t25:
    k = c["case"]
    r = runs.get(f"condensate_5_6_case{k}")
    rc = runs_c.get(f"condensate_5_6_case{k}")
    s = str(k)
    xd, yd = figs[40]["series"][s]
    xc_d = x_cond(xd, yd)
    rows.append(
        (
            k,
            f"{c['T_in_C']:.0f}",
            f"{c['U']:.0f}",
            f"{at(figs[39], s, 7950):.1f}",
            f"{r.T[-1] - 273.15:.1f}" if r else "—",
            f"{rc.T[-1] - 273.15:.1f}" if rc else "—",
            f"{at(figs[38], s, 7950):.2f}",
            f"{r.P[-1] / 1e5:.2f}" if r else "—",
            f"{at(figs[40], s, 7950):.4f}",
            f"{r['w_G'][-1]:.4f}" if r else "—",
            f"{xc_d:.0f}" if np.isfinite(xc_d) else "—",
            f"{x_cond(r.x, r['w_G']):.0f}" if r and np.isfinite(x_cond(r.x, r["w_G"])) else "—",
        )
    )
table(
    rows,
    [
        "caso",
        "T_in",
        "U",
        "T_out tese",
        "T_out cmfsim",
        "T_out °C-hip.",
        "P_out tese",
        "P_out cmfsim",
        "wG_out tese",
        "wG_out cmfsim",
        "x_cond tese",
        "x_cond cmfsim",
    ],
)

# %% [markdown]
# ## Figuras 38–40
# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for c in t25:
    k = c["case"]
    s = str(k)
    col = colors[(k - 1) % len(colors)]
    r = runs.get(f"condensate_5_6_case{k}")
    if r:
        axes[0].plot(r.x, r.P / 1e5, color=col, label=f"caso {k} cmfsim")
        axes[1].plot(r.x, r.T - 273.15, color=col)
        axes[2].plot(r.x, r["w_G"], color=col)
    rc = runs_c.get(f"condensate_5_6_case{k}")
    if rc:
        axes[1].plot(rc.x, rc.T - 273.15, "--", color=col, lw=0.8)
        axes[2].plot(rc.x, rc["w_G"], "--", color=col, lw=0.8)
    for ax, n in zip(axes, (38, 39, 40)):
        if s in figs[n]["series"]:
            ax.plot(
                *figs[n]["series"][s], ":", color=col, lw=2, label=f"caso {k} dissertação" if ax is axes[0] else None
            )
axes[0].set_ylabel("P (bar)")
axes[1].set_ylabel("T (°C)  (tracejado: hipótese Cp em °C)")
axes[2].set_ylabel("fração vaporizada mássica")
axes[2].set_xlabel("x (m)")
axes[0].legend(fontsize=7, ncol=3)
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle("Figuras 38–40 — cmfsim (linhas) contra a dissertação (pontilhado)")
save(fig, "nb_5_6_fig38_40")

# %% [markdown]
# ## Tabela 26 — composições na saída do caso 2 (título da tabela diz caso 1; pela Figura 40 é o caso 2)
# %%
cs, z = make_fluid("table23_rich_gas")
t26 = tables()["table26_outlet_composition_case2"]
r = runs.get("condensate_5_6_case2")
names = list(t26["global"])
rows = []
if r is not None:
    # na saída o cmfsim ainda não condensou (issue 004): mostra-se o flash às condições da dissertação
    # (T e P da saída das Figuras 38–39, caso 2) e às nossas
    P_d, T_d = at(figs[38], "2", 7950) * 1e5, at(figs[39], "2", 7950) + 273.15
    st_d = phase_state(np.array([T_d]), np.array([P_d]), z, cs, FlashOptions(stability_test=True))
    for i, nm in enumerate(names):
        rows.append(
            (
                nm,
                f"{t26['global'][nm]:.3f}",
                f"{t26['liquid'][nm]:.3f}",
                f"{st_d.x[0, i]:.3f}",
                f"{t26['vapor'][nm]:.3f}",
                f"{st_d.y[0, i]:.3f}",
            )
        )
    table(rows, ["componente", "global", "líquido tese", "líquido cmfsim*", "vapor tese", "vapor cmfsim*"])
    print(
        f"* flash do cmfsim nas condições de saída da dissertação (P = {P_d/1e5:.2f} bar, T = {T_d-273.15:.1f} °C); beta = {st_d.beta[0]:.4f}"
    )
