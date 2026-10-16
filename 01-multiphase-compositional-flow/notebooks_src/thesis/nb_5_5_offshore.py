# %% [markdown]
# # §5.5 — Sistema de produção offshore (Figuras 31–37, Tabelas 19–22)
#
# Poço vertical de 1500 m, linha horizontal de 6000 m e riser vertical de 300 m (um tubo com perfil
# axial), fluido de 9 componentes com C7+ (Tabela 19), 170 bar, 80 °C e 10 kg/s na entrada. Dois
# modelos (não escorregamento por marcha; mecanicista por diferenças finitas), corridas
# `runs/dissertation/offshore_5_5_{noslip,slip}.npz`; curvas da dissertação («Modelo», «Modelo
# no-slip») e do HYSYS (TULSA, Beggs & Brill) digitalizadas das Figuras 32–37. Não simula.
# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np

from dissertation.notebook_utils import (
    components_db,
    figure,
    fluids_db,
    load_runs,
    make_fluid,
    plt,
    save,
    table,
    tables,
)

figs = {n: figure(n) for n in (32, 33, 34, 35, 36, 37)}
runs = load_runs("dissertation/offshore_5_5_*.npz")
runs = {k.replace("offshore_5_5_", ""): v for k, v in runs.items()}
pair = {"noslip": "Modelo no-slip", "slip": "Modelo"}
ALIAS = {"Modelo": ("Modelo", "Modelo mecanístico"), "Modelo no-slip": ("Modelo no-slip",)}  # nomes por figura


def series_of(n, name):
    for alias in ALIAS.get(name, (name,)):
        if alias in figs[n]["series"]:
            return figs[n]["series"][alias]
    return None


print("corridas:", list(runs))
for n in (32, 33, 34, 35, 36, 37):
    print(f"Figura {n}: séries {list(figs[n]['series'])}")

# %% [markdown]
# ## Tabelas 19–22 (entradas) e Figura 31 (esquema)
# %%
t = tables()
cs, z = make_fluid("table19_offshore")
table([(nm, f"{zi:.3f}") for nm, zi in zip(cs.names, z)], ["componente", "fração molar"])
rows = [
    (
        k,
        f"{v['D']:.4f}",
        f"{v['roughness']*1e3:.1f}",
        f"{v['L']:.0f}",
        f"{v['elevation']:.0f}",
        f"{v['U']:.0f}",
        f"{v['T_amb_C']:.0f}",
    )
    for k, v in t["table21_offshore_pipes"].items()
]
table(rows, ["trecho", "D (m)", "rugosidade (mm)", "L (m)", "elevação (m)", "U (W/m²K)", "T_amb (°C)"])
print("Tabela 22: P_in 170 bar, T_in 80 °C, m 10 kg/s")


def at(n, name, x):
    s = series_of(n, name)
    if s is None:
        return np.nan
    xs, ys = s
    return float(np.interp(x, xs, ys)) if xs.min() <= x <= xs.max() else np.nan


# %% [markdown]
# ## Pontos de quebra: pressão, temperatura, holdup e padrão
# %%
rows = []
for k, r in runs.items():
    nm = pair[k]
    for xq in (1500.0, 7500.0, 7799.0):
        i = int(np.argmin(np.abs(r.x - xq)))
        rows.append(
            (
                k,
                f"{xq:.0f}",
                f"{r.P[i]/1e5:.2f}",
                f"{at(32, nm, xq):.2f}",
                f"{r.T[i]-273.15:.1f}",
                f"{at(33, nm, xq):.1f}",
                f"{r['H_L'][i]:.3f}",
                f"{at(34, nm, xq):.3f}",
                f"{r['pattern_code'][i]:.0f}",
                f"{at(37, nm, xq):.0f}",
            )
        )
table(
    rows,
    [
        "modelo",
        "x (m)",
        "P cmfsim",
        "P tese",
        "T cmfsim",
        "T tese",
        "H_L cmfsim",
        "H_L tese",
        "padrão cmfsim",
        "padrão tese",
    ],
)

# %% [markdown]
# ## Figuras 32–34 e 37 — perfis
# %%
fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
for k, r in runs.items():
    c = "C0" if k == "noslip" else "C3"
    nm = pair[k]
    axes[0].plot(r.x, r.P / 1e5, color=c, label=f"cmfsim {k}")
    axes[1].plot(r.x, r.T - 273.15, color=c)
    axes[2].plot(r.x, r["H_L"], color=c)
    axes[3].plot(r.x, r["pattern_code"], color=c, drawstyle="steps-post")
    for ax, n in zip(axes, (32, 33, 34, 37)):
        s = series_of(n, nm)
        if s is not None:
            ax.plot(*s, ":", color=c, lw=2, label=f"dissertação «{nm}»" if ax is axes[0] else None)
for nm, ls in (("TULSA", "--"), ("BB", "-."), ("Beggs & Brill", "-.")):
    for ax, n in zip(axes, (32, 33, 34, 37)):
        if nm in figs[n]["series"]:
            ax.plot(*figs[n]["series"][nm], ls, lw=0.8, color="gray", label=f"HYSYS {nm}" if ax is axes[0] else None)
axes[0].set_ylabel("P (bar)")
axes[1].set_ylabel("T (°C)")
axes[2].set_ylabel("holdup H_L")
axes[3].set_ylabel("padrão (0 estrat., 2 anular, 3 interm., 4 bolhas)")
axes[3].set_xlabel("x (m)")
axes[0].legend(fontsize=7, ncol=2)
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle("Figuras 32–34 e 37 — cmfsim contra a dissertação e o HYSYS")
save(fig, "nb_5_5_fig32_37")

# %% [markdown]
# ## Figura 35 — fração de líquido: flash (equilíbrio) contra mecanicista
# %%
if "slip" in runs:
    r = runs["slip"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r.x, r["H_L_noslip"], "C0", label="fração de líquido do flash (λ_L) cmfsim")
    ax.plot(r.x, r["H_L"], "C3", label="holdup mecanicista cmfsim")
    for nm, (x, y) in figs[35]["series"].items():
        ax.plot(x, y, ":", lw=2, label=f"dissertação «{nm}»")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("fração de líquido")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "nb_5_5_fig35")

# %% [markdown]
# ## Figura 36 — CO₂ nas fases líquido e vapor ao longo do escoamento
# %%
r = runs.get("slip", runs.get("noslip"))
if r is not None:
    i_co2 = list(cs.names).index("CO2")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r.x, 100 * r["x_comp"][:, i_co2], "C0", label="CO₂ no líquido, cmfsim (% mol)")
    ax.plot(r.x, 100 * r["y_comp"][:, i_co2], "C3", label="CO₂ no vapor, cmfsim (% mol)")
    for nm, (x, y) in figs[36]["series"].items():
        ax.plot(x, y, ":", lw=2, label=f"dissertação «{nm}»")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("fração molar de CO₂ (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "nb_5_5_fig36")
    print(
        f"Figura 36: eixos {figs[36]['axes']['y'].get('label')} / {figs[36]['axes'].get('y2', {}).get('label') if isinstance(figs[36]['axes'].get('y2'), dict) else ''}"
    )

# %% [markdown]
# ## Figura 34 — o desfasamento do holdup é termodinâmico (C7+ da Tabela 20)
#
# A curva «Modelo no-slip» da dissertação é 0,85× a nossa λ_L nos três troços, e a mecanicista
# 0,82× no poço. λ_L = Q_L/(Q_L + Q_G) só depende do flash (w_G, ρ_L, ρ_V), e o líquido é 66 % C7+
# em massa (MW 560, ω 1,32, ρ de referência 950 kg/m³ na Tabela 20, que não usamos). Abaixo, a razão
# por troço e o que seria preciso mudar no C7+ (ou na massa específica do líquido) para obter a
# curva da dissertação (issue 004, «Achados nas Figuras 34 e 40»).
# %%
if "noslip" in runs:
    import copy

    from cmfsim.thermo import ComponentSet

    r = runs["noslip"]
    xs34, ys34 = series_of(34, "Modelo no-slip")
    rows = []
    for seg, (a, b) in (("poço", (100, 1450)), ("linha", (1600, 7400)), ("riser", (7550, 7780))):
        xx = np.linspace(a, b, 12)
        yd = np.interp(xx, xs34, ys34)
        ours = np.interp(xx, r.x, r["H_L"])
        rows.append((seg, f"{yd.mean():.3f}", f"{ours.mean():.3f}", f"{np.mean(yd / ours):.3f}"))
    table(rows, ["troço", "λ_L dissertação", "λ_L cmfsim", "razão"])
    comp = components_db()
    fl = fluids_db()["table19_offshore"]
    names = list(fl)
    zz = np.array([fl[k] for k in names])
    zz = zz / zz.sum()
    idx = [int(np.argmin(np.abs(r.x - x0))) for x0 in (200.0, 4000.0, 7000.0)]
    T3, P3 = r.T[idx], r.P[idx]

    def lam_of(recs, kij=None):
        cs_ = ComponentSet.from_records([dict(id=k, **recs[k]) for k in names], kij=kij)
        st_ = phase_state(T3, P3, zz, cs_)
        return (1 - st_.w_G) / st_.rho_L / ((1 - st_.w_G) / st_.rho_L + st_.w_G / st_.rho_V), st_

    from cmfsim.thermo import phase_state

    base, st0 = lam_of(comp)
    yd3 = np.interp(r.x[idx], xs34, ys34)
    rows = [
        ("dissertação (Figura 34)", *[f"{v:.3f}" for v in yd3]),
        ("cmfsim (Tabela 20, k_ij = 0)", *[f"{v:.3f}" for v in base]),
    ]
    for key, fac in (("Pc", 1.25), ("omega", 0.8), ("Tc", 0.95)):
        rec = copy.deepcopy(comp)
        rec["C7plus"][key] = comp["C7plus"][key] * fac
        rows.append((f"C7+ {key} ×{fac}", *[f"{v:.3f}" for v in lam_of(rec)[0]]))
    n = len(names)
    i7 = names.index("C7plus")
    kij = np.zeros((n, n))
    kij[i7, :] = 0.1
    kij[:, i7] = 0.1
    kij[i7, i7] = 0.0
    rows.append(("k_ij(C7+, outros) = 0,1", *[f"{v:.3f}" for v in lam_of(comp, kij)[0]]))
    rho_need = (1 - st0.w_G) / (yd3 / (1 - yd3) * st0.w_G / st0.rho_V)
    rows.append(("ρ_L necessária com o nosso flash [kg/m³]", *[f"{v:.0f}" for v in rho_need]))
    rows.append(("ρ_L da PR [kg/m³]", *[f"{v:.0f}" for v in st0.rho_L]))
    table(rows, ["hipótese", f"x = {r.x[idx[0]]:.0f} m", f"{r.x[idx[1]]:.0f} m", f"{r.x[idx[2]]:.0f} m"])
