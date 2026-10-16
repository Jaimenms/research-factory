# %% [markdown]
# # §5.6 — Gasoduto com formação de condensado (Figuras 38–40, Tabelas 23–26)
#
# Seis casos (T_in 60/80 °C × U 1/3/6 W/m²K na Tabela 25) do gasoduto de 8 km e 12 in com o gás rico
# da Tabela 23, modelo de não escorregamento por marcha em x. Os casos usam **U_eff = 3,42 × U da
# Tabela 25**: no troço monofásico os perfis da Figura 39 são exponenciais puras, com a mesma razão
# a/U nos seis casos e 3,42× a do balanço de energia com os U impressos, enquanto a §5.5, com o mesmo
# modelo, não precisa de fator nenhum. O dono decidiu tratar como erro de apresentação de U
# (issue 004). Corridas do lote `batches/dissertation.json` (`runs/dissertation/condensate_5_6_case*.npz`);
# Figuras 38–40 digitalizadas sobrepostas. Não simula.
# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import curve_fit

from dissertation.notebook_utils import RUNS, figure, load_runs, make_fluid, plt, save, table, tables

from cmfsim.thermo import FlashOptions, phase_state

figs = {n: figure(n) for n in (38, 39, 40)}
runs = load_runs("dissertation/condensate_5_6_case*.npz")
print("corridas:", list(runs))
t25 = tables()["table25_condensate_cases"]
T_AMB = 25.0


def at(fig, name, x):
    xs, ys = fig["series"][name]
    return float(np.interp(x, xs, ys)) if xs.min() <= x <= xs.max() else np.nan


def x_cond(xs, w):
    m = w < 0.9995
    return float(xs[m][0]) if np.any(m) else np.nan


def U_eff(r):
    return float(r.case_dict["pipe"]["profile"][0]["U"])


def beta_molar(r):
    """Fração vaporizada molar a partir da mássica e das composições das fases."""
    MW = np.array([c["MW"] for c in r.case_dict["fluid"]["components"]])
    w = r["w_G"]
    nG = w / (r["y_comp"] * MW).sum(1)
    nL = (1.0 - w) / (r["x_comp"] * MW).sum(1)
    return nG / (nG + nL)


# %% [markdown]
# ## Tabelas 23–25 (entradas) e saída de cada caso
#
# `U tab` é o valor impresso na Tabela 25; `U eff` o usado na corrida. A «fração vaporizada» da Figura 40 é
# a **molar** (β): no troço bifásico o cmfsim segue a figura a 0,004 em base molar, contra 0,02 em base
# mássica e 0,015 em volumétrica (questão B6 de docs/05). A série 5 das Figuras 39–40 (pontilhada,
# segmentada por tonalidade) está mal digitalizada e os seus números têm pouco valor.
# %%
rows = []
for c in t25:
    k = c["case"]
    r = runs.get(f"condensate_5_6_case{k}")
    s = str(k)
    xd, yd = figs[40]["series"][s]
    xc_d = x_cond(xd, yd)
    xc_r = x_cond(r.x, r["w_G"]) if r else np.nan
    rows.append(
        (
            k,
            f"{c['T_in_C']:.0f}",
            f"{c['U']:.0f}",
            f"{U_eff(r):.2f}" if r else "—",
            f"{at(figs[39], s, 7950):.1f}",
            f"{r.T[-1] - 273.15:.1f}" if r else "—",
            f"{at(figs[38], s, 7950):.2f}",
            f"{r.P[-1] / 1e5:.2f}" if r else "—",
            f"{at(figs[40], s, 7950):.4f}",
            f"{beta_molar(r)[-1]:.4f}" if r else "—",
            f"{r['w_G'][-1]:.4f}" if r else "—",
            f"{xc_d:.0f}" if np.isfinite(xc_d) else "—",
            f"{xc_r:.0f}" if np.isfinite(xc_r) else "—",
        )
    )
table(
    rows,
    [
        "caso",
        "T_in",
        "U tab",
        "U eff",
        "T_out tese",
        "T_out cmfsim",
        "P_out tese",
        "P_out cmfsim",
        "β_out tese (Fig. 40)",
        "β_out cmfsim (molar)",
        "wG_out cmfsim (mássica)",
        "x_cond tese",
        "x_cond cmfsim",
    ],
)


# %% [markdown]
# ## Ajuste exponencial no troço monofásico: T = T_amb + (T_in − T_amb) exp(−a x)
#
# Com cp constante, −U π D (T − T_amb) = m cp dT/dx dá a = U π D / (m cp). Ajusta-se `a` às curvas
# digitalizadas (T > 34,5 °C, antes da condensação) e às corridas; a razão tese/cmfsim mede o fator que
# falta no termo térmico. Com U_eff a razão é ≈ 1; com os U impressos era 3,1–3,3 nos seis casos
# (issue 004). `cp_eff` é o cp constante que reproduz a curva da tese com o U impresso.
# %%
def expo(x, a, T_in):
    return T_AMB + (T_in - T_AMB) * np.exp(-a * x)


def fit_a(x, T, Tmin=34.5):
    o = np.argsort(x)
    x, T = np.asarray(x)[o], np.asarray(T)[o]
    k = (T > Tmin) & (x > 50.0)
    (a, T_in), _ = curve_fit(expo, x[k], T[k], p0=[1e-4, T[k][0]])
    rms = float(np.sqrt(np.mean((expo(x[k], a, T_in) - T[k]) ** 2)))
    return a, rms


rows = []
for c in t25:
    k = c["case"]
    r = runs.get(f"condensate_5_6_case{k}")
    xt, yt = figs[39]["series"][str(k)]
    a_t, rms_t = fit_a(xt, yt)
    cp_eff = c["U"] * np.pi * 0.3048 / (15.0 * a_t)
    if r:
        a_r, rms_r = fit_a(r.x, r.T - 273.15)
        o = np.argsort(xt)
        kk = (r.x >= xt.min()) & (r.x <= xt.max())
        dev = np.interp(r.x[kk], xt[o], yt[o]) - (r.T[kk] - 273.15)
        rows.append(
            (
                k,
                f"{c['U']:.0f}",
                f"{a_t:.3e}",
                f"{rms_t:.2f}",
                f"{a_t / c['U']:.2e}",
                f"{cp_eff:.0f}",
                f"{a_r:.3e}",
                f"{a_t / a_r:.3f}",
                f"{np.sqrt(np.mean(dev**2)):.2f}",
                f"{np.abs(dev).max():.2f}",
            )
        )
table(
    rows,
    [
        "caso",
        "U tab",
        "a tese [1/m]",
        "rms ajuste tese",
        "a/U tab",
        "cp_eff tese [J/kgK]",
        "a cmfsim (U eff)",
        "a tese / a cmfsim",
        "rms T perfil inteiro",
        "max |ΔT|",
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
        axes[2].plot(r.x, beta_molar(r), color=col)
        axes[2].plot(r.x, r["w_G"], "--", color=col, lw=0.8)
    for ax, n in zip(axes, (38, 39, 40)):
        if s in figs[n]["series"]:
            ax.plot(
                *figs[n]["series"][s], ":", color=col, lw=2, label=f"caso {k} dissertação" if ax is axes[0] else None
            )
axes[0].set_ylabel("P (bar)")
axes[1].set_ylabel("T (°C)")
axes[2].set_ylabel("fração vaporizada molar (tracejado: mássica)")
axes[2].set_xlabel("x (m)")
axes[0].legend(fontsize=7, ncol=3)
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle("Figuras 38–40 — cmfsim com U_eff = 3,42 × U (linhas) contra a dissertação (pontilhado)")
save(fig, "nb_5_6_fig38_40")

# %% [markdown]
# ## Tabela 26 — composições na saída do caso 2 (título da tabela diz caso 1; pela Figura 40 é o caso 2)
#
# Duas colunas do cmfsim: a saída da corrida (U_eff) e o flash às condições de saída da dissertação
# (P e T das Figuras 38–39). O balanço material das composições impressas implica 4–7 % molar de
# líquido (β ≈ 0,93–0,96 por componente), incompatível com os 1,8 % molares da Figura 40 e com os 1,4 %
# do cmfsim: a tabela não serve de prova fina do flash. O líquido do cmfsim é mais pesado (31 % de C1
# contra 46 %).
# %%
cs, z = make_fluid("table23_rich_gas")
t26 = tables()["table26_outlet_composition_case2"]
r = runs.get("condensate_5_6_case2")
names = list(t26["global"])
rows = []
if r is not None:
    P_d, T_d = at(figs[38], "2", 7950) * 1e5, at(figs[39], "2", 7950) + 273.15
    st_d = phase_state(np.array([T_d]), np.array([P_d]), z, cs, FlashOptions(stability_test=True))
    x_out, y_out = r["x_comp"][-1], r["y_comp"][-1]
    for i, nm in enumerate(names):
        rows.append(
            (
                nm,
                f"{t26['global'][nm]:.3f}",
                f"{t26['liquid'][nm]:.3f}",
                f"{x_out[i]:.3f}",
                f"{st_d.x[0, i]:.3f}",
                f"{t26['vapor'][nm]:.3f}",
                f"{y_out[i]:.3f}",
                f"{st_d.y[0, i]:.3f}",
            )
        )
    table(
        rows,
        [
            "componente",
            "global",
            "líquido tese",
            "líquido cmfsim (saída)",
            "líquido cmfsim (P, T tese)",
            "vapor tese",
            "vapor cmfsim (saída)",
            "vapor cmfsim (P, T tese)",
        ],
    )
    beta_tab = [
        (t26["liquid"][nm] - t26["global"][nm]) / (t26["liquid"][nm] - t26["vapor"][nm])
        for nm in names
        if abs(t26["liquid"][nm] - t26["vapor"][nm]) > 1.0
    ]
    print(
        f"saída da corrida: T = {r.T[-1]-273.15:.1f} °C, P = {r.P[-1]/1e5:.2f} bar, w_G = {r['w_G'][-1]:.4f}; "
        f"flash às condições da tese (P = {P_d/1e5:.2f} bar, T = {T_d-273.15:.1f} °C): beta = {st_d.beta[0]:.4f}; "
        f"beta molar implícito na Tabela 26 por componente: {', '.join(f'{b:.3f}' for b in beta_tab)}"
    )

# %% [markdown]
# ## Figura 40 — na dissertação o início da condensação é um salto
#
# No PDF as curvas caem na vertical de 1 para ≈ 0,986 em menos de 75 m e só depois descem
# suavemente (a digitalização é fiel, ver `references/figures/overlay/fig_40.png`). Um salto no
# orvalho é fisicamente impossível (β − 1 é contínuo) e o nosso início contínuo está validado
# contra a `thermo`. Abaixo: onde a dissertação salta, para que valor, o que o nosso flash dá nesse
# ponto, a temperatura local contra o orvalho da PR e a TPD (teste de estabilidade) nesse ponto.
# Substituições sucessivas sem polimento convergem em 14 iterações em todo o troço, e o orvalho
# pelo K de Wilson daria bifásico desde a entrada: nenhuma das duas explica o salto; a TPD ≈ −0,01
# no ponto do salto aponta para uma banda morta no teste de fase do código da dissertação.
# %%
from cmfsim.backend import NUMPY
from cmfsim.thermo.flash import stability_tpd

cs23, z23 = make_fluid("table23_rich_gas")


def dew_T(Pa):
    lo, hi = 273.15, 333.15
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        b = phase_state(np.array([mid]), np.array([Pa]), z23, cs23).beta[0]
        lo, hi = (mid, hi) if b < 1.0 else (lo, mid)
    return 0.5 * (lo + hi)


rows = []
for c in t25:
    k = c["case"]
    s = str(k)
    r = runs.get(f"condensate_5_6_case{k}")
    if not r or s not in figs[40]["series"] or k == 5:
        continue
    xd, yd = figs[40]["series"][s]
    ii = np.where(yd < 0.9995)[0]
    if ii.size == 0:
        rows.append((k, "—", "—", "—", "—", "—", "—", "—"))
        continue
    i = int(ii[0])
    j = i
    while j + 1 < len(xd) and (yd[j] - yd[j + 1]) / max(xd[j + 1] - xd[j], 1e-9) > 2e-5:
        j += 1
    bm = beta_molar(r)
    x_jump, b_end = xd[i], yd[j]
    T_j, P_j = np.interp(x_jump, r.x, r.T), np.interp(x_jump, r.x, r.P)
    tpd, _ = stability_tpd(np.array([T_j]), np.array([P_j]), z23[None, :], cs23, FlashOptions(), NUMPY)
    x_ours = r.x[bm < 0.9999][0] if np.any(bm < 0.9999) else np.nan
    rows.append(
        (
            k,
            f"{x_jump:.0f}–{xd[j]:.0f}",
            f"{b_end:.4f}",
            f"{np.interp(x_jump, r.x, bm):.4f}",
            f"{T_j - 273.15:.1f}",
            f"{dew_T(P_j) - 273.15:.1f}",
            f"{tpd[0]:+.1e}",
            f"{x_ours:.0f}",
        )
    )
table(
    rows,
    [
        "caso",
        "salto da tese [m]",
        "β depois do salto",
        "β cmfsim nesse x",
        "T no salto [°C]",
        "orvalho PR [°C]",
        "TPD no salto",
        "x_cond cmfsim [m]",
    ],
)
