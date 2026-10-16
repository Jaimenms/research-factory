# %% [markdown]
# # §5.1 — Validação do modelo termodinâmico (Figuras 13–18, Tabelas 6–11)
#
# A dissertação compara a sua implementação de Peng–Robinson com o HYSYS em 159 pares (P, T) dentro
# do envelope de fases do fluido da Tabela 6. Não temos os valores do HYSYS (decisão A2), só as
# figuras digitalizadas: a Figura 13 dá os 159 pontos e o envelope; as Figuras 14–18 dão, no eixo
# x, os valores calculados pela dissertação (sem a correspondência ponto a ponto). O que se
# reproduz aqui: (1) o envelope de fases pelo nosso flash contra o digitalizado; (2) as propriedades
# nos 159 pontos contra o oráculo independente `thermo` (no lugar do HYSYS), com R² como nas
# Tabelas 10–11; (3) a distribuição dos nossos valores contra a dos valores da dissertação lidos
# das paridades (Q–Q).
#
# Cálculo termodinâmico instantâneo (não é simulação): o notebook avalia o flash diretamente.
# %%
import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

from dissertation.notebook_utils import figure, make_fluid, plt, qq_parity, save, table

from cmfsim.thermo import FlashOptions, phase_state

cs, z = make_fluid("table6_validation")
cs_exact = cs.with_exact_pr_constants()
f13 = figure(13)
pts_T, pts_P = f13["series"]["Pontos analisados (P,T)"]  # x = T, y = P (bar)
# o eixo de temperatura da Figura 13 está em kelvin (os pontos vão de ~130 a ~530): converte-se se preciso
T_is_kelvin = pts_T.max() > 400.0
T = pts_T if T_is_kelvin else pts_T + 273.15
P = pts_P * 1e5
print(
    f"{pts_T.size} pontos digitalizados; T de {T.min():.0f} a {T.max():.0f} K, P de {pts_P.min():.0f} a {pts_P.max():.0f} bar"
)
st = phase_state(T, P, z, cs_exact, FlashOptions(stability_test=True))
print(f"bifásicos no nosso flash: {int(st.two_phase.sum())} de {T.size}")


# %% [markdown]
# ## Figura 13 — envelope de fases
# Linhas de bolha e de orvalho por bisseção em P a cada T (flash com teste de estabilidade).
# %%
def two_phase_band(T_K, P_grid):
    """(P_orvalho, P_bolha) em bar à temperatura T_K: extremos da faixa bifásica num varrimento em P
    (vetorizado) refinados por bisseção; NaN se não houver faixa bifásica."""
    Tv = np.full(P_grid.size, T_K)
    two = phase_state(Tv, P_grid, z, cs_exact, FlashOptions(stability_test=True)).two_phase > 0
    if not two.any():
        return np.nan, np.nan
    idx = np.flatnonzero(two)

    def refine(p_in, p_out):
        lo, hi = p_in, p_out  # lo bifásico, hi monofásico
        for _ in range(25):
            mid = 0.5 * (lo + hi)
            b = (
                phase_state(np.array([T_K]), np.array([mid]), z, cs_exact, FlashOptions(stability_test=True)).two_phase[
                    0
                ]
                > 0
            )
            lo, hi = (mid, hi) if b else (lo, mid)
        return 0.5 * (lo + hi)

    i0, i1 = idx[0], idx[-1]
    P_dew = refine(P_grid[i0], P_grid[i0 - 1]) if i0 > 0 else P_grid[0]
    P_bub = refine(P_grid[i1], P_grid[i1 + 1]) if i1 < P_grid.size - 1 else P_grid[-1]
    return P_dew / 1e5, P_bub / 1e5


Tb, Pb = f13["series"]["Curva azul do envelope (bolha)"]
Td, Pd = f13["series"]["Curva laranja do envelope (orvalho)"]
if not T_is_kelvin:
    Tb, Td = Tb + 273.15, Td + 273.15
T_grid = np.linspace(min(Tb.min(), Td.min()) + 2, max(Tb.max(), Td.max()) - 2, 30)
P_grid = np.logspace(np.log10(0.5e5), np.log10(300e5), 120)
band = np.array([two_phase_band(t_, P_grid) for t_ in T_grid])
P_dew, P_bub = band[:, 0], band[:, 1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Tb, Pb, "b:", lw=2, label="bolha (HYSYS, digitalizada)")
ax.plot(Td, Pd, ":", color="orange", lw=2, label="orvalho (HYSYS, digitalizada)")
ax.plot(T_grid, P_bub, "b-", label="bolha (cmfsim)")
ax.plot(T_grid, P_dew, "-", color="orange", label="orvalho (cmfsim)")
ax.plot(T, pts_P, "k.", ms=4, label="159 pontos analisados")
ax.set_xlabel("T (K)")
ax.set_ylabel("P (bar)")
ax.set_title("Figura 13 — envelope de fases do fluido da Tabela 6")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "nb_5_1_fig13_envelope")
ok = np.isfinite(P_bub)
dev_b = np.abs(P_bub[ok] - np.interp(T_grid[ok], Tb, Pb))
dev_d = np.abs(P_dew[ok] - np.interp(T_grid[ok], Td, Pd))
print(
    f"linha de bolha: desvio mediano {np.median(dev_b):.2f} bar (máx {dev_b.max():.2f}); linha de orvalho: {np.median(dev_d):.2f} bar (máx {dev_d.max():.2f})"
)

# %% [markdown]
# ## Figuras 14–18 e Tabelas 10–11 — propriedades nos 159 pontos contra o oráculo `thermo`
# %%
try:
    from tests.test_thermo_reference import _flasher

    flasher = _flasher(cs_exact)
    ref = dict(rho_V=[], rho_L=[], h_V=[], h_L=[], beta=[])
    for Tk, Pk in zip(T, P):
        r = flasher.flash(T=float(Tk), P=float(Pk), zs=list(z))
        gas, liqs = getattr(r, "gas", None), getattr(r, "liquids", None) or []
        liq = liqs[0] if liqs else None
        ref["beta"].append(0.0 if r.VF is None else float(r.VF))
        ref["rho_V"].append(gas.rho_mass() if gas else np.nan)
        ref["rho_L"].append(liq.rho_mass() if liq else np.nan)
        ref["h_V"].append(np.nan)
        ref["h_L"].append(np.nan)
    ref = {k: np.array(v) for k, v in ref.items()}
    have_oracle = True
except Exception as e:  # noqa: BLE001
    print("oráculo thermo indisponível:", e)
    have_oracle = False


def r2(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    return 1.0 - np.sum((a - b) ** 2) / np.sum((b - b.mean()) ** 2), int(m.sum())


rows = []
if have_oracle:
    for nm, mine, lab in (
        ("rho_V", st.rho_V, "massa específica do vapor"),
        ("rho_L", st.rho_L, "massa específica do líquido"),
        ("beta", st.beta, "fração vaporizada"),
    ):
        m = st.two_phase > 0
        val, n = r2(mine[m], ref[nm][m])
        rows.append((lab, n, f"{val:.6f}"))
    table(rows, ["propriedade", "pontos bifásicos", "R² cmfsim × thermo"])
    print("(Tabelas 10–11 da dissertação, cmfsim × HYSYS: 0,997 / 0,992 / 0,978)")

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
# eixos das figuras em SI (kg/m³, J/kg, -): sem reescala
pairs = [(14, st.rho_V), (15, st.rho_L), (16, st.h_V), (17, st.h_L), (18, st.beta)]
ratios = []
for ax, (n, mine) in zip(axes.ravel(), pairs):
    rh, rd = qq_parity(ax, figure(n), mine, T.size)
    ratios.append((f"Figura {n}", f"{rh:.3f}", f"{rd:.3f}"))
axes.ravel()[0].legend(fontsize=7)
axes.ravel()[-1].axis("off")
fig.tight_layout()
save(fig, "nb_5_1_fig14_18_qq")
table(ratios, ["figura", "mediana cmfsim / HYSYS", "mediana cmfsim / dissertação"])
# %% [markdown]
# Leitura: a referência é o eixo HYSYS (y) das paridades; o eixo x é a implementação da
# dissertação. Os quantis alinham-se na diagonal quando a distribuição dos nossos valores nos 159
# pontos é a mesma que a da figura. Não há correspondência ponto a ponto, e os marcadores
# sobrepostos nas zonas densas fundem-se na digitalização (número de marcadores no título): a
# distribuição lida fica enviesada para os valores raros, o que exagera os desvios nas caudas
# (Figuras 14 e 18, com 60 e 72 marcadores). Só valores emparelhados do HYSYS (decisão A2) dariam
# R² como as Tabelas 10–11; o R² contra a `thermo` acima é a validação que temos.
