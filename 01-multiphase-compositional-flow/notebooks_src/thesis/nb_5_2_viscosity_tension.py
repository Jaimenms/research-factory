# %% [markdown]
# # §5.2 — Viscosidades e tensão superficial (Figuras 19–21, Tabela 12)
#
# Correlações de Pedersen (vapor), Allan & Teja (líquido) e Brock & Bird (tensão) nos 159 pontos
# da Figura 13, comparadas por distribuição (Q–Q) com os dois eixos das paridades digitalizadas:
# o HYSYS (eixo y, a referência) e a implementação da dissertação (eixo x). Não há correspondência
# ponto a ponto, e os marcadores sobrepostos fundem-se na digitalização (o número lido fica no
# título de cada painel). Não simula.
# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np

from dissertation.notebook_utils import figure, make_fluid, plt, qq_parity, save, table

from cmfsim.thermo import FlashOptions, phase_state
from cmfsim.thermo.viscosity import allan_teja_viscosity

cs, z = make_fluid("table6_validation")
f13 = figure(13)
pts_T, pts_P = f13["series"]["Pontos analisados (P,T)"]
# o eixo de temperatura da Figura 13 está em kelvin (os pontos vão de ~130 a ~530 K); a versão
# anterior deste notebook somava 273,15 e punha 155 dos 159 pontos fora do envelope
T_is_kelvin = pts_T.max() > 400.0
T = pts_T if T_is_kelvin else pts_T + 273.15
P = pts_P * 1e5
st = phase_state(T, P, z, cs, FlashOptions(stability_test=True))
two = st.two_phase > 0
print(f"{T.size} pontos, {int(two.sum())} bifásicos; T de {T.min():.0f} a {T.max():.0f} K")
table(
    [
        (
            "viscosidade do vapor (Pa s)",
            f"{np.median(st.mu_V[two]):.2e}",
            f"{st.mu_V[two].min():.2e}",
            f"{st.mu_V[two].max():.2e}",
        ),
        (
            "viscosidade do líquido (Pa s)",
            f"{np.median(st.mu_L[two]):.2e}",
            f"{st.mu_L[two].min():.2e}",
            f"{st.mu_L[two].max():.2e}",
        ),
        (
            "tensão superficial (N/m)",
            f"{np.median(st.sigma[two]):.4f}",
            f"{st.sigma[two].min():.4f}",
            f"{st.sigma[two].max():.4f}",
        ),
    ],
    ["propriedade", "mediana", "mín", "máx"],
)

# %% [markdown]
# ## Figuras 19–21 por quantis (eixos das figuras em Pa·s e N/m, sem reescala)
# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ratios = []
for ax, (n, mine, lab) in zip(axes, ((19, st.mu_V, "μ_V"), (20, st.mu_L, "μ_L"), (21, st.sigma, "σ"))):
    rh, rd = qq_parity(ax, figure(n), mine[two], int(two.sum()))
    ratios.append((f"Figura {n} ({lab})", f"{rh:.3f}", f"{rd:.3f}"))
axes[0].legend(fontsize=7)
fig.tight_layout()
save(fig, "nb_5_2_fig19_21_qq")
table(ratios, ["figura", "mediana cmfsim / HYSYS", "mediana cmfsim / dissertação"])
print(
    "Tabela 12 (dissertação × HYSYS): R² 0,941 (vapor), 0,964 (líquido), 0,863 (tensão); sem os valores emparelhados do HYSYS não há R² equivalente aqui."
)

# %% [markdown]
# ## Viscosidade do líquido: a regra de mistura de Allan & Teja (issue 008)
#
# A nossa μ_L fica 3–4× abaixo do HYSYS e da implementação da dissertação, que concordam entre si
# (medianas 0,70 e 0,54 mPa·s contra 0,18). O nC8 puro a 300 K dá 0,50 mPa·s (certo), logo a
# diferença está na mistura: a dissertação não define o número de carbonos efetivo da mistura
# (eq. 141, issue 001) e usamos a média molar. Alternativas testadas nos mesmos 159 estados:
# %%
f20 = figure(20)
_, (xd20, yd20) = next(iter(f20["series"].items()))
x_L = st.x[two]
MW = np.asarray(cs.MW)
Nc_i = np.asarray(cs.Nc_eff)
w_L = x_L * MW / (x_L * MW).sum(axis=1, keepdims=True)


def allan_teja_Nc(T, Nc):
    A = 145.73 + 99.01 * Nc + 0.83 * Nc**2 - 0.125 * Nc**3
    B = 30.48 + 34.01 * Nc - 1.23 * Nc**2 + 0.017 * Nc**3
    C = -3.07 - 1.99 * Nc
    return np.exp(A * (1.0 / (T + C) - 1.0 / B)) * 1e-3


def q5(a):
    return [f"{v*1e3:.3f}" for v in np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95])]


rows = [
    ("HYSYS (eixo y da Figura 20)", *q5(yd20)),
    ("dissertação (eixo x)", *q5(xd20)),
    ("cmfsim: Nc médio molar (atual)", *q5(allan_teja_Nc(T[two], x_L @ Nc_i))),
    ("Nc médio mássico", *q5(allan_teja_Nc(T[two], w_L @ Nc_i))),
    (
        "ln μ médio molar dos puros",
        *q5(np.exp(np.sum(x_L * np.log(np.stack([allan_teja_Nc(T[two], n) for n in Nc_i], 1)), 1))),
    ),
]
table(rows, ["μ_L [mPa s]", "q05", "q25", "q50", "q75", "q95"])
i8 = list(cs.names).index("nC8")
e8 = np.zeros((1, cs.n))
e8[0, i8] = 1.0
print(f"controlo: nC8 puro a 300 K = {1e3*allan_teja_viscosity(np.array([300.0]), e8, cs)[0]:.3f} mPa s (medido 0,51)")
# %% [markdown]
# Nenhuma regra de mistura simples fecha a diferença; a forma impressa de (141) e os Nc efetivos dos
# não alcanos (CO₂, N₂, H₂S: hipótese Nc = 1 na issue 001) ficam como causas a investigar. O
# Apêndice A horizontal, sensível a μ_L, é o lugar onde a diferença pesa (issues 005 e 008).
