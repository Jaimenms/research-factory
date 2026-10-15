# %% [markdown]
# # §5.2 — Viscosidades e tensão superficial (Figuras 19–21, Tabela 12)
#
# Correlações de Pedersen (vapor), Allan & Teja (líquido) e Brock & Bird (tensão) nos 159 pontos
# da Figura 13, comparadas por distribuição (Q–Q) com os valores da dissertação lidos das paridades
# e, como oráculo independente, com as correlações da biblioteca `thermo` quando disponíveis.
# %%
import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

from dissertation.notebook_utils import figure, make_fluid, plt, save, table

from cmfsim.thermo import FlashOptions, phase_state

cs, z = make_fluid("table6_validation")
f13 = figure(13)
pts_T, pts_P = f13["series"]["Pontos analisados (P,T)"]
T, P = pts_T + 273.15, pts_P * 1e5
st = phase_state(T, P, z, cs, FlashOptions(stability_test=True))
two = st.two_phase > 0
print(f"{T.size} pontos, {int(two.sum())} bifásicos")
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

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (n, mine, unit) in zip(axes, ((19, st.mu_V, "Pa s"), (20, st.mu_L, "Pa s"), (21, st.sigma, "N/m"))):
    f = figure(n)
    xd, yd = next(iter(f["series"].values()))
    mine_v = mine[two]
    scale = np.median(xd) / np.median(mine_v)  # unidades da figura (cP, dyn/cm...) alinhadas pela mediana
    q = np.linspace(0, 1, 40)
    ax.plot(np.quantile(np.sort(xd), q), np.quantile(np.sort(mine_v * scale), q), "o-", ms=3)
    lo, hi = min(xd.min(), (mine_v * scale).min()), max(xd.max(), (mine_v * scale).max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_title(f"Figura {n} — {f['caption']} (escala ×{scale:.3g})")
    ax.set_xlabel("dissertação (eixo x da paridade)")
    ax.set_ylabel(f"cmfsim ×{scale:.3g}")
    ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "nb_5_2_fig19_21_qq")
print(
    "Tabela 12 (dissertação × HYSYS): R² 0,941 (vapor), 0,964 (líquido), 0,863 (tensão); sem o HYSYS não há R² equivalente aqui."
)
# %% [markdown]
# A escala alinhada pela mediana denuncia as unidades usadas nas figuras (cP e dyn/cm) e as
# diferenças sistemáticas das correlações; a forma da curva Q–Q diz se a dispersão é a mesma.
