# %% [markdown]
# # §5.8 — Desempenho computacional (Tabelas 34–35, Figura 59)
#
# A dissertação mede, no sistema offshore de §5.5 (Intel i5 2,5 GHz), o método integrador (7 s), o
# sistema não linear com 12, 30 e 78 nós (16, 29 e 140 s sem escorregamento; 37, 47 e 230 s com) e a
# rede neuronal (0,07 s), e nota que o flash é 90 % do custo. Aqui lêem-se os tempos gravados nas
# corridas do lote `batches/performance.json` (`runs/performance/`): marcha (numpy e JAX) e método
# 3.2 em JAX com 12, 30 e 78 nós, no caso offshore. Não simula.
# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np

from dissertation.notebook_utils import figure, load_runs, plt, save, table, tables

runs = load_runs("performance/*.npz")
print("corridas:", list(runs))
t34 = tables()["table34_timings_noslip_s"]
t35 = tables()["table35_timings_slip_s"]
rows = [
    (
        "marcha (integrador)",
        f"{t34['march']:.0f}",
        "—",
        f"{runs['perf_march_numpy'].info['cpu']:.1f}" if "perf_march_numpy" in runs else "—",
        f"{runs['perf_march_jax'].info['cpu']:.2f}" if "perf_march_jax" in runs else "—",
    )
]
for n in (12, 30, 78):
    r = runs.get(f"perf_fd_jax_n{n}")
    rows.append(
        (
            f"sistema não linear, {n} nós",
            f"{t34[f'fd_{n}']:.0f}",
            f"{t35[f'fd_{n}']:.0f}",
            "—",
            f"{r.info['cpu']:.2f} ({r.info['iterations']} Newton)" if r else "—",
        )
    )
rows.append(("rede neuronal", f"{t34['neural_network']:.2f}", "—", "—", "adiado"))
table(
    rows,
    [
        "método",
        "dissertação sem escorregamento (s)",
        "dissertação com escorregamento (s)",
        "cmfsim numpy (s)",
        "cmfsim JAX (s)",
    ],
)
print(
    "Os tempos JAX incluem a compilação na primeira corrida de cada estrutura (ver issue 006); o teste dos 90 % do flash (p. 158) corresponde às medições de 0,7 ms por flash compilado."
)

# %%
ns = [n for n in (12, 30, 78) if f"perf_fd_jax_n{n}" in runs]
if ns:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([12, 30, 78], [t34["fd_12"], t34["fd_30"], t34["fd_78"]], "o--", label="dissertação, sem escorregamento")
    ax.plot([12, 30, 78], [t35["fd_12"], t35["fd_30"], t35["fd_78"]], "s--", label="dissertação, com escorregamento")
    ax.plot(
        ns,
        [runs[f"perf_fd_jax_n{n}"].info["cpu"] for n in ns],
        "^-",
        label="cmfsim JAX (com escorregamento, inclui compilação)",
    )
    f59 = figure(59)
    for nm, (x, y) in f59["series"].items():
        ax.plot(x, y, ":", lw=1, color="gray")
    ax.set_xlabel("número de nós")
    ax.set_ylabel("tempo (s)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Figura 59 — tempo de execução contra número de nós")
    save(fig, "nb_5_8_fig59")
