# Dados da dissertação

Ponte com o texto da dissertação (`references/tese-gustavo-carvalho.pdf`), fora do simulador.

| ficheiro | conteúdo | origem |
|---|---|---|
| `appendix_a_horizontal.csv`, `appendix_a_vertical.csv` | os 200 casos de perda de carga (Tabelas 36–37): D (in), P_in (bar), m (kg/s) e ΔP (bar) da dissertação e dos modelos do HYSYS | gerados por `uv run python -m dissertation.extract` a partir do PDF (p. 168–172); 3 algarismos significativos |
| `components.json` | propriedades dos componentes puros e do C7+ | Tabelas 7, 8, 9 e 20; hipóteses registadas em `_assumptions` |
| `fluids.json` | composições dos quatro fluidos | Tabelas 6, 17, 19, 23 |
| `cases_tables.json` | geometrias, condições de contorno, grelhas e resultados numéricos avulsos | Tabelas 13–16, 18, 21–26, 34–35 |

As figuras digitalizadas estão em `references/figures/`.
