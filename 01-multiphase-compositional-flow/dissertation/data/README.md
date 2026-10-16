# Dados da dissertação

Ponte com o texto da dissertação (`references/tese-gustavo-carvalho.pdf`), fora do simulador.

| ficheiro | conteúdo | origem |
|---|---|---|
| `appendix_a_horizontal.csv`, `appendix_a_vertical.csv` | os 200 casos de perda de carga (Tabelas 36–37): D (in), P_in (bar), m (kg/s) e ΔP (bar) da dissertação e dos modelos do HYSYS | gerados por `uv run python -m dissertation.extract` a partir do PDF (p. 168–172); 3 algarismos significativos |
| `appendix_b_{well,line,riser}.csv` | as 299 amostras do modelo físico por trecho (Tabelas 38–40) | `dissertation.extract` a partir do PDF (p. 173–192); arredondadas a 0,1 |
| `appendix_b_{well,line,riser}_full.csv` | as mesmas amostras em precisão completa, **em uso** | `dissertation.extract` a partir do código original (`legacy/masters_code/*_results_rna.csv`); iguais ao PDF a ±0,05 (teste `test_legacy_data_match_pdf`) |
| `appendix_c_{well,line,riser}.csv` | R² das 135 configurações da grelha por trecho (Tabelas 41–43) | `dissertation.extract` a partir do PDF |
| `hysys_states.csv` | os 159 estados (P [bar], T [K]) da comparação de propriedades com o HYSYS (§5.1–5.2), **em uso** | `dissertation.extract` a partir do código original (`legacy/masters_code/pt_values.csv`); a Figura 13 desenha-os a T − 10 K (`legacy/issues` L03.4) |
| `components.json` | propriedades dos componentes puros e do C7+ | Tabelas 7, 8, 9 e 20; hipóteses registadas em `_assumptions` |
| `fluids.json` | composições dos quatro fluidos | Tabelas 6, 17, 19, 23 |
| `cases_tables.json` | geometrias, condições de contorno, grelhas e resultados numéricos avulsos | Tabelas 13–16, 18, 21–26, 34–35 |

As figuras digitalizadas estão em `references/figures/`.
