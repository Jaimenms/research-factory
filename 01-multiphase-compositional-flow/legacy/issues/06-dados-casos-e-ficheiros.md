# L06 · Dados de entrada dos casos e ficheiros do repositório original

| | |
|---|---|
| família | entradas dos casos (scripts em `tests/`), ficheiros de dados e resultados, ficheiros em falta |
| código | [`tests/`](../masters_code/tests), [`data/`](../masters_code/data), raiz do repositório original |
| verificado | leitura e comparação com as tabelas transcritas em `dissertation/data/` |
| responde a | rugosidade e fluido do Apêndice A (issue 005); comprimento do riser; o que pedir ao autor |

## L06.1 Entradas por caso, tal como os scripts as passam

| caso | script | o que difere do texto ou do cmfsim |
|---|---|---|
| Apêndice A (§5.4) | [`validation_hydrodynamic_model.py`](../masters_code/tests/validation_hydrodynamic_model.py) | rugosidade **2e-4 m** (cmfsim: 4,5e-5); C1/C20pseud **6/94 % em massa** (≈ 69/31 % molar; Tabela 17: 70/30); 313 K; 100 m; 3 nós; U = 0 |
| §5.5 offshore | [`result_base_case_model.py`](../masters_code/tests/result_base_case_model.py), [`result_base_case_slip_model.py`](../masters_code/tests/result_base_case_slip_model.py) | versão final com fluido N2/CO2/C1/nC8 (e U = 500 W/(m² K) no poço do script com escorregamento): são scripts de teste, não os da §5.5; geometria da Tabela 21 com riser de **250 m** |
| §5.6 gasoduto | [`validation_condensate_pipeline.py`](../masters_code/tests/validation_condensate_pipeline.py) | **D_ext passado como espessura** (L01.2); C1–nC5 em massa [0,5, 0,2, 0,15, 0,1, 0,05] × 15 kg/s (= Tabela 23 molar); T_amb 298 K; 70 bar |
| Apêndice B / §5.7 | [`test_generate_samples_sim.py`](../masters_code/tests/test_generate_samples_sim.py) | fluido da Tabela 6 em **frações mássicas** [0,003, 0,0059, 0,0006, 0,0904, 0,007, 0,0171, 0,0251, 0,0271, 0,0301, 0,0502, 0,7433] (N2, CO2, H2S, C1–nC8); RK45; riser 250 m; U no D_ext (L01.3) |
| §5.1–5.2 propriedades | [`validation_thermo_model.py`](../masters_code/tests/validation_thermo_model.py) | fluido C1–nC8 sem inertes (L03.4); grelha 20 × 20 |

**Apêndice A (issue 005).** A rugosidade de 2e-4 m pesa: com 4,5e-5 m o próprio código original desce
para 0,75× a dissertação (L04.5). O cmfsim usa 4,5e-5 m porque a Tabela 15 a omite. A diferença de
composição (69/31 contra 70/30 molar) é desprezável.

## L06.2 Ficheiros de dados e de resultados no repositório original

| ficheiro | conteúdo | uso possível |
|---|---|---|
| [`data/pure_comp_prop.csv`](../masters_code/data/pure_comp_prop.csv) | MW, ω, Tc, Pc, Zc, Vc, Tb, H_f, dipolo; inclui `C20pseud` (= C7+ da Tabela 20) e `C7pseud` (MW 274, não usado) | conferido contra `dissertation/data/components.json` |
| [`data/ideal_gas_prop_hysys.csv`](../masters_code/data/ideal_gas_prop_hysys.csv) | polinómios de entalpia de gás ideal do HYSYS (kJ/kg, A–F); linhas vazias para H2S, C7pseud e outros | Cp do C7+ (L02.4) |
| [`data/ideal_gas_prop.csv`](../masters_code/data/ideal_gas_prop.csv) | Cp de Smith (Tabela 8); o C20pseud tem os coeficientes do nC8 | não entra na entalpia |
| [`data/validation_hysys/`](../masters_code/data/validation_hysys) | binário C2/nC6 e puros C2, nC6 no HYSYS (PR e COSTALD): ρ, h, μ, composições, Z | validação adicional da termodinâmica |
| [`pt_values.csv`](../masters_code/pt_values.csv) | os 159 estados (P, T) das Figuras 13–21 | substitui a leitura da Figura 13 |
| [`well_results_rna.csv`](../masters_code/well_results_rna.csv), [`flow_results_rna.csv`](../masters_code/flow_results_rna.csv), [`riser_results_rna.csv`](../masters_code/riser_results_rna.csv) | 299 amostras por trecho (P, T, ṁ → P, T de saída) em precisão completa | o Apêndice B sem arredondamento a 0,1 |
| [`well_samples_optim.csv`](../masters_code/well_samples_optim.csv), `flow_…`, `riser_…` | as entradas das amostras (hipercubo latino) | idem |
| [`props_hysys.xlsx`](../masters_code/props_hysys.xlsx), [`all_system_data.csv`](../masters_code/all_system_data.csv) | 421 casos de um estudo de casos no HYSYS: P de entrada 150–160 bar (10 níveis), ṁ 1–26,8 kg/s (9), T de entrada 315–350 K (5) → P e T de saída (0,4–64,8 bar; 283–317 K) | comparação do sistema completo contra o HYSYS (volume I); fluido e geometria a confirmar com o autor |
| [`results_flowline.csv`](../masters_code/results_flowline.csv) e variantes | 422 casos da linha (P, T, ṁ → saída) | idem, só a linha |
| [`simulation_result.csv`](../masters_code/simulation_result.csv) | perfil de 300 pontos a 170 bar, 80 °C e 20 kg/s (poço + linha + riser de 250 m), com propriedades e padrões | perfil físico do §5.7 (Figuras 57–58): saída a 59,2 bar e 45,2 °C (a dissertação mostra 55,2 bar e 45,4 °C) |
| [`validation_pressure_loss_results_with_slip_filtered.csv`](../masters_code/validation_pressure_loss_results_with_slip_filtered.csv) | a coluna «Dissertação» do Apêndice A | reproduzida (L04.5) |
| `*.pt` | redes PyTorch: propriedades (L03.5) e modelos substitutos por trecho | registo; o volume II não compara com eles |

## L06.3 Ficheiros que os scripts leem e que não estão no repositório

Estão todos em caminhos locais do autor (`C:\Users\Gusta\OneDrive\…`):

| ficheiro | script | o que daria |
|---|---|---|
| `hydro_results_hysys_new.csv` (e `_filtered`) | `validation_hydrodynamic_model.py` | as entradas e os ΔP do HYSYS do Apêndice A (já transcritos do PDF) |
| `hysys_all_props_values.csv` | `tests/auxiliar/hysys_values.py` → `validation_thermo_model.py` | **as propriedades do HYSYS nos 400 estados da grelha**: ρ, h, μ, σ e fração vaporizada; resolveria a comparação das Figuras 14–21 ponto a ponto, sem digitalização |
| `hysys_phase_fraction.csv` | `hysys_values.py` | fração vaporizada do HYSYS na grelha |
| `phase_envelope_bubble.csv`, `phase_envelope_dew.csv` | `plot_test.py` | o envelope da Figura 13 (o script desenha os 159 pontos com T − 10 K) |
| `Propriedades HYSYS.csv`, `nascimento_comp.csv` | `app/rna/run_RNA.py` | dados de treino das redes de propriedades |

**Pedido ao autor:** os três primeiros ficheiros, sobretudo `hysys_all_props_values.csv`, e a
configuração do estudo de casos de `props_hysys.xlsx` (fluido, trechos, modelo de escoamento do HYSYS).

## L06.4 Para o cmfsim

- **Apêndice B:** os ficheiros `*_results_rna.csv` podem substituir a extração do PDF em
  `dissertation/extract.py` (precisão completa). Fica para decisão do dono, porque o volume II já não
  compara com o autor.
- **Estados das Figuras 13–21:** `pt_values.csv` pode substituir os 159 estados digitalizados nos
  notebooks 5.1 e 5.2.
- **Estudo de casos do HYSYS:** `props_hysys.xlsx` daria uma comparação do sistema completo com o
  HYSYS em 421 casos. Falta saber o fluido e a configuração.
