# L03 · Propriedades de transporte: viscosidades e tensão superficial

| | |
|---|---|
| família | viscosidade do líquido e do vapor, tensão superficial, validação de propriedades (§5.1–5.2) |
| código | [`viscosity_correlation.py`](../masters_code/app/thermodynamic_v2/equation_of_state/viscosity_correlation.py), [`surface_tension.py`](../masters_code/app/thermodynamic_v2/equation_of_state/surface_tension.py), [`surface_tension_correlation.py`](../masters_code/app/thermodynamic_v2/equation_of_state/surface_tension_correlation.py), [`equation_of_state.py`](../masters_code/app/thermodynamic_v2/equation_of_state/equation_of_state.py), [`hydrodynamic.py`](../masters_code/app/hydrodynamic/hydrodynamic.py), [`hydrodynamic_slip.py`](../masters_code/app/hydrodynamic/hydrodynamic_slip.py), [`tests/validation_thermo_model.py`](../masters_code/tests/validation_thermo_model.py) |
| verificado | corridas do código original (2026-10-16): Apêndice A ([`scripts/appendix_a.py`](../scripts/appendix_a.py)) e grelha das propriedades ([`scripts/property_grid.py`](../scripts/property_grid.py)) |
| responde a | «dificuldade de modelar a viscosidade em função dos estados»; Nc_eff do C7+ (issue 001, B3); Figura 20 com 24 marcadores (issue 008) |

## Resumo

O código tem duas camadas de viscosidade que não se falam:

- **Na equação de estado:** a viscosidade é constante, com um valor para «líquido» e outro para
  «gás» escolhidos por um limiar de massa específica (L03.1).
- **Nos modelos hidrodinâmicos:** é calculada pelo estado com Allan & Teja e Pedersen. Mas o número de
  carbonos de Allan & Teja vem do nome do componente, e o pseudocomponente C7+ conta zero (L03.2).

Com o fluido C1/C20pseud do Apêndice A, a viscosidade do líquido sai 0,0134 mPa·s, constante, uma
ordem de grandeza típica de gás. A dificuldade de modelar a viscosidade pelo estado está no código
como dois atalhos: a constante e o nome. Houve também uma tentativa de substituir as propriedades
por redes treinadas no HYSYS (L03.5).

## L03.1 Viscosidade e tensão superficial constantes na equação de estado

- **Viscosidade na equação de estado:** `viscosity = calculate_viscosity_simple(rho)`
  ([equation_of_state.py:167](../masters_code/app/thermodynamic_v2/equation_of_state/equation_of_state.py#L167))
  devolve 1,26e-3 Pa·s se **todas** as massas específicas do vetor passarem de 150 kg/m³, e 1,195e-5
  Pa·s se alguma não passar
  ([viscosity_correlation.py:8-12](../masters_code/app/thermodynamic_v2/equation_of_state/viscosity_correlation.py#L8-L12)).
- **Efeito do teste vetorial:** a decisão vale para o vetor inteiro. Um gás denso (ρ_G > 150 kg/m³,
  comum acima de ~150 bar) recebe a viscosidade do líquido quando os três pontos das diferenças finitas
  passam do limiar.
- **Onde se usa:** é esta viscosidade que entra no Reynolds da mistura do modelo no-slip
  ([hydrodynamic.py:52](../masters_code/app/hydrodynamic/hydrodynamic.py#L52), [:106](../masters_code/app/hydrodynamic/hydrodynamic.py#L106)).
- **Tensão superficial na equação de estado:** é constante, 0,03 N/m
  ([`surface_tension_correlation.py`](../masters_code/app/thermodynamic_v2/equation_of_state/surface_tension_correlation.py)).
  Os modelos hidrodinâmicos não a usam; recalculam-na por Brock & Bird (L03.3).

## L03.2 Allan & Teja: número de carbonos lido do nome do componente

```python
for index, comp in enumerate(props.comp):
    if "n" in comp:
        n += composition[index]*float(comp[-1])
```

([viscosity_correlation.py:483-484](../masters_code/app/thermodynamic_v2/equation_of_state/viscosity_correlation.py#L483-L484))

- **Que nomes contam:** só contam os nomes com a letra «n», e o número de carbonos é o **último
  carácter** do nome. nC4–nC8 contam 4–8.
- **Contam zero:** C1, C2, C3, iC4, CO2, N2, H2S e o pseudocomponente `C20pseud`, que é o C7+ da
  Tabela 20.
- **Nomes com dois dígitos:** um nome como nC10 contaria 0.
- **Resultado:**
  - no Apêndice A (C1 + C20pseud), Nc_m = 0 e A, B, C de (138–140) caem nos valores do «n-alcano de
    zero carbonos»;
  - a 313 K, μ_L = exp[145,73 (1/309,9 − 1/30,48)] × 1e-3 = 1,34e-5 Pa·s (0,0134 mPa·s);
  - o valor medido é o mesmo nos 100 casos horizontais do Apêndice A.
- **Fluido da Tabela 19 (§5.5):** só nC4–nC6 contam, e o C7+ (66 % da massa do líquido) não entra.

O cmfsim usa Nc_eff = 20 para o C7+ (issue 001), com μ_L de 0,7 a 3 mPa·s nos mesmos casos. No
Apêndice A, só esta troca no código original leva o ΔP horizontal a 1,9× a dissertação. Sem o atrito
4× (L04.1), a viscosidade de gás era o que mantinha a dissertação perto do HYSYS.

## L03.3 Viscosidade do vapor (Pedersen/Ali) e tensão superficial (Brock & Bird)

**Viscosidade do vapor** (`calculate_viscosity_ali`): estados correspondentes de Pedersen com
metano de referência.

- **Referência:** massa específica do metano por PR (o BWR existe mas está comentado),
  ρ_c(CH₄) = 162,8 kg/m³ e tangente hiperbólica com T − 90,4 K. O cmfsim usa 162,84 kg/m³ e T − 91 K
  (issue 001); a diferença é de arredondamento.
- **Constantes:** s1 = 1,6969859271, como Hanley et al. (1975) e como o cmfsim; a Tabela 2 imprime
  1,696986e5.
- **Unidades:** o resultado vem em µP × 1e-6 = P, e `vapour_viscosity *= 0.1`
  ([hydrodynamic_slip.py:338](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L338)) converte
  para Pa·s. Está certo.
- **Valores:** no Apêndice A, μ_G = 12–17 µPa·s, igual ao cmfsim (12,0–16,9 µPa·s).

**Tensão superficial** (`surface_tension.py`): Brock & Bird por componente.

- **Unidades:** Pc em bar, resultado em dyn/cm, depois ×1e-3 para N/m
  ([:342](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L342)).
- **Mistura:** média molar na fase líquida.
- **Componentes supercríticos:** dão (1 − Tr)^(11/9) = NaN e as linhas com NaN são retiradas da soma,
  o que equivale a contribuírem com zero. É a mesma regra do cmfsim.
- **Valores:** no Apêndice A, σ = 20–30 mN/m, contra 15–29 mN/m no cmfsim.

## L03.4 Validação de propriedades (§5.1–5.2, Figuras 13–21)

[`tests/validation_thermo_model.py`](../masters_code/tests/validation_thermo_model.py) mostra como as
figuras foram feitas:

- **Fluido:** C1, C2, C3, nC4–nC8 com frações mássicas
  [0,0968, 0,0075, 0,0183, 0,0269, 0,0290, 0,0323, 0,0538, 0,7355], **sem N2, CO2 nem H2S**. É o
  fluido da Tabela 6 sem os inertes, que não têm entalpia do HYSYS (L02.4). Em base molar:

  | componente | C1 | C2 | C3 | nC4 | nC5 | nC6 | nC7 | nC8 |
  |---|---|---|---|---|---|---|---|---|
  | código | 0,405 | 0,017 | 0,028 | 0,031 | 0,027 | 0,025 | 0,036 | 0,432 |
  | Tabela 6 | 0,384 | 0,016 | 0,026 | 0,030 | 0,026 | 0,024 | 0,034 | 0,443 |

  A Tabela 6 tem ainda 0,017 de inertes.
- **Estados:** a grelha é P = linspace(1, 140, 20) bar × T = linspace(100, 550, 20) K. Os 159
  estados são os pontos bifásicos no código original **e** no HYSYS.
  [`pt_values.csv`](../masters_code/pt_values.csv) é essa lista: P de 1 a 110,7 bar, T de 147 a 526 K.
- **Figura 13 deslocada:** [`tests/plot_test.py`](../masters_code/tests/plot_test.py) desenha os 159
  pontos a **T − 10 K** (`df_pt_values['1'] - 10`). Os pontos digitalizados da Figura 13 coincidem
  com os de `pt_values.csv` deslocados 10 K: mediana |ΔT| de 0,22 K, contra 10,05 K sem o
  deslocamento. Os notebooks 5.1 e 5.2 e o volume I avaliavam as propriedades 10 K abaixo dos estados
  em que o HYSYS foi comparado. Passaram a usar `dissertation/data/hysys_states.csv` (2026-10-16).
- **Viscosidade do líquido:** com este fluido, nC4–nC8 contam e Nc_m é razoável. Nos 159 estados, a
  viscosidade recalculada com o código original tem mediana 0,157 mPa·s (q10 0,098, q90 0,45). Isto
  está na mesma faixa do cmfsim (mediana 0,18, issue 008).
- **Filtro da Figura 20:** a figura só desenha os pontos com μ_L > 1,8e-5 Pa·s. Hoje esse filtro
  deixa passar os 159. A explicação da issue 008 para os 24 marcadores, a fusão de marcadores
  sobrepostos, mantém-se.
- **Valores do HYSYS:** vêm de `hysys_all_props_values.csv`, num caminho local do autor
  ([`tests/auxiliar/hysys_values.py`](../masters_code/tests/auxiliar/hysys_values.py)). O ficheiro
  **não está no repositório** (L06.3).
- **Reprodutibilidade:** hoje o código original dá 158 dos 159 estados como bifásicos. Os 159
  vieram de uma corrida anterior; pode ser o flash (L02.2–L02.3) ou uma versão diferente.

## L03.5 Redes neuronais para propriedades

Os ficheiros `*.pt` na raiz do repositório original são redes PyTorch para cada propriedade:
`liquid_viscosity_cP.pt`, `vapour_viscosity_cP.pt`, `surface_tension_nasc.pt` e as restantes.
`app/rna/` e `flash_rna.py` / `rna_stream.py` substituem o flash e as propriedades por essas redes,
treinadas em dados do HYSYS (`Propriedades HYSYS.csv`, `nascimento_comp.csv`, fora do repositório).
O simulador com escorregamento importa `RNAFlash`, mas os scripts de validação usam o flash por PR.
Fica como registo de uma via tentada, não usada nos resultados publicados.

## Consequências para o cmfsim

- Nc_eff = 20 para o C7+ (issue 001) mantém-se. O autor não tinha valor: o código dava 0.
- Nada a mudar no simulador. Se se quiser reproduzir a dissertação, uma bandeira «Nc pelo nome» com
  Nc = 0 para o C7+ daria μ_L de gás. Não é física e só serve para comparação.
- Os notebooks 5.1 e 5.2 usam o fluido da Tabela 6 com inertes. Para comparar com as Figuras 13–21
  ponto a ponto, o fluido certo é o de L03.4. A diferença é pequena (~2 % molar) mas existe.
