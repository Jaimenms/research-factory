# L02 · Termodinâmica: Peng–Robinson, flash e entalpia

| | |
|---|---|
| família | equação de estado, coeficientes de fugacidade, flash, entalpia de gás ideal |
| código | [`equation_of_state.py`](../masters_code/app/thermodynamic_v2/equation_of_state/equation_of_state.py), [`cubic_equation.py`](../masters_code/app/thermodynamic_v2/equation_of_state/cubic_equation.py), [`flash2.py`](../masters_code/app/thermodynamic_v2/flash/flash2.py), [`flash2_via_ss_normal.py`](../masters_code/app/thermodynamic_v2/flash/flash2_via_ss_normal.py), [`data/`](../masters_code/data) |
| verificado | corridas do código original (2026-10-16), comparação com o cmfsim e com a biblioteca `thermo` ([`scripts/flash_checks.py`](../scripts/flash_checks.py), [`scripts/condensate_5_6.py`](../scripts/condensate_5_6.py), [`scripts/offshore_5_5.py`](../scripts/offshore_5_5.py)) |
| responde a | holdup no-slip da §5.5 a 0,85× (Figura 34, issue 004); banda morta da Figura 40 (issue 004); Cp do C7+, k_ij e Nc_eff (issue 001, B1–B3) |

## Resumo

- **ln φ:** o código original não usa o ln φ de Peng–Robinson padrão (L02.1). Essa diferença
  explica sozinha o holdup no-slip 0,85× da §5.5.
- **Salto da Figura 40:** o flash converge para a solução trivial perto do orvalho e chama-lhe vapor
  (L02.2).
- **Fecho de fase:** há um defeito latente no fecho da fase (L02.3), sem efeito nos casos testados.
- **Entalpia de gás ideal:** vem dos polinómios do HYSYS (L02.4), e não do Cp de Smith da Tabela 8.
  Por isso o H2S não tem entalpia.

## L02.1 Coeficiente de fugacidade fora da forma de Peng–Robinson

- **a_ij:** é construído como `((1 − k_ij) √a_i √a_j)**2`, ou seja a_i a_j
  ([equation_of_state.py:55](../masters_code/app/thermodynamic_v2/equation_of_state/equation_of_state.py#L55)).
- **ln φ** ([:159](../masters_code/app/thermodynamic_v2/equation_of_state/equation_of_state.py#L159)):

  ```
  ln φ_i = (b_i/b)(Z − 1) − ln(Z − B) + A/(2,82 B) · (2 Σ_j x_j a_ij / a − b_i/b) · ln[(Z + 2,414 B)/(Z − 0,414 B)]
  ```

  O sinal do último termo é **+**. Na forma padrão é −, com a_ij = (1 − k_ij)√(a_i a_j).
- **Termo de atração:** como a_i a_j é ~1e-11 nas unidades do código (bar, m³/mol), o termo em a_ij
  praticamente desaparece. O que sobra, −A/(2,82 B)·(b_i/b)·ln(…), só depende do componente através
  de b_i.
- **Mistura:** a regra de mistura de a (`am`) está certa, por isso Z e as massas específicas de cada
  fase estão certos. A diferença está só no equilíbrio de fases, nos K.

**Medido.** Estado de 116 bar e 328,3 K, composição global da Tabela 19 com C7+ = C20pseud:

| componente | ln φ código original | ln φ PR padrão (= cmfsim) |
|---|---|---|
| CO2 | +0,589 | −0,469 |
| N2 | +0,712 | +1,191 |
| C1 | +0,583 | +0,459 |
| C7+ | −33,82 | −29,92 |

O mesmo flash do código original, com ln φ padrão injetado em memória, reproduz o cmfsim e a `thermo`:

| estado da §5.5 | β original / padrão / cmfsim | λ_L original / padrão / cmfsim | λ_L da Figura 34 |
|---|---|---|---|
| 162,8 bar, 78,4 °C (200 m) | 0,881 / 0,747 / 0,747 | 0,332 / 0,394 / 0,394 | 0,333 |
| 116,0 bar, 55,2 °C (4000 m) | 0,887 / 0,756 / 0,756 | 0,285 / 0,334 / 0,334 | 0,290 |
| 112,0 bar, 43,6 °C (7000 m) | 0,887 / 0,737 / 0,737 | 0,295 / 0,349 / 0,349 | 0,306 |

A `thermo`, com os mesmos Tc, Pc e ω e k_ij = 0, dá β = 0,7559 e 0,7372 nos dois últimos estados,
como o cmfsim. Trocar 2,82 por 2√2 e 2,414/0,414 por 1 ± √2 não muda nada (λ_L 0,3323 contra 0,3324).

**Consequências:**

- O holdup no-slip da §5.5 (Figura 34) é 0,85× o do cmfsim por causa de L02.1. A issue 004
  levantava duas hipóteses: ρ_L de referência ou Pc do C7+ × 1,25. Ambas ficam descartadas. A
  dissertação tem mais vapor (β 0,88 contra 0,75), e não um líquido mais denso.
- O erro pesa nos fluidos com C7+ e componentes intermédios (Tabela 19). No binário C1/C20pseud do
  Apêndice A o efeito é pequeno: λ_L 0,228 contra 0,223 no caso 1. No gás rico da §5.6 (C1–nC5) o
  equilíbrio fica perto do nosso: depois do salto, as curvas da Figura 40 seguem as do cmfsim a ±0,5 %.
- O envelope de fases e as paridades das Figuras 13–21 da dissertação vêm deste ln φ (ver L03.4). O
  volume I compara o cmfsim diretamente com o HYSYS e não é afetado.

## L02.2 Banda morta da Figura 40: solução trivial sem teste de estabilidade

- **Arranque:** o flash ([`flash2_via_ss_normal.py`](../masters_code/app/thermodynamic_v2/flash/flash2_via_ss_normal.py))
  parte de K de Wilson a cada chamada
  ([flash2.py:42](../masters_code/app/thermodynamic_v2/flash/flash2.py#L42)).
- **Iteração:** faz substituições sucessivas `K *= factor`, com tolerância 1e-6 e até 200 iterações
  ([:106](../masters_code/app/thermodynamic_v2/flash/flash2_via_ss_normal.py#L106)).
- **Rachford–Rice:** é resolvido sem limites por `optimize.root` a partir de 0,5 (flash negativo)
  ([:74](../masters_code/app/thermodynamic_v2/flash/flash2_via_ss_normal.py#L74)).
- **Sem teste de estabilidade:** perto do orvalho, as substituições sucessivas caem na solução
  trivial (K → 1). A raiz de Rachford–Rice sai de (0, 1) e diverge (|β| ~ 1e85), e o ponto é
  declarado vapor.

**Medido.** Gás rico da §5.6 a 69,1 bar. O orvalho de PR, do cmfsim e da `thermo`, está a 34,3 °C:

| T [°C] | 36,0 … 32,0 | 31,75 | 31,5 | 31,0 |
|---|---|---|---|---|
| resultado do flash original | vapor (max \|ln K\| < 1e-3, solução trivial) | β = 0,9587 | 0,9568 | 0,9530 |

- **Salto:** a fronteira de deteção fica 2,5 K abaixo do orvalho. Ao sair da bacia da solução
  trivial, o flash salta de 1 para β ≈ 0,959 (molar), ou seja ≈ 1,4 % de líquido em volume. É o
  salto da Figura 40.
- **Reprodução:** o código original reproduz as Figuras 39 e 40 da §5.6:
  - rms de T de 0,02–0,06 °C nos casos 1, 2, 3, 4 e 6;
  - salto em 6949, 3394 e 4364 m, contra 6948, 3319–3393 e 4292–4361 m na Figura 40;
  - fração vaporizada de 0,985–0,987 logo depois do salto.
- **Coerência com a issue 004:** a issue 004 media TPD de −0,008 a −0,012 no ponto do salto. O estado
  já era instável e o flash original não o via.

**Consequência:** o «x_cond» da dissertação está sistematicamente ~1 km depois do físico. A issue 004
já o tratava assim; a causa está agora identificada. O cmfsim faz teste de estabilidade (TPD) e
polimento de Newton, e não tem esta banda.

## L02.3 Fecho da fase com o valor da iteração anterior (defeito latente)

- **Inicialização:** `v = np.empty_like(temperature)`, sem valores definidos
  ([:51](../masters_code/app/thermodynamic_v2/flash/flash2_via_ss_normal.py#L51)).
- **O que se testa:** `if v[i] >= 1` / `elif v[i] <= 0`
  ([:82-85](../masters_code/app/thermodynamic_v2/flash/flash2_via_ss_normal.py#L82-L85)) testa o valor
  guardado na iteração anterior, ou o lixo de memória na primeira, e não a raiz nova `vi`.
- **Efeito:** se a raiz sai de (0, 1) numa iteração qualquer, a fase fica trancada em vapor ou
  líquido até ao fim.
- **Nos casos medidos:** o lixo inicial foi 5e-324, dentro de (0, 1), e inicializar com 0,5 dá
  resultados idênticos na varredura de L02.2. Não é a causa da banda morta.
- **Risco:** pode tornar o resultado dependente da memória noutras máquinas ou versões do numpy.

## L02.4 Entalpia: polinómios de gás ideal do HYSYS, não o Cp da Tabela 8

- **Fórmula:** `enthalpy = HR + ideal_gas_mix_enthalpy_hysys`
  ([:221](../masters_code/app/thermodynamic_v2/equation_of_state/equation_of_state.py#L221), [:231](../masters_code/app/thermodynamic_v2/equation_of_state/equation_of_state.py#L231)).
- **H_ig:** é o polinómio do HYSYS H = A + B T + … + F T⁵ em kJ/kg
  ([`data/ideal_gas_prop_hysys.csv`](../masters_code/data/ideal_gas_prop_hysys.csv)), com média
  ponderada pelas frações mássicas da fase.
- **Smith:** os coeficientes de Smith da Tabela 8
  ([`data/ideal_gas_prop.csv`](../masters_code/data/ideal_gas_prop.csv)) não entram na entalpia.

**Cp do C7+ (B1 da issue 001).** O C20pseud tem B = 5,255e-2, C = 2,9e-3, D = −7,426e-7 (kJ/kg).
O Cp do código e a extrapolação do cmfsim (issue 001: Cp/R = 21,4 + 0,34 T − 108,6e-6 T²) dão:

| T [K] | 300 | 350 | 400 |
|---|---|---|---|
| Cp do código original [kJ/(kg K)] | 1,59 | 1,81 | 2,02 |
| Cp extrapolado do cmfsim [kJ/(kg K)] | 1,69 | 1,89 | 2,08 |

A extrapolação do cmfsim fica 3–6 % acima. A hipótese da issue 001 sai confirmada na ordem de
grandeza e pode trocar-se pelo polinómio do autor.

**H2S sem entalpia.** A linha do H2S no ficheiro do HYSYS está vazia
([linha 80](../masters_code/data/ideal_gas_prop_hysys.csv#L80)), tal como as de C7pseud e de muitos
outros componentes. Com H2S no fluido, a entalpia sai NaN e a integração para com «local variable 'Z'
referenced before assignment». Consequências:

- O fluido da Tabela 6, com 0,06 % de H2S em massa, não corre com o código tal como está no
  repositório. Para o Apêndice B retirou-se o H2S (L01.3).
- A validação de propriedades (Figuras 13–21) usou o fluido sem N2, CO2 e H2S (L03.4). Esta é a «falta
  de propriedades» no código.

## L02.5 Outros pontos confirmados

- **k_ij:** zero em todos os pares
  ([:78](../masters_code/app/thermodynamic_v2/equation_of_state/equation_of_state.py#L78)), como
  assumido na issue 001.
- **Constantes de PR:** Ω_a = 0,45724 e Ω_b = 0,07780. κ(ω) = 0,37464 + 1,54226 ω − 0,26992 ω², sem o
  ramo de PR78 para ω > 0,49 (o C7+ tem ω = 1,32), igual ao cmfsim.
- **Raiz cúbica:** escolhe a de menor energia de Gibbs entre a menor e a maior, com a mesma regra para
  as duas fases
  ([cubic_equation.py:60](../masters_code/app/thermodynamic_v2/equation_of_state/cubic_equation.py#L60)).
- **Massa específica:** PR sem translação de volume. A ρ_L = 950 kg/m³ da Tabela 20 não é usada.
- **C7+ = `C20pseud`:** MW 560, ω 1,32, Tc 991,4 K, Pc 8,379 bar, Tb 841,9 K e H_f −1023 kJ/kg
  ([`data/pure_comp_prop.csv`](../masters_code/data/pure_comp_prop.csv)), como na Tabela 20. Existe
  também um `C7pseud` (MW 274) sem H_ig, não usado.
