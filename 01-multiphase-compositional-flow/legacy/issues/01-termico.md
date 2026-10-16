# L01 · Balanço térmico: perímetro da perda de calor e geometria passada aos casos

| | |
|---|---|
| família | termo de troca de calor com o ambiente, U e diâmetro de referência |
| código | [`app/simulator/simulator.py`](../masters_code/app/simulator/simulator.py), [`app/simulator/simulator_slip.py`](../masters_code/app/simulator/simulator_slip.py), [`app/edge/segment.py`](../masters_code/app/edge/segment.py), scripts em [`tests/`](../masters_code/tests) |
| verificado | corridas do código original (2026-10-16): §5.6 ([`scripts/condensate_5_6.py`](../scripts/condensate_5_6.py)), Apêndice B linha ([`scripts/appendix_b_line.py`](../scripts/appendix_b_line.py)), §5.5 no-slip ([`scripts/offshore_5_5.py`](../scripts/offshore_5_5.py)); cmfsim no Apêndice B com U × D_ext/D |
| responde a | fator 3,42 em U na §5.6 (issue 004); linha do Apêndice B a arrefecer 1,4× mais (issue 009); §5.5 só fechar com D_in (issue 004) |

## Resumo

- **Perímetro:** o código original multiplica U pelo perímetro **externo**, π (D + 2 e)
  (L01.1).
- **§5.6:** o script passou o diâmetro externo no lugar da espessura. O perímetro sai 3,395× o
  interno, e esse é o fator 3,42 que ajustámos (L01.2).
- **Apêndice B:** com a espessura da Tabela 21, a linha e o riser usam 1,83× o perímetro interno. Isso
  explica o arrefecimento 1,4× maior; o cmfsim com U × 1,83 reproduz o Apêndice B a 0,4 K (L01.3).
- **§5.5:** a figura da dissertação só se reproduz com o perímetro interno. Foi feita com outra
  geometria ou outra versão do código (L01.4).

## L01.1 Perda de calor com o perímetro externo

- **Geometria:** `Segment.set_geometry(diameter, thickness, roughness)` guarda D interno, espessura
  e rugosidade ([`segment.py`](../masters_code/app/edge/segment.py)).
- **No-slip** ([simulator.py:430](../masters_code/app/simulator/simulator.py#L430)):

  ```
  B2 = −(1/A) U π (D + 2 e) (T − T_amb) − (ρ v)_m g dz/dL
  ```

- **Com escorregamento:** o modelo usa a mesma forma
  ([simulator_slip.py:107](../masters_code/app/simulator/simulator_slip.py#L107)).
- **Versão sem uso:** `system_ivp_dassl`
  ([simulator.py:540](../masters_code/app/simulator/simulator.py#L540)) tem U π D, com sinal +. Não é
  chamada por `solve_integrate`; é uma versão antiga.

A eq. (100) da dissertação escreve U π D com D interno. O código usa D_ext = D + 2e, coerente com a
Tabela 21, que dá D_ext.

## L01.2 §5.6: diâmetro externo passado como espessura (fator 3,395)

[`tests/validation_condensate_pipeline.py`](../masters_code/tests/validation_condensate_pipeline.py):

```python
inside_diameter = 304.8e-3
outside_diameter = 365e-3
thickness = (outside_diameter - inside_diameter)/2      # calculada e não usada
seg1.set_geometry(inside_diameter, outside_diameter, roughness)
```

A espessura efetiva é 0,365 m. O perímetro térmico fica π (0,3048 + 0,730) = π · 1,0348 m:

| perímetro | fator sobre π D_in |
|---|---|
| código | 3,395 |
| ajuste do cmfsim (issue 004) | 3,42 (a 0,7 % do código) |
| intenção, D_ext = 0,365 m | 1,197 |

O código original reproduz a Figura 39 com rms de 0,02–0,06 °C nos casos 1, 2, 3, 4 e 6, e a pressão
de saída da Figura 38 (69,04–69,18 bar).

**Consequência:** o fator está identificado. A issue 004 tratou-o como erro de apresentação de U
(decisão do dono, U_eff = 3,42 U), e isso equivale a este erro de entrada no script. Duas opções para
o cmfsim, a decidir pelo dono:

- manter 3,42, que reproduz as figuras;
- usar 3,395, o valor exato do código, que reproduz as figuras com a mesma qualidade.

O valor «físico» com o D_ext pretendido seria 1,197.

## L01.3 Apêndice B (e §5.7): U referido ao diâmetro externo da Tabela 21

[`tests/test_generate_samples_sim.py`](../masters_code/tests/test_generate_samples_sim.py) gera as
amostras do Apêndice B com esta geometria:

| trecho | D [m] | espessura [m] | perímetro / π D | U [W/(m² K)] | T_amb [K] | comprimento |
|---|---|---|---|---|---|---|
| poço | 0,127 | 0,00635 | 1,10 | 7 | 313 | 1500 m vertical |
| linha | 0,1524 | 0,0635 | 1,83 | 5 | 285 | 6000 m horizontal |
| riser | 0,1524 | 0,0635 | 1,83 | 5 | 285 | **250 m** vertical (Tabela 21: 300 m) |

O integrador é RK45 (`solve_integrate('RK45')`) e o fluido é o da Tabela 6 em frações mássicas (L06.2).

**Medido.** Linhas 0–9 do Apêndice B (linha), código original sem H2S (L02.4):

| | T_out − tese | P_out − tese | linhas que correm |
|---|---|---|---|
| código com e = 0,0635 m (perímetro 1,83×) | −0,14 a −0,38 K | −0,06 a +0,12 bar | 8 de 10 |
| código com e = 0 (perímetro interno) | +6,7 a +16,5 K | −0,24 a −0,05 bar | 8 de 10 |
| cmfsim, U × 1 | +10,8 K (média), máx. 16,2 K | −1,5 bar (média), máx. 6,2 | 10 de 10 |
| cmfsim, U × 1,833 | −0,37 K (média), máx. 0,73 K | −1,2 bar (média), máx. 5,4 | 10 de 10 |

As linhas 0 e 4 param no código original («local variable 'Z' referenced before assignment», sem raiz
real na cúbica).

**Consequência:**

- **Temperatura:** o «arrefece 1,4× mais» da issue 009 fica explicado. O cmfsim reproduz a temperatura
  do Apêndice B com U referido ao D_ext, com U × D_ext/D no caso.
- **Pressão:** a diferença que resta é de pressão (cmfsim −1,2 bar em média) e vem do atrito do
  no-slip original: Reynolds da mistura com viscosidade constante (L03.1) e τ/D_h com velocidades
  superficiais.
- **Riser:** os 250 m do riser (contra 300 m na Tabela 21) também estão no script.

## L01.4 §5.5: a figura da dissertação usa o perímetro interno

O script do caso offshore no repositório,
[`tests/result_base_case_model.py`](../masters_code/tests/result_base_case_model.py), não é o que
gerou a §5.5. Na versão final usa um fluido de 4 componentes (N2, CO2, C1, nC8). Por isso a §5.5 foi
reconstruída com o código original, o fluido da Tabela 19 (C7+ = C20pseud), 170 bar, 80 °C e 10 kg/s,
com a geometria da Tabela 21 e riser de 250 m.

| variante | λ_L (Figura 34) rms | T (Figura 33) rms | P (Figura 32) rms |
|---|---|---|---|
| perímetro interno (e = 0) | 0,007 | 0,41 °C | 2,7 bar |
| perímetro externo (e da Tabela 21) | interrompida aos 22 min sem terminar (a integração LSODA não avança; com o perímetro interno leva ~1 min) | — | — |

Com o perímetro interno, o código original reproduz o holdup no-slip e a temperatura:

| x | T código / Figura 33 | λ_L código / Figura 34 | P código / Figura 32 |
|---|---|---|---|
| 200 m | 78,35 / 78,26 °C | 0,333 / 0,333 | 162,8 / 162,8 bar |
| 1500 m | 67,8 / 67,3 °C | 0,274 / 0,276 | 119,5 / 120,0 bar |
| 4000 m | 55,2 / 54,6 °C | 0,285 / 0,290 | 116,0 / 118,1 bar |
| 7500 m | 42,0 / 41,8 °C | 0,296 / 0,308 | 111,3 / 116,1 bar |

A issue 004 já tinha visto que o cmfsim só fecha a §5.5 com D_in: rms 0,65 °C contra 8,1 °C com D_ext.
Isto está confirmado: a figura foi gerada com espessura nula ou com uma versão do termo em π D, como
`system_ivp_dassl`.

**Pressão:** a pressão na linha não se reproduz com o código tal como está: perde 4,8 bar a mais até à
base do riser. A causa não é térmica, é o padrão de escoamento. Na tese a linha é estratificada; no
código atual é anular, por uma margem de 2 % no critério de Kelvin–Helmholtz. Com o critério em
sigmoide do módulo com escorregamento, o rms cai de 2,7 para 0,6 bar (L04.6).

## Resumo das questões térmicas

| questão | resposta |
|---|---|
| fator 3,42 da §5.6 | diâmetro externo passado como espessura; fator do código 3,395 |
| linha do Apêndice B 1,4× mais fria | U referido ao D_ext (1,83×); cmfsim com U × 1,833 reproduz a −0,37 K em média |
| §5.5 fecha com D_in | figura gerada sem espessura (ou com o termo antigo em π D) |
| riser | 250 m no código (300 m na Tabela 21) |
