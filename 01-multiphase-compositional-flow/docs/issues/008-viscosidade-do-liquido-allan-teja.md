# 008 · Viscosidade do líquido (Allan & Teja): o desvio aparente de 3–4× era um artefacto da digitalização; formulação confirmada

| | |
|---|---|
| tipo | estudo |
| área | termodinâmica (propriedades de transporte, §2.5.2, eqs. 137–141); método de comparação com as paridades digitalizadas |
| estado | fechada (2026-10-15): nada a mudar no simulador; comparação por quantis corrigida |
| onde | `cmfsim/thermo/viscosity.py` (`allan_teja_viscosity`), `dissertation/notebook_utils.py` (`merge_like_figure`, `qq_parity`), `notebooks_src/thesis/nb_5_1_thermo.py`, `nb_5_2_viscosity_tension.py` |
| dados | Figuras 14–21 digitalizadas (paridades «Implementado vs HYSYS»), 159 estados (P, T) da Figura 13, fluido da Tabela 6; Apêndice A horizontal |

## O que parecia

O notebook 5.2 somava 273,15 a temperaturas já em kelvin (155 dos 159 estados fora do envelope;
corrigido). Com os estados certos, a distribuição da nossa viscosidade do líquido nos 159 pontos
parecia 3–4× abaixo das duas da Figura 20 (HYSYS no eixo y, implementação da dissertação no eixo
x), em todos os quantis (medianas 0,18 contra 0,70 e 0,54 mPa·s), e a da massa específica do
vapor (Figura 14) 0,70× a do HYSYS.

## A formulação

A fonte original (Allan & Teja, Can. J. Chem. Eng. 69, 986–991, 1991) está atrás de paywall e as
fontes secundárias encontradas (Orbey & Sandler 1993; estudos de misturas de álcoois; revisões de
correlações) não reproduzem a regra de mistura; a única referência verificável é a própria
dissertação, pp. 83–84, que imprime:

- (137) Nc_m = Σ x_i Nc_i, com Nc igual ao número de carbonos para n-alcanos;
- (138–140) A = 145,73 + 99,01 Nc_m + 0,83 Nc_m² − 0,125 Nc_m³; B = 30,48 + 34,01 Nc_m − 1,23 Nc_m² +
  0,017 Nc_m³; C = −3,07 − 1,99 Nc_m;
- (141) η_L = exp[A(−1/B + 1/(T + C))].

É exatamente o que `allan_teja_viscosity` faz (com o fator 1e-3 de cP para Pa·s; o texto não diz a
unidade). O nC8 puro a 300 K dá 0,503 mPa·s (medido 0,51). O ⚠ de (141) no índice de equações fica
levantado.

## O artefacto

As paridades são imagens; os marcadores sobrepostos fundem-se em componentes conexas e a
digitalização conta-os uma vez (a Figura 20 tem 24 marcadores para 159 pontos, a 14 tem 60, a 18
tem 72). Os nossos 159 valores de μ_L concentram-se em 0,1–0,3 mPa·s, que na escala da Figura 20
(0–6 mPa·s em 1013 px) ocupam 17–50 px: fundem-se em meia dúzia de marcadores, e a distribuição
lida fica dominada pelos valores raros e altos. Simulando a fusão nos nossos valores (dois valores a
menos de um diâmetro de marcador ao longo da diagonal viram um só; `merge_like_figure`, diâmetro
da área do marcador na spec):

| figura | marcadores na figura | cmfsim fundido (6 / 10 / 14 px) | mediana crua / HYSYS | mediana fundida / HYSYS |
|---|---|---|---|---|
| 20, μ_L | 24 | 25 / 20 / 15 | 0,25 | 0,82 / 1,04 / 1,02 |
| 14, ρ_V | 60 | 64 / 45 / 38 | 0,70 | 0,86 / 0,94 / 0,96 |
| 18, β | 72 | 62 / 43 / 35 | 0,99 | 1,06 / 1,08 / 1,13 |
| 15, ρ_L | 91 | 68 / 57 / 46 | 1,04 | 0,99 / 1,01 / 0,98 |
| 21, σ | 110 | 84 / 64 / 49 | 0,83 | 0,94 / 1,00 / 1,01 |
| 19, μ_V | 86 | 59 / 41 / 32 | 0,89 | 0,83 / 0,85 / 0,83 |

Fundidos, os nossos valores reproduzem o número de marcadores e a distribuição da figura em μ_L,
ρ_V, β, ρ_L e σ. A única propriedade com desvio consistente é a viscosidade do vapor (Pedersen):
~15 % abaixo do HYSYS em todas as leituras, com dispersão menor (q95 18,8 contra 22,8 µPa·s), o que
é coerente com o que a dissertação diz da correlação de Pedersen para componentes mais pesados.

## Sensibilidade do Apêndice A horizontal a μ_L (informação, não correção)

Teste direto (`allan_teja_viscosity` × 3, 10 casos horizontais e 5 verticais do lote
`batches/dp_comparison.json`, método 3.2 em JAX): a mediana cmfsim/dissertação horizontal sobe de
0,43 para 0,52 (mais nos casos de 4 in, intermitentes; quase nada nos de 8–10 in, estratificados);
a vertical fica em 1,05. A viscosidade do líquido não é a causa do défice horizontal (issue 005).

## Decisão e lição

Nada muda no simulador. A comparação por quantis dos notebooks 5.1 e 5.2 passa a fundir os nossos
valores como a figura (curva a cheio) e a mostrar a distribuição crua a tracejado; o número de
marcadores lido e o fundido ficam na tabela. Regra para o futuro: um Q–Q contra marcadores
digitalizados sem simular a fusão é enganador sempre que a propriedade se concentra em valores
baixos.
