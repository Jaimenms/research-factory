# 008 · Viscosidade do líquido: a regra de mistura de Allan & Teja dá 3–4× menos do que a dissertação e o HYSYS

| | |
|---|---|
| tipo | hipótese / estudo |
| área | termodinâmica (propriedades de transporte, §2.5.2, eqs. 137–141) |
| estado | aberto (2026-10-15): achado quantificado, causa por identificar; nada adotado |
| onde | `cmfsim/thermo/viscosity.py` (`allan_teja_viscosity`), `notebooks_src/thesis/nb_5_2_viscosity_tension.py`, `dissertation/data/components.json` (`Nc_eff`) |
| dados | Figura 20 digitalizada (paridade «Implementado vs HYSYS», 24 marcadores de 159 pontos), 159 estados (P, T) da Figura 13, fluido da Tabela 6; Apêndice A horizontal |

## O achado

Nos 159 estados da Figura 13 (fluido da Tabela 6, todos bifásicos com T em kelvin; o notebook 5.2
somava 273,15 a temperaturas já em kelvin e comparava 155 estados fora do envelope, corrigido
nesta issue), a distribuição da nossa viscosidade do líquido fica 3–4× abaixo das duas da Figura
20, que concordam entre si:

| μ_L [mPa·s] | q05 | q25 | q50 | q75 | q95 |
|---|---|---|---|---|---|
| HYSYS (eixo y da Figura 20) | 0,116 | 0,442 | 0,699 | 1,165 | 3,749 |
| implementação da dissertação (eixo x) | 0,160 | 0,341 | 0,540 | 0,894 | 3,513 |
| cmfsim, Nc efetivo médio molar (atual) | 0,100 | 0,128 | 0,175 | 0,292 | 0,891 |
| cmfsim, Nc efetivo médio mássico | 0,122 | 0,158 | 0,226 | 0,397 | 1,640 |
| cmfsim, ln μ médio molar dos puros | 0,087 | 0,115 | 0,158 | 0,262 | 0,809 |

As distribuições não são emparelhadas e a Figura 20 só tem 24 marcadores legíveis (os sobrepostos
fundem-se), o que enviesa a distribuição da figura para os valores raros; mas a razão de 3–4× é
muito maior do que esse viés e aparece em todos os quantis. O nC8 puro a 300 K dá 0,503 mPa·s
(medido 0,51): a correlação (137–140) está bem; o problema está na **mistura**. A dissertação não
define o número de carbonos efetivo da mistura (eq. 141, marcada ⚠ no índice de equações) nem os
Nc efetivos de CO₂, N₂ e H₂S (hipótese Nc = 1 na issue 001); usamos Nc_m = Σ x_i Nc_i. Nenhuma
regra de mistura simples fecha a diferença (a mássica sobe 30 %).

Candidatos a investigar (por ordem): (1) a forma impressa de (141) pode não ser uma média: Allan &
Teja (1991) definem o Nc efetivo de um componente não alcano pela viscosidade de referência, e
para a mistura há variantes com média em fração mássica ou com pesos de MW; (2) os Nc efetivos
dos não alcanos; (3) a dissertação pode avaliar a correlação à composição global ou com outra
unidade de temperatura; (4) o HYSYS usa outro método (o eixo x, que é a dissertação, está perto
do HYSYS, logo o que falta está do nosso lado).

## Onde pesa: Apêndice A horizontal

A issue 005 registou os casos horizontais a ~0,4× a dissertação e apontou a viscosidade do
líquido como sensibilidade. Teste direto (`allan_teja_viscosity` × 3, 10 casos horizontais e 5
verticais do lote `batches/dp_comparison.json`, método 3.2 em JAX):

| | cmfsim/dissertação, mediana | casos a < 25 % |
|---|---|---|
| horizontal, μ_L atual | 0,43 | 0 / 10 |
| horizontal, μ_L × 3 | 0,52 | 1 / 10 |
| vertical, μ_L atual | 1,05 | 5 / 5 |
| vertical, μ_L × 3 | 1,05 | 5 / 5 |

Por caso horizontal, μ_L × 3 sobe a razão de 0,32–0,62 para 0,40–0,77 (mais nos casos de 4 in,
intermitentes; quase nada nos de 8–10 in, estratificados). Logo a viscosidade explica uma parte
pequena do défice horizontal; o grosso continua na leitura dos fechamentos horizontais (issue 005:
anular/estratificado contra a dissertação). O vertical não mexe.

## Decisão

Nada adotado: `Nc_eff` e a média molar ficam como estão (default = leitura literal), o notebook
5.2 mostra o painel e a tabela acima, e a investigação fica em aberto até se ler Allan & Teja
(1991) na fonte ou se ter o código do autor. Se se confirmar outra regra, entra como opção com
bandeira em `FlashOptions`/`ComponentSet` (regra 2).
