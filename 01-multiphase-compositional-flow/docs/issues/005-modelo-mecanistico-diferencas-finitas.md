# 005 · Modelo mecanicista com escorregamento por diferenças finitas e Newton

| | |
|---|---|
| tipo | feature / hipótese / estudo |
| área | modelo com escorregamento (§2.2, §3.2), solver algébrico (§3.2.6) |
| estado | feito (2026-10-15); Apêndice A em curso |
| onde | `cmfsim/models/mechanistic.py`, `cmfsim/solvers/newton.py`, `cmfsim/simulation.py` (`solve_fd`), `generators/dp_comparison.py` |
| dados | consistência contra a marcha de não escorregamento; Apêndice A (`dissertation/data/appendix_a_*.csv`) |

## Antes

A dissertação discretiza as equações de massa por fase, momento da mistura e energia com
`numpy.gradient` (2.ª ordem) e resolve o sistema algébrico com o Powell híbrido do SciPy, sem
Jacobiana analítica, com holdup e flash aninhados no resíduo; relata problemas de convergência
«talvez pela presença de descontinuidades, como … aparecimento e desaparecimento de uma fase»
(p. 99) e custo que cresce «de forma exponencial» com o número de nós (Tabela 35).

## Depois

- **Resíduos** (188–190) com o mesmo operador de derivada (2.ª ordem, `edge_order` 1 ou 2),
  incógnitas [m_G, m_L, P, T] por nó escaladas por [m, m, P_ref, T_ref], condições de contorno na
  entrada (P, T, m) ou P na saída, continuidade nos pontos de quebra do tubo.
- **Newton amortecido com Jacobiana colorida** (`solvers/newton.py`): 20 direções (banda de 2 nós
  × 4 variáveis), `jvp` vetorizado em JAX ou diferenças avançadas em numpy, busca linear por
  bisseção, limites físicos. Jacobiana colorida igual à densa a 0,0 (teste).
- **Fases que aparecem e desaparecem**: dois cuidados que explicam o que travava a dissertação.
  (1) `mass_split="algebraic"` (m_G = m w_G(P, T)) em vez da EDO (25): no vinco de w_G no ponto de
  orvalho a regra da cadeia discreta m dw/dx ≠ M(m w) falha e o sistema fica inconsistente (o
  Newton estagna com |F| ~ 1e-3; com a forma algébrica converge a 1e-13). (2) O fluxo de momento
  escrito como m_k v_k/A com as velocidades do fechamento, e não como ρ H v² = ρ v_S²/H com H
  num piso: a segunda forma tem curvatura 1/H nos nós monofásicos e a Jacobiana fica quase
  singular (cond 3e9, passo de Newton inútil); a primeira é suave quando a fase some. Mais um
  atrito de fase única com mistura C¹ (`single_phase_blend`).
- **Estimativa inicial** pela marcha de não escorregamento (`initial_guess="march_noslip"`).
- **Parâmetros como argumentos**: tubo, entrada e malha vão em `params`; uma compilação por
  estrutura (fluido, opções, n) serve os 200 casos do Apêndice A.

## Apêndice A (§5.4): 100 linhas horizontais e 100 verticais (2026-10-15, Nc_eff do C7+ = 20)

| orientação | modelo | mediana do desvio à coluna «Dissertação» | < 10 % | < 25 % | dentro da faixa dos modelos do HYSYS |
|---|---|---|---|---|---|
| vertical | mecanicista (3.2) | +3,1 % | 72 % | 94 % | 88 % (dissertação: 96 %) |
| vertical | não escorregamento (3.1) | −14 % | 37 % | 72 % | — |
| horizontal | mecanicista (3.2) | −61 % | 0 % | 0 % | 5 % (dissertação: 63 %) |
| horizontal | não escorregamento (3.1) | −54 % | 0 % | 16 % | — |

Vertical reproduzido (a perda de carga é gravítica e o holdup mecanicista está certo). Horizontal
sistematicamente a ≈ 0,4× da dissertação, em todos os 100 casos, com o estratificado ou o
intermitente identificados pelo critério de Taitel & Dukler; a perda de carga da dissertação
coincide, a baixa vazão, com o nosso **fechamento anular** (110–133 % nos casos 2 e 26), o que
sugere que a identificação de padrão da dissertação (sigmoides de ganho não documentado) pesou
o anular nessas condições. Testado e rejeitado: atrito de Darcy como impresso (×4: mediana
+81 % no horizontal e caso 5 vertical a +471 %), D_G como impresso, viscosidade do líquido
(Nc_eff 39,9 → 20 não muda a mediana horizontal). Sem o código não se fecha; o teste noturno
guarda os valores medidos como regressão.

## Caso offshore com escorregamento (§5.5, Figuras 32–37)

Três troços (poço vertical, linha horizontal, riser), 81 nós, Newton em 3 iterações (|F| 3e-14,
68 s em JAX com compilação). Contra a curva «Modelo» da dissertação:

| ponto | P cmfsim / dissertação (bar) | T (°C) | H_L | padrão cmfsim / dissertação |
|---|---|---|---|---|
| topo do poço (1500 m) | 115,6 / 120,3 | 65,5 / 63,8 | 0,38 / 0,34 | anular / anular |
| base do riser (7500 m) | 102,3 / 115,3 | 41,2 / 40,0 | 0,33 / 0,40 | **anular / estratificado** |
| topo do riser (7800 m) | 91,5 / — | 38,0 / — | 0,45 / — | anular / anular |

A diferença está na linha horizontal: a 117 bar o gás é denso (ρ_G ≈ 100 kg/m³) e o critério de
Kelvin–Helmholtz (32) com v_G real ou com v_SG dá «não estratificado» (v_G ≈ 2,8–3,8 m/s contra
v_KH ≈ 0,7–0,9 m/s), e com h/D < 0,5 segue-se anular, com 2,7× a perda de carga que a dissertação
mostra para o seu estratificado. A classificação «estratificado» da dissertação nesse troço não é
reprodutível com o critério de Taitel & Dukler tal como publicado; fica em aberto (o modelo
TULSA do HYSYS também dá estratificado ali, com outro critério).

## Benefício (consistência medida em 2026-10-15)

Padrão de bolhas fixo com C₀ = 1 no horizontal é, por construção, o modelo de não escorregamento:
o método 3.2 (24 nós) reproduz a marcha 3.1 num gasoduto com condensação a 4 Pa, 0,006 K e
3e-5 em w_G, com Newton em 2–4 iterações (|F| 1e-14, JAX e numpy). Em JAX, 49 nós convergem em
4 iterações; em numpy com diferenças avançadas, 25 nós convergem e 49 estagnam (precisão das
diferenças no vinco do orvalho): o backend de produção do método 3.2 é o JAX, o numpy fica como
referência eager. Apêndice A: ver `notebooks_src/thesis/nb_5_4_dp_comparison.py` (a preencher).
