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

## Benefício (consistência medida em 2026-10-15)

Padrão de bolhas fixo com C₀ = 1 no horizontal é, por construção, o modelo de não escorregamento:
o método 3.2 (24 nós) reproduz a marcha 3.1 num gasoduto com condensação a 4 Pa, 0,006 K e
3e-5 em w_G, com Newton em 2–4 iterações (|F| 1e-14, JAX e numpy). Em JAX, 49 nós convergem em
4 iterações; em numpy com diferenças avançadas, 25 nós convergem e 49 estagnam (precisão das
diferenças no vinco do orvalho): o backend de produção do método 3.2 é o JAX, o numpy fica como
referência eager. Apêndice A: ver `notebooks_src/thesis/nb_5_4_dp_comparison.py` (a preencher).
