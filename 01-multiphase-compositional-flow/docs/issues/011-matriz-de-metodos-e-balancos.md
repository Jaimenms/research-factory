# 011 · Matriz de métodos (2 modelos × 3 solvers × 2 backends) e balanços globais

| | |
|---|---|
| tipo | feature / estudo |
| área | modelos (interface comum), solvers (volumes finitos, marcha mecanicista, diferenças finitas no no-slip), diagnóstico |
| estado | feito (2026-10-15) |
| onde | `cmfsim/models/base.py` (`FluxModel`), `cmfsim/models/noslip.py` e `mechanistic.py` (`fluxes_sources`), `cmfsim/solvers/nodal.py` (`NodalSystem`, esquemas `fd` e `fv`), `cmfsim/simulation.py` (`march`, `solve_nodal`, `balances`, `simulate`, `march_many`), esquema (`solver.type = fv`), `batches/solver_matrix.json`, `batches/convergence.json`, notebooks 5.5, 5.6 e 5.8 |
| dados | caso de condensação (§5.6, bolhas a C_0 = 1: mecanicista ≡ no-slip por construção), sistema offshore (§5.5, 78 nós), caso 3 da §5.6 |

## Antes

Dois caminhos fixos: `noslip` só por marcha (RK45 em P, T) e `mechanistic` só por diferenças finitas
em 4 incógnitas por nó (método 3.2 da dissertação). Sem medida de fecho dos balanços globais. As
diferenças finitas derivavam os fluxos através dos vincos (orvalho, mudança de padrão), o que
obrigou à repartição algébrica de massa e à discussão da ordem nos extremos (issue 005).

## Depois

**Interface comum** (`FluxModel`): cada modelo expõe `fluxes_sources(P, T, D, ε, θ, U, T_amb, m)`,
vetorizada em n pontos, com Φ = [Σ m_k v_k/A + P, Σ m_k h_k/m] e S = [−F, Q_w/m − g sin θ]. No
no-slip v_G = v_L e F é o atrito homogéneo; no mecanicista v_k e F vêm do fechamento do padrão e
m_G = m w_G(P, T) (repartição algébrica; a forma (25) integra exatamente para esta). Em bolhas com
C_0 = 1 no horizontal os dois modelos dão o mesmo Φ e S à máquina.

**Três solvers sobre a interface**, todos para numpy e JAX:

| `solver.type` | esquema | incógnitas | observações |
|---|---|---|---|
| `march` | RK45 adaptativo em x, J u' = S com J = ∂Φ/∂u por `jax.linearize` (numpy: diferenças como 169–170), arranque quente do flash | P, T | antes só no no-slip; agora também no mecanicista; `march_many` (vmap) para os dois |
| `fd` | (M Φ)_i = S_i, M de 2.ª ordem (Tabela 4); Newton colorido, banda 2 | P, T por nó | no mecanicista `fd` continua a ser o método 3.2 tal e qual (4 incógnitas, `mass_split`); no no-slip é o «sistema não linear sem escorregamento» das Tabelas 34–35 |
| `fv` | Φ_{i+1} − Φ_i = (h/2)(S_i + S_{i+1}) por célula; Newton colorido, banda 1 | P, T por nó | conservativo por construção; não deriva os fluxos através dos vincos; com P na saída, as linhas de momento são atribuídas à célula da direita e a continuidade de P fica no último nó de cada troço |

Pontos de quebra duplicados com continuidade de P e T; condições de contorno P, T na entrada ou T
na entrada e P na saída; estimativa inicial pela marcha no-slip nos nós (ou plana).

**Balanços globais** (`simulation.balances`, em `info["balances"]` de toda a corrida e na linha de
progresso): por troço, (Φ_fim − Φ_início) − ∫ S dx pelo trapézio na malha do próprio resultado,
para quantidade de movimento [Pa] e energia [J/kg] (e relativos a P_ref e |h_in|); massa por
construção (m_G + m_L = m).

## Medições (2026-10-15, caso de condensação U = 6 W/m²K, 33 nós ou saída a 250 m, rtol 1e-8)

| modelo | solver | P_out [bar] | T_out [°C] | contra no-slip/marcha | balanço momento [Pa] | balanço energia [J/kg] | iterações |
|---|---|---|---|---|---|---|---|
| no-slip | marcha | 69,09611 | 31,442 | — | +1,7 | +8,7 (JAX) / +17,5 (numpy) | 45 + 20 passos |
| no-slip | fd | 69,09613 | 31,445 | 2,7 Pa, 0,005 K | +6,0 | +30 | Newton 1 |
| no-slip | fv | 69,09610 | 31,441 | 1,0 Pa, 0,002 K | −3e-6 | −4e-7 | Newton 1 |
| mecanicista | marcha | 69,09611 | 31,442 | 0,1 Pa | +2,3 | +8,7 | 45 + 20 passos |
| mecanicista | fd (3.2) | 69,09612 | 31,445 | 1,4 Pa, 0,005 K | +5,3 | +30 | Newton 2 |
| mecanicista | fv | 69,09609 | 31,441 | 1,7 Pa, 0,002 K | −5e-4 | −4e-7 | Newton 1 |

numpy e JAX coincidem a < 1 Pa (a marcha numpy usa derivadas por diferenças). Os balanços da
marcha e das diferenças finitas são o erro de tolerância ou de discretização (alguns pascal em
70 bar, 1e-7 relativo); os dos volumes finitos fecham a precisão de máquina na própria malha. O
refinamento dos volumes finitos converge a ~h² (`tests/test_solver_matrix.py`). Os números do
sistema offshore (78 nós, três troços, padrão automático) e do caso 3 da §5.6 estão nos notebooks
5.5, 5.6 e 5.8; o 5.8 substitui a Figura 59 por um estudo de convergência de malha (n = 6 … 192)
com erro contra a marcha a rtol 1e-10, fecho dos balanços, iterações e tempo.

## Sistema offshore (§5.5, três troços, padrão automático, 78 nós; `runs/solver_matrix`)

| modelo | solver | backend | P_out [bar] | T_out [°C] | iterações | balanço momento [Pa] | balanço energia [J/kg] |
|---|---|---|---|---|---|---|---|
| no-slip | marcha | JAX | 105,1552 | 39,104 | 158 passos | −22 | +0,15 |
| no-slip | fd | JAX | 105,1538 | 39,105 | Newton 1 | −250 | +1,8 |
| no-slip | fv | JAX | 105,1559 | 39,104 | Newton 1 | +1e-6 | +6e-7 |
| no-slip | fv | numpy | 105,1559 | 39,104 | Newton 1 | −9e-6 | +6e-7 |
| mecanicista | marcha | JAX | 91,4796 | 38,028 | 158 passos | −36 | +0,13 |
| mecanicista | marcha | numpy | 91,4796 | 38,028 | 158 passos | −37 | −0,68 |
| mecanicista | fd (3.2) | JAX | 91,4768 | 38,028 | Newton 3 | −380 | +1,6 |
| mecanicista | fv | JAX | 91,4814 | 38,028 | Newton 3 | −3e-8 | −8e-11 |

Paridade numpy × JAX: 1e-5 Pa nos volumes finitos, 4 Pa e 2e-4 K na marcha mecanicista (Jacobiana
por diferenças no numpy). Entre métodos, 0,4 mbar na saída em 65 bar de queda (erro de discretização
a 100 m, acumulado na linha de 6 km).

## Convergência de malha (notebook 5.8, `runs/convergence`, erro contra a marcha a rtol 1e-10)

| nós | fd no-slip: max|ΔP| [Pa] / balanço | fv no-slip | fd mecanicista | fv mecanicista |
|---|---|---|---|---|
| 12 | 9,4e3 / −1,2e4 | 3,8e3 / −5e-9 | 1,4e4 / −1,6e4 | 7,4e3 / +2e-8 |
| 28 | 1,1e3 / −1,8e3 | 614 / +5e-4 | 2,0e3 / −2,5e3 | 1,2e3 / −8e-9 |
| 51 | 405 / −680 | 208 / +2e-5 | 901 / −1,1e3 | 457 / −2e-8 |
| 99 | 113 / −180 | 52 / +6e-7 | 244 / −290 | 114 / −3e-8 |
| 195 | 31 / −43 | 13 / +1e-8 | 62 / −71 | 30 / −4e-8 |

Os dois esquemas convergem a ~h² (fator ≈ 4 por duplicação a partir de 28 nós); os volumes finitos
têm metade do erro das diferenças finitas à mesma malha e fecham os balanços a 1e-8 Pa em todas as
malhas, enquanto o desequilíbrio das diferenças finitas é da ordem do próprio erro. O Newton
converge em 1–3 iterações em todos os casos (estimativa inicial pela marcha no-slip).

## Benefício

- Os três solvers resolvem as mesmas equações e verificam-se uns aos outros; o `fv` é a opção
  recomendada para os sistemas nodais (conservação exata, banda estreita, sem vincos), a marcha
  para dados de treino (agora também com escorregamento, via `march_many`) e o `fd` do mecanicista
  fica como reprodução fiel do método 3.2.
- Qualquer corrida traz o fecho dos balanços globais; o teste de rotina exige < 1e-9 relativo nos
  volumes finitos e < 1e-4 nos outros.
