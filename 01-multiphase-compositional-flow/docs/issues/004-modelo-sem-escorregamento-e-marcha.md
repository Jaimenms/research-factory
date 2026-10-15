# 004 · Modelo de não escorregamento e marcha em x: fluxos conservados, integrador próprio e o que as figuras mostram

| | |
|---|---|
| tipo | feature / hipótese / estudo |
| área | modelo de não escorregamento (§2.2.4, §3.1), integrador em x (§3.1.6) |
| estado | feito (2026-10-15); achado térmico em aberto |
| onde | `cmfsim/models/noslip.py`, `cmfsim/solvers/rk45.py`, `cmfsim/pipe.py`, `cmfsim/simulation.py`, `cases/condensate/`, `batches/dissertation.json` |
| dados | §5.6 (Tabelas 23–25); Figuras 38–40 digitalizadas (`references/figures/data/fig_38..40.json`) |

## Antes

A dissertação expande as derivadas totais termo a termo (150–163), calcula ∂M/∂P e ∂M/∂T por
diferenças de 1.ª ordem com três flashes por avaliação (169–170, Figura 10) e integra com o LSODA
do SciPy. Não diz como calcula o atrito no modelo de não escorregamento (B15) nem o passo das
diferenças (B4).

## Depois

- **Fluxos conservados**: Φ(P, T) = [G v + P, h_mix], S = [−F, Q_w/m − g sin θ], J u' = S com
  J = ∂Φ/∂u. É a mesma álgebra de (159, 163) sem a expansão manual. `derivatives="autodiff"` (JAX,
  um flash por avaliação, derivadas exatas) ou `"finite_difference"` (dP = 1 kPa, dT = 0,01 K para
  trás, como 169–170; caminho fiel). Diferença entre os dois no caso 2: < 2e2 Pa e < 5e-3 K.
- **Atrito** (`friction="homogeneous"`, hipótese B15): Churchill na mistura homogénea com ρ_mix e
  μ_mix ponderadas em volume, mais ρ_mix g sin θ. A identificação de padrão corre só para reporte.
- **Integrador próprio** (A5): Dormand–Prince 5(4) adaptativo, de ponto de saída a ponto de saída,
  pontos de quebra do tubo como eventos, estágios num `fori_loop` (compila o lado direito uma só
  vez: 34 s de compilação em vez de 298 s), erro escalado por [1 bar, 1 K]. Caso 2 a rtol 1e-6: 42
  passos aceites, 0 rejeitados; 175 ms por corrida compilada contra 12 s em numpy.
- **Tubo único com perfil axial** (A9): `pipe.profile` com D, rugosidade, ângulo, U, T_amb por troço.
- **Casos e lote**: `generators/condensate.py` → `cases/condensate/` (base + 6 variantes por
  `$extends`), `batches/dissertation.json`.

## Achados contra as Figuras 38–40 (caso 2: 60 °C, U = 3 W/m²K)

| grandeza | dissertação (digitalizada) | cmfsim | nota |
|---|---|---|---|
| T na saída, caso 1 (U = 1) | 43,5 °C | — | |
| T na saída, caso 2 (U = 3) | 31,0 °C | 44,9 °C | |
| T na saída, caso 3 (U = 6) | 26,9 °C | — | |
| T onde começa a condensação (casos 2, 3, 6) | ≈ 33 °C (x ≈ 6900, 3350, 4300 m) | orvalho do flash a 69 bar: 34,4 °C | **termodinâmica concordante** |
| ΔP caso 2 | 0,89 bar | 0,97 bar | atrito homogéneo; T diferente |

Os perfis de temperatura da dissertação arrefecem cerca de 2,7× mais depressa do que o balanço
−U π D (T − T_amb) = m dh/dx com o h de PR + gás ideal: o seu caso 1 (U = 1) chega a 43,5 °C, o nosso
caso 2 (U = 3) a 44,9 °C. Um balanço exponencial com cp constante implica para a dissertação cp
efetivo ≈ 800 J/(kg K) (contra ≈ 2300 do modelo), e não constante entre os casos (802, 869, 1053
J/(kg K) para U = 1, 3, 6). Hipóteses: inconsistência de unidades em (160)/(182) (Q_w por metro
contra o lado esquerdo por metro cúbico), entalpia por mol contra vazão em kg, ou (148) sem os
fatores do integral. Sem código nem autor não se resolve; fica como opção futura
(`energy.cp_scale` ou variante de Q_w) só se o dono quiser reproduzir as figuras tal e qual. A
posição em que a condensação começa é consequência direta deste achado (a temperatura em que
começa, não).

### Hipótese testada: Cp de gás ideal avaliado com T em °C

Se o polinómio Cp/R = A + B T + C T² for avaliado com T em °C (erro de unidade plausível num
código que recebe °C nas tabelas), o cp da mistura a 45 °C cai de ≈ 2050 para ≈ 800 J/(kg K),
exatamente o valor implícito no caso 1 da dissertação. Opção `fluid.cp_temperature = "celsius"`
(`ComponentSet.cp_temperature`; lote `batches/dissertation_celsius.json`):

| caso | T_out dissertação | T_out Kelvin (físico) | T_out Celsius (hipótese) | x_cond dissertação / Celsius |
|---|---|---|---|---|
| 1 (60 °C, U 1) | 43,6 | 53,6 | 49,4 | — |
| 2 (60, 3) | 30,9 | 44,9 | 37,9 | 6948 / — |
| 3 (60, 6) | 26,9 | 36,6 | 31,7 | 3319 / 5600 |
| 4 (80, 1) | 53,8 | 70,0 | 63,2 | — |
| 5 (80, 3) | 33,7 | 55,9 | 44,7 | 4795 / — |
| 6 (80, 6) | 27,5 | 42,9 | 33,5 | 4292 / 7300 |

A hipótese explica cerca de metade do excesso de arrefecimento (fator ≈ 2 dos ≈ 2,7) e faz os
casos 3 e 6 condensar como na dissertação; resta um fator ≈ 1,4 por explicar. A temperatura a
que a condensação começa coincide a 1,5 °C (34,2 contra 32,2–32,9). O default fica em
`"kelvin"` (física correta, que é o que interessa para treinar redes); a opção existe para quem
quiser reproduzir as figuras.

## Benefício

Caso 5.6 reexecutável em segundos, com derivadas exatas, corrida gravada com o caso completo e
comparação direta com as figuras digitalizadas (`notebooks_src/thesis/nb_5_6_condensate.py`).
