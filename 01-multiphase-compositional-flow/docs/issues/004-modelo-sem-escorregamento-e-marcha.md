# 004 · Modelo de não escorregamento e marcha em x: fluxos conservados, integrador próprio e o que as figuras mostram

| | |
|---|---|
| tipo | feature / hipótese / estudo |
| área | modelo de não escorregamento (§2.2.4, §3.1), integrador em x (§3.1.6) |
| estado | feito (2026-10-15); achado térmico resolvido no mesmo dia: U_eff = 3,42 × U da Tabela 25 (decisão do dono) |
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

## Achados contra as Figuras 38–40 com os U impressos (caso 2: 60 °C, U = 3 W/m²K)

*Secção histórica: números com os U da Tabela 25 tal como impressos. A resolução está mais abaixo.*

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

### Hipótese testada e superada: Cp de gás ideal avaliado com T em °C

Se o polinómio Cp/R = A + B T + C T² for avaliado com T em °C (erro de unidade plausível num
código que recebe °C nas tabelas), o cp da mistura a 45 °C cai de ≈ 2050 para ≈ 800 J/(kg K),
exatamente o valor implícito no caso 1 da dissertação. Opção `fluid.cp_temperature = "celsius"`
(`ComponentSet.cp_temperature`; o lote `batches/dissertation_celsius.json` que gerou esta tabela foi apagado em 2026-10-15 com a aprovação do dono):

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

### Contraprova no caso offshore (§5.5, Figuras 32–33)

Com o mesmo balanço de energia (Kelvin, físico), o modelo de não escorregamento reproduz as curvas
«Modelo no-slip» da dissertação no sistema poço–linha–riser: P = 119,98 contra 119,97 bar no topo
do poço (1500 m), 117,1 contra 117,9 bar a meio da linha, 114,4 contra 116,1 bar na base do
riser; T = 65,9 / 52,2 / 42,0 °C contra 67,3 / 52,5 / 41,8 °C; padrão anular em todo o percurso,
como a dissertação relata para o seu modelo no-slip. Logo a equação de energia e o seu
fecho estão certos e a discrepância térmica fica **isolada na secção 5.6** (entradas do gasoduto
diferentes das Tabelas 24–25, ou outra forma de Q_w nesse caso); a hipótese dos °C fica como
curiosidade, não como correção.

## Resolução do achado térmico (2026-10-15): um só fator no termo de perda de calor

**Método.** Aos perfis digitalizados da Figura 39 ajustou-se, no troço monofásico (T > 34,5 °C), a lei
dT/dx = −a (T − T_amb) − b. Resultado: exponenciais puras (desvio 0,01–0,09 °C), b nulo, e a/U igual nos
seis casos (7,5–8,2e-5 m⁻¹ por W/m²K, com 60 ou 80 °C na entrada). O erro é, portanto, um fator
multiplicativo constante no termo U π D (T − T_amb) / (m cp), sem termo aditivo e sem dependência da
temperatura; a razão para o cmfsim foi 3,25 ± 0,07 no ajuste e 3,42 na varredura direta.

| U × k no cmfsim | rms contra a Fig. 39, casos 1 / 2 / 3 / 4 / 6 [°C] | média do desvio |
|---|---|---|
| 1,00 | 6,5 / 11,5 / 11,7 / 10,6 / 18,8 | — |
| 3,14 (π) | 0,60 / 0,91 / 0,77 / 0,99 / 1,18 | tese mais fria |
| 3,28 (1/D) | 0,27 / 0,52 / 0,40 / 0,46 / 0,62 | −0,2 a −0,6 |
| **3,42** | **0,06 / 0,27 / 0,13 / 0,06 / 0,20** | ≈ 0 |
| 3,60 | 0,47 / 0,54 / 0,42 / 0,71 / 0,68 | +0,4 a +0,7 |

O caso 5 (série pontilhada, mal digitalizada) ficou de fora. Com U × 3,42 a pressão de saída da Figura 38
bate a 0,04 bar nos cinco casos (69,039 contra 69,039 bar no caso 1) e o início da condensação
(critério 0,9995) fica a 0,3–1,1 km do da tese (5800/6948, 3000/3319, 3900/4292 m nos casos 2, 3, 6; no
caso 2 o gás arrefece 1,5 K/km perto do orvalho e a série da Figura 40 é pontilhada). A «fração
vaporizada» da Figura 40 é a molar: no troço bifásico o cmfsim segue-a a 0,004 (β), contra 0,02 em base
mássica e 0,015 em volumétrica (B6 de docs/05 respondida para a figura).

**Hipóteses excluídas.** Cp do C7+ (o fluido da Tabela 23 não o tem); termo de expansão ou cinético
(exigiria b ≠ 0); Cp em °C sozinho (fator 1,8, rms 4–9 °C); entalpia ideal sem o fator R (ordem certa,
forma errada: rms até 2,5 °C, porque a parte residual varia ao longo do tubo); diâmetro externo no termo
térmico (a eq. (100) define D como interno e o no-slip da §5.5 só fecha com D_in: rms 0,65 °C contra 8,1
com o D_ext da Tabela 21 e 20,5 com o fator 3,42); erro no código da energia (a §5.5 usa o mesmo modelo
sem fator nenhum). Resta um erro de entrada ou de apresentação específico da §5.6: U, m ou D no termo
térmico. Os candidatos redondos mais próximos, 1/D = 3,28 e π = 3,14, ficam 4 % e 8 % abaixo do ótimo.

**Decisão do dono.** Tratar como erro de apresentação de U na Tabela 25: os casos usam U_eff = 3,42 × U
(`generators/condensate.py`, constante `U_FACTOR`; `meta.U_table25` e `meta.U_factor` guardam a
origem). Lote `batches/dissertation.json` recorrido; `nb_5_6_condensate` mostra o ajuste exponencial e
as figuras; teste noturno com rms(T) < 0,5 °C e início da condensação a < 1 km. A hipótese dos °C fica
superada (a opção `cp_temperature` continua disponível, default `kelvin`; o lote
`batches/dissertation_celsius.json` foi apagado com a aprovação do dono). A série 5 das Figuras 39–40
(pontilhada, segmentada por tonalidade) está mal digitalizada e fica fora das verificações.

**O que fica por explicar.** A Tabela 26 (composições na saída do caso 2) implica, pelo balanço material
por componente, 4–7 % molar de líquido, contra 1,8 % na Figura 40 e 1,4 % no cmfsim, e um líquido
com 46 % de C1 contra 31 % no cmfsim; a tabela é internamente inconsistente com a figura e não serve de
prova fina do flash. Perguntar ao autor pelos
ficheiros de entrada da §5.6 continua a ser a única forma de identificar o mecanismo do fator.

## Benefício

Caso 5.6 reexecutável em segundos, com derivadas exatas, corrida gravada com o caso completo e
comparação direta com as figuras digitalizadas (`notebooks_src/thesis/nb_5_6_condensate.py`).
