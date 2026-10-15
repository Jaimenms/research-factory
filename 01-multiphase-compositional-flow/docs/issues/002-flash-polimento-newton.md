# 002 · Polimento de Newton no flash e derivadas pelo teorema da função implícita

| | |
|---|---|
| tipo | feature |
| área | termodinâmica (flash P-T) |
| estado | feito (2026-10-15) |
| onde | `cmfsim/thermo/flash.py` (`FlashOptions.polish_iter`, `_G`, `_jacobian_dirs`) |
| dados | fluido da Tabela 6; oráculo `thermo` (FlashVL com PR) |

## Antes

A dissertação resolve o flash por substituições sucessivas (124) a partir de Wilson (125) até o
critério (123). A convergência é linear e lenta perto da fronteira de fases, e as derivadas da
fração vaporizada em P e T, que a eq. (25) e o método 3.1 precisam, são feitas por diferenças
finitas de 1.ª ordem com dois flashes adicionais (169–170, Figura 10).

## Depois

Com `polish_iter > 0` (default 2), o resultado das substituições entra com `stop_gradient` e
seguem-se iterações de Newton no sistema completo G(ln K, β) = [ln K − ln φ_L + ln φ_V; Σ(y − x)]
com Jacobiana por `jvp` (JAX) ou diferenças centrais (numpy), só nos pontos dentro dos limites
do flash negativo. A solução fica a precisão de máquina e, em JAX, `jacfwd` do flash devolve
d(β, x, y)/d(P, T, z) exatas pelo teorema da função implícita, com um flash em vez de três.
Com `polish_iter = 0` fica a dissertação tal e qual.

O polimento é um Newton amortecido (melhor de s = 1, 1/2, 1/4 em |G|²) em laço até
`polish_tol` (1e-26) ou `polish_iter` (10), aplicado só onde o flash negativo tem raiz genuína.

## Benefício (medido em 2026-10-15, fluido da Tabela 19 com C7+, 8 estados)

- Substituições sucessivas a 1e-12 sem polimento: até 200 iterações sem convergir em dois estados
  (β 0,927 contra 0,856 do oráculo, resíduo 1,5e-4). Com polimento: resíduo 1e-28 e β igual ao
  oráculo `thermo` a 7 algarismos em todos os estados bifásicos.
- Default (substituições a 1e-6 + polimento): 3–9 iterações (22 no pior) e 28 ms contra 77 ms com
  a tolerância 1e-12; resultado idêntico a precisão de máquina.
- Derivadas por AD (`jacfwd` de `phase_state`) iguais às diferenças centrais a 1e-8 relativo
  (`tests/test_thermo_parity.py`); paridade numpy × JAX 1e-15; 0,7 ms por avaliação compilada de 5 pontos.
- Caso em que o oráculo falha e nós não: fluido da Tabela 19 a 300 K e 150 bar, a `thermo` dá
  líquido único e o flash negativo dá β = 0,645 com G/RT −4,146 contra −4,054 (TPD −0,17).
