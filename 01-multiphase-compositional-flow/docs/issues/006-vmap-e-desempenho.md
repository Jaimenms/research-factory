# 006 · Parâmetros como argumentos, `march_many` (jit + vmap) e o que custa cada avaliação

| | |
|---|---|
| tipo | feature / estudo |
| área | desempenho (geração de dados para redes neuronais) |
| estado | feito (2026-10-15); otimizações seguintes identificadas |
| onde | `cmfsim/simulation.py` (`_COMPILED`, `march_many`), `cmfsim/models/*.py` (`params`), `cmfsim/solvers/rk45.py` (estágios em laço) |
| dados | caso 5.6 (gasoduto 8 km) com variações de U, T_in, m, P_in |

## Antes

Modelos com o tubo e a vazão como atributos: cada caso novo forçava um novo traçado e uma
nova compilação em JAX (34 s por caso de marcha; 60–300 s com os sete estágios desenrolados).

## Depois

- Tubo, entrada e malha são **argumentos** (`params`) das funções compiladas; `_COMPILED` guarda a
  função por estrutura (fluido, opções, malha). Os 200 casos do Apêndice A correm com uma
  compilação por modelo.
- Estágios do Runge–Kutta num `fori_loop`: o lado direito (com o flash) é traçado uma vez
  (compilação 298 s → 34 s).
- `march_many(case, variations)`: `jit(vmap(integrate))` sobre um lote de casos com a mesma
  estrutura; idêntico ao caso individual (0,0 Pa).

## Medições (2026-10-15, CPU Apple Silicon, caso 5.6 com 17 pontos de saída, rtol 1e-6)

| | tempo |
|---|---|
| marcha individual compilada (42 passos) | 175 ms |
| `march_many` N = 64, 2.ª chamada | 26 s (0,41 s/caso) |
| `march_many` N = 256, 2.ª chamada | 151 s (0,59 s/caso) |
| método 3.2 (fd) em JAX, 49 nós, 4 Newton | ~3 s depois de compilar |
| flash de 5 pontos compilado | 0,7 ms |

Em CPU o `vmap` não ganha por caso (os laços `while` correm até o lanço mais lento acabar e o
custo é dominado pelo flash dentro de cada avaliação do lado direito: 7 estágios × ~20
substituições sucessivas + polimento + jvp); ganha em GPU/TPU e em conveniência. Para gerar
10⁴ casos em CPU: ~1 h.

## Próximos ganhos (não feitos)

1. **Arranque quente do flash** ao longo de x (K do passo anterior em vez de Wilson): 20 → 2–3
   substituições; é o maior ganho (×3–5 na marcha). **Feito na issue 007** (com `tol` 1e-3 o
   arranque frio já só precisa de 4 substituições, e o quente de 1; o ganho medido está lá).
2. Tolerância do flash nas iterações de Newton do método 3.2 relaxada nas primeiras iterações.
3. `vmap` com `lax.scan` de passo fixo (sem `while`) para lotes homogéneos.
4. Pós-processamento (flash e fechamentos nos pontos de saída) também compilado.
