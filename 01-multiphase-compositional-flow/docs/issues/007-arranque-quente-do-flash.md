# 007 · Arranque quente do flash ao longo de x, Rachford–Rice nos extremos e novos defaults do flash

| | |
|---|---|
| tipo | feature / bugfix |
| área | termodinâmica (flash P-T), solvers (marcha e Newton), desempenho |
| estado | feito (2026-10-15) |
| onde | `cmfsim/thermo/flash.py` (`lnK0`, `lnK` guardado, regra do passo completo no polimento, `polish_iterations`, `rachford_rice`), `cmfsim/thermo/state.py` (`PhaseState.lnK`), `cmfsim/solvers/rk45.py` e `newton.py` (variável auxiliar `aux`), `cmfsim/models/noslip.py` e `mechanistic.py` (`rhs`/`residual` com `aux`), `cmfsim/simulation.py` (flash da entrada e estimativa inicial dentro das funções compiladas), esquema (`fluid.flash.warm_start`) |
| dados | caso 5.6 (gasoduto 8 km, U = 6 W/m²K, T_amb 15 °C), corridas guardadas de `runs/dissertation` e `runs/dp_comparison` |

## Antes

Cada avaliação do lado direito (marcha) ou do resíduo (método 3.2) resolvia o flash a partir de
Wilson (125): ~7 substituições sucessivas (critério 1e-6) mais 2 iterações do polimento de
Newton, em cada um dos sete estágios de cada passo de Runge–Kutta e em cada nó de cada iteração
de Newton. A dissertação faz o mesmo (secção 3.1.2: Wilson em todos os flashes). A issue 006
identificou o arranque quente como o maior ganho de desempenho por fazer, estimando 20 → 2–3
substituições; a estimativa estava errada, ver «O que o flash custa» abaixo.

## Depois

`FlashOptions.warm_start` (default `True`): o ln K convergido do flash anterior é a estimativa
inicial do seguinte.

- **No flash** (`flash_pt(..., lnK0=)`): `lnK0` substitui Wilson quando é finito e não trivial
  (max |ln K| > 1e-3); com arranque quente o teste de estabilidade (opção) não corre, porque o
  flash negativo transporta a solução através da fronteira de fases. O flash devolve `lnK`, o
  valor a guardar para o vizinho: o ln K convergido só onde o flash negativo tem raiz genuína
  (dentro dos limites de Rachford–Rice, com margem, e fora da solução trivial); nos pontos de
  fase única sem raiz, em que a bisseção encosta a um limite, o K é arbitrário e deriva de chamada
  para chamada, e reentrar na região bifásica a partir dele falha, por isso guarda-se Wilson (o
  vizinho arranca a frio, como antes).
- **Passo completo num ponto já convergido** (regra na procura em linha do polimento): o
  polimento amortecido escolhia o melhor de s = 1, 1/2, 1/4 "e nunca pior do que ficar parado";
  um arranque quente que já cai sobre a solução (|G|² ≤ `polish_tol` à entrada) não dava passo
  nenhum e saía com o ln K das substituições, que entra com `stop_gradient`: derivadas **nulas**
  em JAX, e a marcha ficava "ruidosa" (4,5 Pa e 0,056 K de diferença para o arranque frio a
  rtol 1e-6; 1015 passos aceites + 1275 rejeitados em vez de 31 + 21 a rtol 1e-9). Agora, num
  ponto convergido à entrada aplica-se o passo completo, nulo em valor, porque é ele que transporta
  a derivada exata (teorema da função implícita). Um passo final incondicional, testado primeiro,
  custava uma Jacobiana a mais por flash (~9 substituições) e foi substituído por esta regra.
  O polimento também para quando |G|² deixa de descer (piso de arredondamento), em vez de gastar as
  iterações até `polish_iter`.
- **No integrador** (`integrate(..., aux0=)`): o lado direito devolve `(du, aux)`; cada estágio
  do Runge–Kutta arranca do ln K do estágio anterior e o passo aceite guarda o do último estágio
  (um passo rejeitado não o altera).
- **No Newton** (`newton_solve(..., aux0=)`): o resíduo devolve `(F, aux)` com um ln K por nó; a
  Jacobiana colorida é calculada com a auxiliar fixa e a procura em linha atualiza-a com o ponto
  aceite.
- **Na simulação**: o flash da entrada (semente da marcha) e a estimativa inicial do método 3.2
  (um flash por nó) ficam **dentro** das funções compiladas. Em JAX, um flash ansioso fora do
  `jit` custa ~0,7 s por chamada (o `while_loop` é traçado de novo a cada chamada): foi a razão de
  o arranque quente parecer 5× mais lento do que o frio nas primeiras medições, com o mesmo
  número de passos. Fica como armadilha no `CLAUDE.md`.

## Dois achados pelo caminho

1. **Rachford–Rice nos extremos do intervalo** (bugfix, `rachford_rice`): o teste `inside` era
   estrito (`lo < β_Newton < hi`) e, quando a bisseção já tinha colapsado o intervalo sobre a
   raiz, o passo de Newton caía num extremo, era rejeitado e substituído pelo ponto médio, num
   ciclo que terminava onde o limite de iterações apanhasse. Num estado perto do ponto de orvalho
   numpy e JAX davam β diferentes (0,6846 contra 0,8692). Agora o teste é inclusivo e a convergência
   (|f| < tol ou passo nulo) é verificada antes de substituir β; o mesmo estado converge em 4
   iterações e numpy e JAX coincidem a 1e-16. Consequência: na solução trivial (K = 1, f ≡ 0) β
   fica no ponto de partida 0,5, em vez de cair por acaso em −500 como antes, e o rótulo de fase
   única da solução trivial (regra de Kay, issue 002) passou a ser explícito: `two_phase` exclui
   `trivial`.
2. **Defaults do flash**: `polish_iter` 10 → 30 (é só o teto; o polimento para por tolerância ou
   por falta de progresso). `tol` fica em 1e-6: testou-se 1e-3 (as substituições só porem a
   iteração na bacia do Newton) e a marcha fria ficou 25 % mais lenta (166 → 207 ms), porque cada
   iteração de Newton poupada em substituições custa ~9 delas (ver abaixo). Os resultados são
   idênticos aos anteriores: as corridas guardadas de `runs/dissertation` (marcha numpy) e
   `runs/dp_comparison` (método 3.2 em JAX) reproduzem-se a 1e-7 Pa e 1e-11 K.

## O que o flash custa (e porque o arranque quente rende pouco)

Contagem ao longo do perfil convergido do caso 5.6 (33 pontos de saída, 8 bifásicos; `iterations`
e o novo `polish_iterations` do `PhaseState`), com 1 substituição sucessiva = 1 unidade (duas
avaliações de ln φ) e 1 iteração do polimento ≈ nc + 1 direções da Jacobiana + 3 avaliações de G
≈ 9 unidades para o gás de 5 componentes:

| `tol` | `polish_tol` | frio: SS + polimento | quente: SS + polimento |
|---|---|---|---|
| 1e-6 (default) | 1e-26 (default) | 7,0 + 2,2 = 27 un. | 3,7 + 3,0 = 31 un. |
| 1e-6 | 1e-20 | 7,0 + 2,0 = 25 | 3,7 + 2,0 = 22 |
| 1e-3 | 1e-26 | 4,0 + 3,2 = 33 | 1,1 + 3,6 = 34 |
| 1e-10 | 1e-20 | 11,7 + 1,0 = 21 | 9,0 + 1,0 = 18 |
| 1e-12 | 1e-26 | 14,0 + 1,1 = 24 | 11,7 + 1,2 = 22 |

O custo é dominado pelo polimento, não pelas substituições: o arranque quente poupa 3–4
substituições (1 unidade cada) e, nos pontos de saída (250 m entre flashes), gasta uma iteração
de Newton a mais porque as substituições param logo abaixo de `tol`, enquanto a frio o último
passo da iteração linear cai bem abaixo. Entre estágios de Runge–Kutta (dezenas de metros) o K
anterior é muito mais próximo e o ganho é o da tabela de tempos. As alavancas que ficam
identificadas, **não feitas** (mudam o default ou o resultado a 1e-10 relativo, decisão do dono):
`polish_tol` 1e-26 → 1e-20 poupa ~1 iteração de Newton por flash (−20–30 %); as 3 avaliações da
procura em linha são desperdício perto da convergência (−20 %); a Jacobiana por `jvp` em nc + 1
direções podia vir de uma só chamada vetorizada.

## Medições (2026-10-15, CPU Apple Silicon, caso 5.6 com U = 6 W/m²K e T_amb 15 °C, 17 pontos de saída)

| | frio | quente | identidade (quente − frio) |
|---|---|---|---|
| marcha JAX, rtol 1e-6, defaults anteriores (`polish_iter` 10) | 122 ms | — | 0 Pa (defaults anteriores − atuais) |
| marcha JAX, rtol 1e-6 (21 + 4 passos) | 135 ms | 139 ms | 5,6e-9 Pa, 7e-11 K |
| marcha JAX, rtol 1e-9 (31 + 21 passos) | 226 ms | 242 ms | 1,9e-9 Pa, 4e-11 K |
| método 3.2 JAX, 32 nós, 2 Newton (inclui a marcha inicial) | 0,23 s | 0,24 s | 3,7e-9 Pa |
| `march_many` N = 64 (ms por caso) | 37 | 30 | 3,7e-9 Pa |

Tempos mínimos de 5 (marcha) ou 3 (fd, lote) chamadas compiladas, máquina sem outra carga. Numa
segunda corrida nas mesmas condições a marcha deu 207/209 ms (rtol 1e-6) e 454/341 ms (rtol 1e-9),
e o lote 54/44 ms por caso: na marcha isolada o ganho do arranque quente fica dentro do ruído de
medição (±10 %); no `march_many` (o caminho dos dados de treino) é consistente, ~20 %.

## Benefício

- Resultado idêntico ao arranque frio (a precisão de máquina); o ganho em tempo é o da tabela:
  modesto, porque o custo do flash está no polimento (ver acima). `warm_start` fica ligado por
  omissão por nunca ser mais lento.
- Derivadas do flash com arranque quente iguais às do arranque frio (`tests/test_flash_warm.py`).
- Rachford–Rice robusto nos extremos e idêntico em numpy e JAX (200 sistemas aleatórios a 1e-13).
