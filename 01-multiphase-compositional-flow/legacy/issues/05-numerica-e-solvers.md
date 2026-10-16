# L05 · Numérica: integradores, diferenças finitas e malhas

| | |
|---|---|
| família | discretização, derivadas, integradores e solvers não lineares |
| código | [`app/simulator/simulator.py`](../masters_code/app/simulator/simulator.py), [`app/simulator/simulator_slip.py`](../masters_code/app/simulator/simulator_slip.py) |
| verificado | leitura do código e corridas de L01–L04 |
| responde a | passos das derivadas e tolerâncias (B4 de docs/05), malha do Apêndice A, integrador das amostras do §5.7 |

## L05.1 Modelo de não escorregamento (marcha em x)

- **Integrador:** `solve_integrate` chama `scipy.integrate.solve_ivp(system_ivp, …)` com LSODA,
  `rtol = 1e-4` e atol por omissão (1e-6), e 100 pontos de saída
  ([simulator.py:773-782](../masters_code/app/simulator/simulator.py#L773-L782)). As amostras do
  Apêndice B usam RK45 ([`test_generate_samples_sim.py`](../masters_code/tests/test_generate_samples_sim.py)).
- **Derivadas das propriedades em P e T:** são diferenças **atrasadas** com ε_P = 0,1 bar e
  ε_T = 0,1 K. Cada avaliação do lado direito faz um flash em três estados, (P, T), (P − ε_P, T) e
  (P, T − ε_T) ([:294-299](../masters_code/app/simulator/simulator.py#L294-L299)).
- **Arranque do flash:** a cada chamada, do K de Wilson, sem arranque quente. A variável global
  `global_variable_K0` existe mas não é usada.
- **Saída:** `force_plot_results` volta a correr o flash ponto a ponto. A «fração vaporizada» das
  figuras é a fração volumétrica do no-slip.

O cmfsim, pelo contrário, usa J = ∂Φ/∂u por diferenciação automática, Dormand–Prince próprio com
rtol 1e-7 e arranque quente (issues 004 e 007). As diferenças atrasadas de 0,1 bar e 0,1 K dão erros
de truncatura de primeira ordem nas derivadas de ρ e h. Perto do orvalho, combinadas com L02.2, criam
descontinuidades no lado direito. É a causa provável de a §5.5 com perímetro externo não terminar (L01.4).

## L05.2 Modelo com escorregamento (diferenças finitas + Powell)

- **Sistema:** `solve_single_slip` monta T, P, ṁ_L e ṁ_V nos nós interiores e resolve com
  `scipy.optimize.root(method='hybr')`, Powell com Jacobiana por diferenças
  ([simulator_slip.py:494-560](../masters_code/app/simulator/simulator_slip.py#L494-L560)).
- **Derivadas em x:** `np.gradient`, central de segunda ordem no interior e de primeira ordem nas
  pontas, com Δx constante.
- **Partição de massa:** o resíduo de massa usa a derivada da fração mássica de vapor de equilíbrio,
  ṁ dw_eq/dx.
- **Momento:** inclui a aceleração (ρ α v² de cada fase) e a pressão. O resíduo de energia inclui a
  energia cinética.
- **Malha:**
  - Apêndice A: 3 nós em 100 m (Δx = 50 m), com estimativa inicial linear a P − 4 bar e T − 1 K
    (horizontal) ou P − 6 bar e T − 4 K (vertical);
  - caso offshore com escorregamento: 4 nós no poço e 10 na linha;
  - a pressão de saída é a do último nó.
- **Saída de diagnóstico:** cada avaliação imprime a soma dos resíduos, os padrões e o holdup.

Com 3 nós, o ΔP do Apêndice A é praticamente o gradiente local na entrada × 100 m. O cmfsim usa 31
nós e Newton com Jacobiana exata, mas no Apêndice A os perfis são quase lineares e a malha não pesa.

## L05.3 Pontos frágeis observados nas corridas

- **Raiz da cúbica:** `cubic_equation` não tem ramo para zero raízes reais. Quando A ou B vêm NaN
  (H2S sem entalpia, L02.4) ou em estados extremos, falha com «local variable 'Z' referenced before
  assignment». Isto aconteceu em 2 de 10 linhas do Apêndice B.
- **Raízes de holdup:** são procuradas por `optimize.root` com várias estimativas e `except: continue`.
  Se nenhuma converge, o código segue com o último resultado não convergido.
- **Flash:** ver L02.2 (solução trivial) e L02.3 (fecho com memória não inicializada).

## Consequência

Nada a mudar no cmfsim: a numérica já foi substituída (issues 004, 005, 007, 011). Os valores ficam
registados para a resposta à B4 de docs/05: ε_P = 0,1 bar, ε_T = 0,1 K, diferenças atrasadas, LSODA
com rtol 1e-4, e 3 nós no Apêndice A.
