# 010 · Substituto de perfil (posição axial como entrada) e PINN com o resíduo do modelo

| | |
|---|---|
| tipo | feature / estudo |
| área | modelo substituto (cap. 4, §5.7) para além da dissertação |
| estado | feito (2026-10-15) |
| onde | `surrogate/profile.py`, `surrogate/pinn.py`, `surrogate/pipeline.py` (passos `profiles`, `profile_models`, `pinn`), `surrogate/dataset.py` (`overrides`), `models/surrogate/*_profile.json`, `*_pinn.json`, `notebooks_src/studies/nb_surrogate_profile_pinn.py`, `tests/test_surrogate.py` |
| dados | `runs/surrogate/{well,line,riser}_profile.npz` (300 perfis por trecho, 51 pontos), `runs/surrogate/pinn_study.json` |

## Antes

O substituto da §5.7 (issue 009) devolve só a saída do trecho a partir de (P_in, T_in, m). O dono pediu
duas extensões: a rede receber a posição axial e entregar o perfil, e uma rede com informação da
física (PINN), em que o resíduo das equações entra na perda. O simulador continua em revisão por
outra sessão e não foi tocado.

## Depois

- **Perfil** (`surrogate.profile`): entradas (P_in, T_in, m, s = x/L); a rede aprende a queda por unidade
  de s, g = (y_in − y(x))/s, e o perfil é y(x) = y_in − s·g, pelo que a condição de entrada é exata por
  construção. Massa de dados com 51 pontos por perfil (`surrogate.dataset --set outputs.dx_out=…`),
  treino com o Adam e o MLP da issue 009 sobre 210 perfis × 50 pontos; `ProfileMLP.predict` dá o perfil
  em qualquer s. Modelos em `models/surrogate/{well,line,riser}_profile.json`.
- **PINN** (`surrogate.pinn`): perda = EQM dos dados + λ·EQM do resíduo escalado r = (du/dx − f(x, u))/esc
  em pontos de colocação (P_in, T_in, m, s) por hipercubo latino, sem os simular; f é o lado direito do
  modelo de não escorregamento (`NoSlipModel.rhs`, fluxos conservados, flash dentro), chamado sem alterar
  o simulador; du/dx vem de um `jvp` da rede em s. **Gradiente sem modo inverso pelo flash**: o flash corre
  num `while_loop`, e tanto a retropropagação por ele como um segundo nível de AD por cima da
  linearização interna do `rhs` esgotam a memória na compilação (processo morto, código 137, mesmo com
  10 pontos). A solução: f e ∂f/∂u entram como valores (f pelo simulador, ∂f/∂u por diferenças avançadas
  com dP = 1 kPa e dT = 0,01 K, como as eqs. 169–170 da dissertação) e o gradiente em θ vem da perda
  auxiliar 2·mean(r₀/esc² · (du/dx − J₀ u)), cujo gradiente é exatamente o de mean((r/esc)²). O passo
  compila em ~3 min e custa ≈ 20 ms por ponto de colocação por época (três avaliações do lado direito).
- **Estudo** (`surrogate.pipeline --steps pinn`): por trecho, modelo só de dados com 20 perfis, PINN a
  partir desses pesos e PINN do zero, contra o modelo com 210 perfis; erro nos 90 perfis de teste e RMS do
  resíduo nos pontos de colocação (`pinn_study.json`); notebook `notebooks/studies/nb_surrogate_profile_pinn.ipynb`.

## Achados

### Perfil com todos os dados (210 perfis de treino, 90 de teste, 51 pontos)

| trecho | RMSE P [bar] | RMSE T [K] | máx |ΔP| | máx |ΔT| | saída RMSE P / T |
|---|---|---|---|---|---|
| poço | 0,019 | 0,150 | 0,17 | 1,96 | 0,046 / 0,215 |
| linha | 0,110 | 0,239 | 0,57 | 4,60 | 0,186 / 0,510 |
| riser | 0,015 | 0,083 | 0,21 | 1,30 | 0,025 / 0,138 |

O perfil inteiro sai com erro da ordem do do substituto de saída da §5.7 (0,1–0,3 bar, 0,3–0,6 K), com
1 282 parâmetros em vez de 122. Com 20 perfis de treino o erro sobe para 0,18 / 1,3 (poço), 0,83 / 3,1
(linha) e 0,11 / 0,22 (riser) bar / K.

### PINN com dados escassos (20 perfis de treino, 60 pontos de colocação, λ = 1, 300 épocas)

| trecho | modelo | RMSE P [bar] | RMSE T [K] | máx |ΔP| / |ΔT| | RMS do resíduo |
|---|---|---|---|---|---|
| poço | só dados, 210 perfis | 0,019 | 0,150 | 0,17 / 1,96 | 0,046 |
| poço | só dados, 20 perfis | 0,182 | 1,320 | 1,07 / 10,4 | 0,139 |
| poço | PINN, 20 perfis (a partir do modelo de dados) | 0,147 | 0,614 | 1,11 / 6,43 | 0,006 |
| poço | PINN, 20 perfis (do zero) | 0,156 | 0,655 | 0,88 / 6,91 | 0,042 |
| linha | só dados, 210 perfis | 0,110 | 0,239 | 0,57 / 4,60 | 0,030 |
| linha | só dados, 20 perfis | 0,832 | 3,083 | 12,2 / 12,2 | 0,369 |
| linha | PINN, 20 perfis (a partir do modelo de dados) | 0,171 | 1,134 | 3,00 / 10,7 | 0,010 |
| linha | PINN, 20 perfis (do zero) | 0,212 | 0,857 | 2,98 / 7,99 | 0,046 |
| riser | só dados, 210 perfis | 0,015 | 0,083 | 0,21 / 1,30 | 0,005 |
| riser | só dados, 20 perfis | 0,106 | 0,223 | 0,98 / 2,20 | 0,082 |
| riser | PINN, 20 perfis (a partir do modelo de dados) | 0,083 | 0,156 | 0,73 / 2,18 | 0,006 |
| riser | PINN, 20 perfis (do zero) | 0,077 | 0,145 | 0,62 / 2,04 | 0,008 |

Com 20 perfis, o resíduo físico reduz o erro de T para metade no poço e para um terço na linha (de 3,1
para 1,1 K) e o erro de P na linha para um quinto, e baixa o RMS do resíduo das equações 10–40×, abaixo
do do modelo treinado com 210 perfis (que nunca foi obrigado a satisfazer as equações). O erro que resta
concentra-se na parte bifásica dos perfis. Custo: 200–540 s por trecho (duas redes, compilação incluída).

### Sobreajuste, viés e divisão treino/teste (pedido do dono)

Com dados sintéticos baratos, `surrogate/validation.py` mede: validação cruzada 5-fold do substituto de
saída, curvas de aprendizagem (erro de teste e de treino contra 10–210 perfis de treino, 3 repetições:
a diferença treino–teste é o sobreajuste, o desvio entre repetições a variância), o viés (erro médio com
sinal) e três estratégias de divisão para os perfis: por perfil (condições de entrada novas), por
posição (30 % dos pontos ao longo de x nunca vistos, todas as condições: interpolação em x) e por bloco
(teste = perfis com P_in no topo da faixa: extrapolação). O resíduo físico é ainda avaliado num conjunto
de colocação nunca usado no treino. Resultados em `runs/surrogate/validation.json`, figura
`surrogate_validation` e tabelas `tab_validation`/`tab_residual` do artigo, e notebook do estudo.

Resultados (2026-10-16, `validation.json`):

| medida | poço | linha | riser |
|---|---|---|---|
| saída, 5-fold: RMSE teste P / T | 0,14 ± 0,05 bar / 0,38 ± 0,21 K | 0,55 ± 0,23 / 0,42 ± 0,07 | 0,07 ± 0,02 / 0,21 ± 0,10 |
| saída, 5-fold: diferença teste − treino | +0,04 / +0,17 | +0,19 / +0,12 | +0,01 / +0,07 |
| saída, 5-fold: viés | +0,006 / +0,02 | +0,07 / −0,01 | −0,002 / −0,01 |
| saída, 10 perfis: teste (treino) | 0,98 (0,00) / 2,5 (0,00) | 2,7 (0,00) / 4,0 (0,00) | 2,1 (0,00) / 1,7 (0,00) |
| perfil, divisão por perfil, 20 → 210 | 0,19 → 0,03 / 1,17 → 0,15 | 0,78 → 0,14 / 1,88 → 0,27 | 0,08 → 0,017 / 0,27 → 0,058 |
| perfil, divisão por posição, 10 → 210 | 0,02 → 0,03 / 0,04 → 0,04 | 0,06 → 0,06 / 0,04 → 0,13 | 0,002 → 0,007 / 0,001 → 0,009 |
| perfil, divisão por bloco, 20 → 210 (viés P) | 0,41 → 0,04 / 1,06 → 0,07 (+0,04 → 0,00) | 0,83 → 0,11 / 2,36 → 0,32 (+0,31 → −0,02) | 0,36 → 0,09 / 0,26 → 0,03 (+0,22 → +0,04) |
| resíduo em colocação nunca vista: 210 dados / 20 dados / PINN | 0,035 / 0,12 / 0,058 | 0,030 / 0,27 / 0,118 | 0,009 / 0,08 / 0,028 |

Leituras: (1) com 10 amostras o substituto de saída interpola o treino (erro zero) e erra 1–2,7 bar no
teste: sobreajuste puro; a diferença treino–teste fecha com o número de amostras. (2) A dispersão entre
dobras é 30–50 % da média: um único split (Tabela 32 da dissertação e a nossa) sobrestima a precisão
da estimativa. (3) A divisão por posição dá erros iguais aos de treino mesmo com 10 perfis: interpolar
em x dentro de perfis conhecidos é trivial para uma rede com s na entrada, e um baralhar por linhas
reportaria um erro 10–50× menor do que o real. A divisão certa é por perfil. (4) A divisão por bloco
(extrapolar em P_in) custa 2–5× no erro de P do riser e expõe um viés sistemático que as divisões
aleatórias escondem. (5) O resíduo do PINN em colocação nunca vista é 5–10× o do seu próprio conjunto:
o análogo do sobreajuste para o termo físico; remedeia-se com mais pontos de colocação (não custam
simulação) e, mesmo assim, fica 2–4× abaixo do modelo só de dados com os mesmos 20 perfis.

## Benefício

Um modelo empírico que devolve o perfil inteiro (e não só a saída) a partir das mesmas três entradas,
com a entrada exata; e um caminho para treinar com o resíduo do modelo físico sem retropropagar pelo
flash, reutilizando o lado direito do simulador tal como está.
