# Questões abertas, decisões pedidas e dados em falta

*Parte do [estudo](README.md) de 2026-10-15. O bloco A foi respondido pelo dono em
2026-10-15 (tabela abaixo); os blocos B e C ficam sem acesso ao autor e resolvem-se por
estudo numérico, cada decisão registada como issue.*

## Respostas do dono (2026-10-15)

| n.º | resposta | consequência |
|---|---|---|
| A1 | **não** há código | reprodução de nível B (Apêndice A, ~1 %) e C (figuras digitalizadas); nível A só entre versões próprias |
| A2 | **não** há dados do HYSYS além do Apêndice A | termodinâmica validada contra referência independente; §5.5 contra as figuras |
| A3 | `cmfsim` | schema `cmfsim-case/1` |
| A4 | monorepo `research-factory/` | `git init` na raiz de `research-factory/` na fase 0; o projeto vive em `01-multiphase-compositional-flow/` |
| A5 | **integrador próprio desde o início** | `solvers/march.py` sem SciPy: RK encaixado adaptativo (e BDF próprio para rigidez) sobre o backend, compilável em JAX; a fidelidade à dissertação vem da tolerância apertada, não do LSODA |
| A6 | referência termodinâmica independente | `thermo` (Caleb Bell) e CoolProp só como oráculos de teste |
| A7 | **digitalizar todas as figuras** e guardar JSON em `references/` | feito em 2026-10-15: `references/figures/` (ver `references/figures/README.md`) |
| A8 | reutilizar o mfsim por cópia | módulos backend-agnósticos copiados com a origem no cabeçalho |
| A9 | tubos em série, de preferência **um único tubo com parâmetros que variam axialmente** | `pipe.profile`: pontos de quebra em x com D, ângulo, rugosidade, U, T_amb constantes por troço ([04](04-plano-de-reproducao.md#o-caso-em-json)) |
| A10 | sim | estudo da coluna Beggs & Brill com o `drift_flux_bb` copiado |
| A11 | **não** se pretende aproveitar para o doutorado | o módulo `thermo/` não precisa de interface para o mfsim; a secção «O que o mfsim ganha» de [02](02-mapeamento-com-o-mfsim.md) fica informativa |
| A12 | sim | termo cinético e pressão interfacial como opções com bandeira, default desligado |
| B, C | sem acesso ao autor | cada item do bloco B fica com a hipótese da coluna «se não houver resposta», como issue, e as gralhas decidem-se pelos números do Apêndice A |

## A. Decisões do dono

| n.º | decisão | recomendação |
|---|---|---|
| A1 | **Existe o código Python do Gustavo?** (não há repositório citado) | pedir; com código, a reprodução sobe para nível A (bit a bit) como no doutorado e as ambiguidades do bloco B desaparecem |
| A2 | **Há resultados do HYSYS além do Apêndice A** (os 159 pontos de 5.1–5.2, os perfis Tulsa e Beggs & Brill de 5.5)? | pedir; sem eles, a termodinâmica valida-se contra outra implementação de PR e 5.5 contra figuras digitalizadas |
| A3 | nome do pacote e do schema | `cmfsim` / `cmfsim-case/1` (alternativas: `mfcomp`, `ssflow`) |
| A4 | repositório git próprio em `01-multiphase-compositional-flow/` ou monorepo `research-factory/` | repositório próprio (como o `doutorado`); `src/` vazia vira `cmfsim/` |
| A5 | caminho fiel do método 3.1: LSODA do SciPy (o que a dissertação usou) ou integrador próprio desde o início | LSODA como default fiel e integrador em JAX como opção da fase 5; a regra «solver próprio» aplica-se ao que é conteúdo da dissertação (flash, holdup, Newton) |
| A6 | referência termodinâmica independente | biblioteca `thermo` (flash PR com estabilidade) para as misturas e CoolProp para os puros; só como oráculo de teste, nunca no simulador |
| A7 | digitalizar figuras? quais? | sim: Figuras 38–40 (5.6), 32–34 (5.5), 22–23 (5.3, só para conferência); guardar em `dissertation/data/figures/` com proveniência |
| A8 | reutilizar código do mfsim por cópia ou por dependência | cópia dos módulos backend-agnósticos (`backend/`, `friction`, `geometry`, sigmoide, Newton), sem dependência entre repositórios; registar a origem no cabeçalho |
| A9 | tubos em série ou grafo desde o início | série (basta para a dissertação); grafo numa fase posterior, reutilizando `network/` do mfsim |
| A10 | reproduzir também a coluna Beggs & Brill do Apêndice A com o `drift_flux_bb` do mfsim | sim, como estudo opcional da fase 4 (mede o que o HYSYS muda no B&B) |
| A11 | o módulo `thermo/` nasce já como pacote reutilizável pelo mfsim (interface `props(P, T, z)` com derivadas) | sim; é o retorno principal para o doutorado |
| A12 | termo cinético na energia e pressão interfacial como opções | sim, com bandeira e default desligado |

## B. Dados em falta ou ambíguos (perguntar ao autor; senão, assumir e registar)

| n.º | item | onde | se não houver resposta |
|---|---|---|---|
| B1 | coeficientes A–D do Cp de gás ideal do pseudocomponente C7+ | Tabela 8 só tem os puros; Tabela 20 dá H_form, T_b, T_c, P_c, ω, MW 560 g/mol, ρ 950 kg/m³ | estimar por Lee–Kesler ou Kesler–Lee a partir de T_b e densidade; registar como issue. Feito em 2026-10-15: Kesler–Lee com K_W = 12,08 dá Cp/R = 0,733 + 0,38776 T − 1,521e-4 T², 8 % abaixo da extrapolação da issue 001, efeito ≤ 1 °C na §5.5; não adotado |
| B2 | parâmetros de interação binária k_ij | não dados (eq. 112) | zero |
| B3 | número de carbonos efetivo (Allan & Teja) para CO₂, N₂, H₂S e C7+ | §2.5.2 só define para n-alcanos | correlacionar com MW; registar |
| B4 | ΔP e ΔT das derivadas numéricas (169–170) e o lado (para a frente ou para trás) | não ditos; Figura 10 diz P + dP | 1 kPa e 0,01 K; estudo de sensibilidade |
| B5 | número de pontos n e tolerâncias usados nos resultados de 5.4 e 5.5 pelo método 3.2 | só 12/30/78 em 5.8 | estudo de malha; comparar os três com o Apêndice A |
| B6 | base da «fração vaporizada» em (25): mássica, molar ou volumétrica | lista de símbolos diz volumétrica; a dimensão exige mássica | mássica; opção para as outras. A Figura 40 (§5.6) é **molar**: com U_eff o cmfsim segue-a a 0,004 em β, contra 0,02 em massa e 0,015 em volume (2026-10-15, issue 004) |
| B7 | forma efetiva no código de (29), (30)/(160), (46), (54), (84), (89), (126), (147) | gralhas listadas em [03](03-avaliacao-e-interpretacao.md) | cada uma com as duas formas como opção; decidir pelo Apêndice A |
| B8 | estimativa inicial do método 3.2 | «necessária em todos os pontos» | perfil plano em P e T com vazões por fase do flash de entrada; opção `march_noslip` |
| B9 | unidades de η_L, A, B, C em Allan & Teja (141) | não ditas | consultar o artigo de 1991 |
| B10 | ρ_c do metano e regra de mistura de α na correlação de Pedersen (128–136) | só a Tabela 2 | consultar Pedersen et al. 1984/1987 |
| B11 | atrito interfacial no filme do intermitente | «similares ao regime estratificado» (p. 71) | f_i = 0,0142 |
| B12 | `edge_order` do `numpy.gradient` | Tabela 4 diz 2.ª ordem nos extremos; o default do NumPy é 1 | opção; default 1 |
| B13 | como o flash foi «desabilitado» no teste de 5 s → 0,5 s | p. 158 | propriedades de fase única congeladas |
| B14 | T_amb do poço (40 °C) constante ou gradiente geotérmico | Tabela 21 | constante |
| B15 | fator de atrito no modelo de não escorregamento (mistura homogénea: Churchill com ρ_M e μ_M?) | não dito | Churchill com propriedades da mistura ponderadas por volume |
| B16 | valores de U da Tabela 25 (§5.6) | os perfis da Figura 39 exigem 3,42× os U impressos, com o mesmo fator nos seis casos; a §5.5 não precisa de fator | **decidido (2026-10-15)**: erro de apresentação de U; casos com U_eff = 3,42 × U (`U_FACTOR` em `generators/condensate.py`, issue 004); pedir ao autor os ficheiros de entrada |

## C. Pontos a confirmar nos resultados

- Tabela 26: o título diz «caso 1», o texto diz «caso 2»; pela Figura 40 só os casos 2, 3 e
  6 formam condensado, logo é o caso 2.
- Figura 30: o ponto «5» vertical está na fronteira anular/intermitente; saber com que
  velocidades superficiais exatas (Apêndice A dá D, P, m; o fluido dá ρ_k).
- Apêndice A: confirmar na extração que a coluna «Dissertação» é a do modelo com
  escorregamento pelo método 3.2 (§5.4 não diz qual método; §5.8 sugere que o modelo com
  escorregamento só correu por 3.2).
- §5.5: qual n e qual método produziram os perfis com escorregamento.

## D. Riscos de decisão

- Se A1 for «não», aceitar que a reprodução é de nível B/C e que parte das ambiguidades
  fica decidida por «o que reproduz o Apêndice A», o que pode não ser único.
- Se B1 não se resolver, o perfil de temperatura do caso 5.5 só se compara
  qualitativamente.
- A escolha A5 (LSODA) cria uma dependência do SciPy no caminho fiel; aceitável porque é
  exatamente o que a dissertação fez e fica isolado em `solvers/march.py`.
