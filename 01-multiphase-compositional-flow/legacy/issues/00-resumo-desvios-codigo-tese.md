# Resumo: desvios entre o código original e a dissertação

*Síntese das famílias L01–L06 desta pasta (2026-10-16). Compara o que o texto da dissertação descreve
com o que o código do autor faz ([`../masters_code`](../masters_code), commit `d0b91fb`). Cada desvio foi
lido no código, e os que mudam números foram medidos a correr o próprio código ([`../scripts`](../scripts)).
O detalhe e as referências às linhas de código estão nos ficheiros L0x indicados.*

## Em poucas linhas

Nove desvios mudam resultados publicados:

- três erros de formulação: o coeficiente de fugacidade, o atrito de parede e a viscosidade do C7+;
- dois erros de entrada nos scripts: o perímetro térmico e o diâmetro externo passado como espessura;
- quatro escolhas não documentadas: os critérios de padrão em sigmoide, o atrito do modelo «no-slip»
  por padrão, a entalpia do HYSYS e o flash sem teste de estabilidade.

Alguns compensam-se: no Apêndice A horizontal, o atrito 4× e a viscosidade de gás deixavam a dissertação
perto do HYSYS. Outros produzem os achados que ficaram por explicar enquanto não havia código:

- o fator 3,42 em U na §5.6;
- o salto da Figura 40;
- o holdup 0,85× da §5.5;
- a linha do Apêndice B mais fria.

Com o código, todos ficam reproduzidos e explicados. O cmfsim não herda nenhum destes desvios, porque
segue o texto, ou a física quando o texto é omisso.

## Desvios que mudam resultados

| # | o texto diz | o código faz | efeito medido | resultados afetados | ref. |
|---|---|---|---|---|---|
| 1 | ln φ de Peng–Robinson padrão (eq. 121) | a_ij = a_i·a_j (raiz elevada ao quadrado) e sinal **+** no termo de atração | K errados. A 116 bar: ln φ do C7+ −33,8 contra −29,9; β = 0,88 contra 0,75. Com ln φ padrão, o flash do autor iguala o cmfsim e a `thermo` | §5.5 holdup no-slip 0,85× (Figura 34); propriedades e envelope das Figuras 13–21; Apêndice B. Pouco efeito no binário C1/C7+ (Apêndice A) e no gás rico da §5.6 | [L02.1](02-termodinamica-e-flash.md) |
| 2 | atrito de Darcy na forma impressa | fator de Churchill (Darcy) em tensões τ = f ρ v²/2 (Fanning): 4× o atrito em todos os padrões do modelo com escorregamento menos o 9 | ΔP horizontal do Apêndice A exatamente 4× | §5.4 (Apêndice A horizontal), §5.5 mecanicista. Coincide com o texto impresso: o erro está nos dois | [L04.1](04-hidrodinamica-mecanicista.md) |
| 3 | Allan & Teja com Nc_m = Σ x_i Nc_i (eq. 137) | Nc lido do **nome**: só contam nomes com «n», pelo último carácter. C1, C2, C3, CO2, N2, H2S e o C7+ (`C20pseud`) contam 0 | Apêndice A: μ_L = 0,0134 mPa·s em todos os casos, um valor de gás (cmfsim 0,7–3 mPa·s) | §5.4 (compensa em parte o #2), §5.5 mecanicista | [L03.2](03-propriedades-de-transporte.md) |
| 4 | U π D com D interno (eq. 100) | U π (D + 2e): perímetro externo, com a espessura da Tabela 21 | 1,10× no poço, 1,83× na linha e no riser. A linha do Apêndice B arrefece 1,4× mais; o cmfsim com U × 1,833 reproduz-a a −0,37 K | Apêndice B e §5.7 | [L01.1, L01.3](01-termico.md) |
| 5 | Tabelas 24–25: D = 12 in, U = 1, 3, 6 W/(m² K) | o script da §5.6 passa o D_ext (0,365 m) **como espessura**: perímetro π (0,3048 + 0,730) = 3,395 × π D_in | perda de calor 3,4× a da Tabela 25. O código reproduz a Figura 39 a 0,02–0,06 °C | §5.6, Figuras 38–40 (o cmfsim usa U × 3,42, decisão do dono) | [L01.2](01-termico.md) |
| 6 | sigmoides nas transições de padrão, sem ganho documentado | ganho 10 e decisão a 1 − 1e-6: cada fronteira desloca-se 1,38 unidades, mais um critério extra (v_SL − 0,09 v_SG − 4,2) | Apêndice A horizontal sempre estratificado (92) ou bolhas forçadas (8); nunca intermitente nem anular | §5.4, §5.5 (padrões da Figura 37) | [L04.2](04-hidrodinamica-mecanicista.md) |
| 7 | não diz como é o atrito do no-slip (B15) | atrito **por padrão**, com as frações sem escorregamento: estratificado τ_L/D_hL + τ_G/D_hG, anular só o filme, intermitente com o perímetro das bolhas | a linha da §5.5 fica a 2 % do critério de Kelvin–Helmholtz, e o atrito salta 2× entre estratificado e anular. Com «>» (código atual) perde 4,8 bar a mais; com sigmoide (como na tese) reproduz a Figura 32 a 0,6 bar rms | §5.5 pressão no-slip (Figura 32) | [L04.6](04-hidrodinamica-mecanicista.md) |
| 8 | Cp de gás ideal de Smith (Tabela 8); o C7+ sem Cp | entalpia de gás ideal pelos **polinómios do HYSYS**; a Tabela 8 não entra | Cp do C7+ 1,59–2,02 kJ/(kg K) a 300–400 K. O H2S não tem polinómio: a entalpia sai NaN e o fluido da Tabela 6 não corre tal como está | todas as temperaturas; Apêndice B (corre sem o H2S) | [L02.4](02-termodinamica-e-flash.md) |
| 9 | flash por substituições sucessivas | arranque de Wilson em cada chamada, Rachford–Rice sem limites e **sem teste de estabilidade** | perto do orvalho cai na solução trivial (K → 1) e declara vapor até 2,5 K abaixo do orvalho; depois salta para ~1,4 % de líquido | §5.6, salto da Figura 40 (x_cond ~1 km tarde) | [L02.2](02-termodinamica-e-flash.md) |

## Desvios menores ou só de entrada

| o texto diz ou omite | o código faz | ref. |
|---|---|---|
| propriedades de transporte pelas correlações (§2.5.2) | a equação de estado devolve viscosidade constante: 1,26 mPa·s se toda a ρ do vetor passar de 150 kg/m³, senão 0,012 mPa·s. Entra no Reynolds da mistura do no-slip e no filme anular. σ = 0,03 N/m constante (não usada na hidrodinâmica) | [L03.1](03-propriedades-de-transporte.md) |
| fecho estratificado de Taitel & Dukler | holdup e dissipação calculados com fechamentos diferentes; tensões com velocidades superficiais | [L04.3](04-hidrodinamica-mecanicista.md) |
| fecho intermitente | golfada multiplicada pelo perímetro das bolhas (~13× o da parede), perímetro do vapor no filme sem ½, Colebrook com ln no lugar de log10, f_i = 0,0142 | [L04.4](04-hidrodinamica-mecanicista.md) |
| validação de propriedades com o fluido da Tabela 6 | fluido sem N2, CO2 e H2S | [L03.4](03-propriedades-de-transporte.md) |
| Figura 13: os 159 estados | desenhados a **T − 10 K** | [L03.4](03-propriedades-de-transporte.md) |
| Figura 18: fração vaporizada do HYSYS | lida de uma coluna `vapour_volume_fraction` (a distribuição bate com a molar) | [L03.4](03-propriedades-de-transporte.md) |
| riser de 300 m (Tabela 21) | 250 m em todos os scripts | [L01.3, L06.1](06-dados-casos-e-ficheiros.md) |
| rugosidade do Apêndice A omitida | 2e-4 m; fluido 6/94 % em massa (≈ 69/31 molar); 3 nós em 100 m | [L06.1](06-dados-casos-e-ficheiros.md) |
| a §5.5 com a geometria da Tabela 21 | as figuras só se reproduzem com o perímetro interno (espessura nula). O script do repositório é outro (fluido N2/CO2/C1/nC8) | [L01.4](01-termico.md) |
| ρ = 950 kg/m³ do C7+ (Tabela 20) | não é usada; ρ_L vem só da PR | [L02.5](02-termodinamica-e-flash.md) |
| s1 = 1,696986e5 (Tabela 2) | 1,6969859271 (Hanley et al.): gralha do texto. Pedersen com T − 90,4 K e ρ_c = 162,8 kg/m³ | [L03.3](03-propriedades-de-transporte.md) |
| 2√2 e 1 ± √2 em ln φ | 2,82, 2,414 e 0,414 (sem efeito mensurável) | [L02.5](02-termodinamica-e-flash.md) |
| — | fecho da fase testa o valor da iteração anterior (memória não inicializada na primeira): defeito latente, sem efeito nos casos medidos | [L02.3](02-termodinamica-e-flash.md) |
| passos e tolerâncias não indicados (B4) | diferenças atrasadas de 0,1 bar e 0,1 K; LSODA com rtol 1e-4; RK45 nas amostras do Apêndice B; Powell (`hybr`) nas diferenças finitas; a cúbica falha sem raiz real | [L05](05-numerica-e-solvers.md) |
| — | k_ij = 0, como o cmfsim assumiu | [L02.5](02-termodinamica-e-flash.md) |

## Impacto por secção da dissertação

| secção | desvios | o que muda na leitura dos resultados |
|---|---|---|
| §5.1–5.2, Figuras 13–21 | 1, 8, fluido sem inertes, Figura 13 a T − 10 K | o eixo «Implementado» vem de um ln φ fora de PR e de outro fluido; as distribuições ainda assim ficam perto do HYSYS |
| §5.4, Apêndice A | 2, 3, 6, rugosidade | o vertical (gravítico) está certo; a concordância horizontal com o HYSYS vem do atrito 4× compensado pela μ_L de gás; corrigidos o atrito e a viscosidade, o próprio código fica a 0,47× da coluna «Dissertação» (0,43× com a rugosidade do cmfsim) |
| §5.5, sistema offshore | 1, 2, 3, 6, 7, perímetro interno, riser de 250 m | holdup no-slip baixo por causa do ln φ; pressão no-slip dependente de um padrão no limite do critério; temperaturas certas com o perímetro interno |
| §5.6, gasoduto | 5, 9 | arrefecimento 3,4× pelo erro de entrada; início da condensação ~1 km tarde pela banda morta do flash |
| §5.7, Apêndice B | 4, 8, riser de 250 m, ln φ | amostras do modelo físico com o perímetro externo; o código reproduz as da linha a 0,1 bar e 0,4 K (sem o H2S; poço e riser não corridos) |

## O que fica em aberto

- **Poço da §5.5:** sobram 0,46 bar na pressão à cabeça do poço, num troço anular nas duas versões.
- **§5.5 com o perímetro externo:** a integração não terminou (interrompida aos 22 min).
- **Ficheiros do autor em falta:** os valores do HYSYS das Figuras 13–21 e a configuração do estudo
  de casos do HYSYS. Confirmariam os pontos L03.4 e L06.3.
