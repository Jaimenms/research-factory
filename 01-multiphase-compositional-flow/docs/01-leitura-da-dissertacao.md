# Leitura estruturada da dissertação (foco no capítulo 3)

*Parte do [estudo](README.md) de 2026-10-15. Números de equações, tabelas e figuras são os
da dissertação; páginas são as impressas no PDF. O índice completo das equações está em
[06-indice-de-equacoes.md](06-indice-de-equacoes.md).*

## Ficha

| | |
|---|---|
| título | Modelagem e simulação composicional do escoamento multifásico em tubulações com geração de modelos substitutos |
| autor | Gustavo Alves de Carvalho |
| grau | Mestre, Programa de Pós-Graduação em Engenharia Química, Instituto de Química, Universidade do Estado do Rio de Janeiro; área Processos Químicos, Petróleo e Meio-ambiente |
| orientadores | Prof. André Luiz Hemerly Costa; Prof. Giovani Cavalcanti Nunes; Dr. Jaime Neiva Miranda de Souza |
| ano | 2022 (ficha catalográfica de 24/04/2023; ficheiro «REVISÃO 4, 24 abr 2023»); 201 f.; bolsa PRH/ANP |
| software | Python; SciPy (`solve_ivp` LSODA, raízes por Powell híbrido), NumPy (`gradient`), PyTorch e SMT (só para o substituto); HYSYS como referência |
| código / dados | **não há repositório, código nem ficheiros de resultados citados**; os únicos resultados numéricos impressos são as Tabelas 36–37 do Apêndice A (200 pontos de perda de carga) |
| ligação ao doutorado | co-orientada pelo autor da tese de 2010 reproduzida no mfsim; cita Souza (2010) para o conjunto de fechamentos (§2.4.1) e para a substituição dos critérios booleanos de padrão por sigmoides (§2.3.3) |

Estrutura: 1 Introdução (p. 27–39); 2 Modelagem do escoamento (40–85); **3 Abordagem de
solução numérica (86–101)**; 4 Modelo substituto (102–106); 5 Resultados (107–158);
Conclusões (159–160); Referências; Apêndice A (168–172), B (173–192), C (193–201).

## Em uma frase

Um simulador de regime permanente, 1-D, de escoamento gás–líquido em tubulações em série,
em que as propriedades e o equilíbrio de fases vêm de uma equação de estado cúbica com
composição (Peng–Robinson + flash), e o escoamento vem ou de um modelo homogéneo em
equilíbrio (não escorregamento) ou de um modelo mecanicista com holdup por padrão de
escoamento (com escorregamento), validado contra correlações do HYSYS e aplicado a um
sistema poço–linha–riser e a um gasoduto com formação de condensado.

## Os dois modelos e os dois métodos

A pergunta «o modelo do capítulo 3 considera o escorregamento?» tem resposta dupla, porque o
capítulo 3 apresenta dois métodos numéricos, cada um amarrado a um dos dois modelos do
capítulo 2:

| | modelo de **não escorregamento** (§2.2.4) | modelo **com escorregamento** (§2.2.1–2.2.3, §2.4.1) |
|---|---|---|
| hipóteses | v_L = v_G (eq. 31); fases em equilíbrio termodinâmico; holdup = fração volumétrica do flash | velocidades de fase independentes; holdup do equilíbrio mecânico local por padrão; transferência de massa à taxa do equilíbrio (eq. 25) |
| incógnitas | P, T (Tabela 3, p. 91) | m_G, m_L, P, T (Tabela 5, p. 100) |
| equações | momento da mistura (eq. 29 → 158/159), energia da mistura (eq. 30 → 162/163); a massa total é constante (eq. 153 fica implícita) | massa do gás e do líquido com M_GL (eq. 175–176), momento da mistura com termo dissipativo F(padrão) (eq. 179), energia da mistura (eq. 182) |
| equações de momento resolvidas | 1 (mistura) | **1 (mistura)**; o escorregamento entra pelo balanço algébrico por padrão (eq. 41, 59, 79, 82), que dá o holdup e o termo F |
| método numérico | **3.1 método integrador** (PVI em x; LSODA); «aqui será adotada para resolução do modelo de não escorregamento» (p. 86) | **3.2 diferenças finitas** + raízes (Powell); «utilizando-se o modelo de diferenças finitas junto com o modelo de escorregamento» (p. 96) |
| também resolvido por | 3.2 (Tabela 34: 12, 30 e 78 pontos) | só 3.2 (Tabela 35) |
| usado em | 5.5 (um dos dois perfis), **5.6 (condensado)**, 5.7 (substituto), 5.8 | **5.4 (200 pontos contra o HYSYS)**, 5.5 (o outro perfil), 5.8 |

Conclusão: **o capítulo 3 considera o escorregamento no método 3.2 e não o considera no
método 3.1**. Em nenhum dos dois há duas equações diferenciais de momento: o «modelo de
dois fluidos» da dissertação é, no sentido de Ishii, um modelo de mistura com fechamento
mecanicista de holdup (a própria dissertação o diz em §2.4.1, p. 59: «o holdup é
implicitamente resolvido para, em seguida, contabilizar o termo de perda de carga»).

## Holdup de líquido e fração vaporizada do flash

São duas grandezas diferentes ligadas por uma cadeia de relações; a Figura 35 (p. 134)
compara-as ao longo do sistema poço–linha–riser. Por secção x, com vazão mássica total m e
composição global z constantes:

1. **Flash** a (P, T, z) → fração vaporizada **molar** β_G e composições x_i, y_i; da equação
   de estado, MW_G, MW_L, ρ_G, ρ_L.
2. **Fração vaporizada mássica** (título): w_G = β_G MW_G / [β_G MW_G + (1 − β_G) MW_L].
3. **Vazões por fase**: m_G = w_G m, m_L = (1 − w_G) m. Nos dois modelos da dissertação isto
   vale em todo o x: no de não escorregamento por hipótese de equilíbrio; no com
   escorregamento porque a eq. (25) integra para m_G(x) − m_G(0) = m [R_G^EQ(x) − R_G^EQ(0)]
   e, com a entrada em equilíbrio, m_G = m R_G^EQ(x). Logo as equações de massa do método
   3.2 (175–178) são a relação algébrica m_G = m w_G(P, T) disfarçada de EDO: o que o
   modelo com escorregamento acrescenta é o **holdup** (e com ele velocidades, atrito e
   gravidade), não a repartição de massa entre as fases. Para que o integral seja a massa,
   R_G^EQ em (25) tem de ser a fração **mássica** w_G; com a volumétrica o resultado fica
   errado pela ponderação das densidades (ambiguidade B6 em [05](05-questoes-abertas.md)).
4. **Velocidades superficiais**: v_SG = m_G/(ρ_G A), v_SL = m_L/(ρ_L A).
5. **Fração volumétrica de equilíbrio** (a «fração de líquido do flash» da Figura 35; na
   literatura, *no-slip holdup* λ_L):
   λ_L = v_SL/(v_SL + v_SG) = [(1 − w_G)/ρ_L] / [(1 − w_G)/ρ_L + w_G/ρ_G].
   É uma fração do **escoamento** (do volume que atravessa a secção por unidade de tempo).
   No modelo de não escorregamento é também o holdup: v_L = v_G ⇒ H_L = λ_L, e a velocidade
   comum é v = m/(ρ_ns A) com 1/ρ_ns = w_G/ρ_G + (1 − w_G)/ρ_L.
6. **Holdup** H_L (fração do **volume do tubo** ocupada pelo líquido): só o fechamento
   mecanicista o dá. Com v_L = v_SL/H_L, v_G = v_SG/(1 − H_L) e razão de escorregamento
   S = v_G/v_L,

   H_L = S λ_L / (1 − λ_L + S λ_L).

   S = 1 devolve H_L = λ_L; S > 1 (gás mais rápido, o caso normal em horizontal e
   ascendente) dá H_L > λ_L: o líquido acumula-se no tubo. No padrão de bolhas S vem do
   drift-flux (77–79), H_L = 1 − λ_G /(C_0 + v_0 sin θ H_L^0,5 / v_M); nos outros padrões vem
   do equilíbrio de momento por padrão (41, 59, 82), que é uma equação implícita em H_L.
7. **Consequência**: a massa e a composição **em trânsito** por secção são as do flash de z;
   a massa **contida** no tubo não: por unidade de comprimento há ρ_L H_L A de líquido e
   ρ_G (1 − H_L) A de gás, e a fração mássica de gás *in situ*,
   ρ_G (1 − H_L) / [ρ_G (1 − H_L) + ρ_L H_L], é menor do que w_G sempre que S > 1. Em regime
   permanente isto é consistente (o que entra por unidade de tempo sai). Num modelo
   dinâmico deixaria de ser, porque o conteúdo da célula já não tem a composição z: é a
   razão pela qual o mfsim decidiu não usar flash de célula fechada na dinâmica
   (`doutorado/docs/future/compositional_mass_transfer.md`) e é o ponto em que a
   reprodução permanente e a extensão dinâmica se separam.

Em resumo: o flash diz **quanto** de cada fase passa (título e, com as densidades, λ_L); o
holdup diz **quanto espaço** cada fase ocupa; a diferença entre os dois é o escorregamento, e
só o modelo mecanicista o calcula.

## Como o flash entra no método 3.2, se o holdup é mecanicista

Dentro de cada avaliação do resíduo do Powell (p. 100: «internamente ao solver das equações de
conservação, encontram-se os cálculos de holdup e os cálculos de flash»), o flash a (P, T, z)
de cada nó entra em quatro sítios, e em nenhum deles dá o holdup:

1. **Repartição de massa entre as fases**: M_GL da eq. (25) é a fonte das equações de massa
   por fase (175–178). Integrado, m_G = m·w_G(P, T); as incógnitas m_G e m_L são obrigadas
   pelo resíduo de massa a seguir a repartição do flash (a menos do erro de discretização
   do operador de `gradient`).
2. **Propriedades de fase**: x_i, y_i do flash → ρ_G, ρ_L, μ_G, μ_L, σ, h_G, h_L pela EoS e
   pelas correlações (126–149). Entram no resíduo de momento (ρ_k R_k v_k² + P e o termo F)
   e no de energia (ρ_k R_k v_k h_k); o calor latente entra implicitamente, porque
   d(m_G h_G + m_L h_L)/dx contém (h_G − h_L) dm_G/dx.
3. **Entradas do fechamento mecanicista**: o equilíbrio de holdup (41, 59, 79, 82) é
   resolvido com v_SG = m_G/(ρ_G A), v_SL = m_L/(ρ_L A), ρ_k, μ_k e σ, todos derivados do
   flash. O fechamento recebe λ_L (fração do escoamento) e devolve H_L (fração do tubo).
4. **Identificação de padrão**: os critérios (32–38) usam as mesmas velocidades superficiais
   e propriedades.

Consequência não explorada na dissertação: como m_G = m·w_G(P, T) é algébrico e
H_L = H_L(P, T) pelo fechamento, o modelo com escorregamento também cabe em incógnitas só
P e T e poderia ser **marchado pelo método 3.1**. O obstáculo que a dissertação aponta
(p. 158: as derivadas só foram «explicitadas» no não escorregamento) é a derivada de H_L em
P e T através do equilíbrio de momento e da chave de padrão, que a diferenciação automática
com `custom_root` e sigmoides fornece. Fica como hipótese a testar na fase 5
([04](04-plano-de-reproducao.md#melhorias-como-opções-com-bandeira)).

## Capítulo 2: o que o capítulo 3 consome

### Equações de conservação (§2.1–2.2)

Derivação pela equação geral de Ishii & Hibiki (eq. 1), reduzida a regime permanente 1-D
com propriedades médias na seção. Para duas fases:

- massa por fase: d(ρ_L R_L v_L)/dx = M_LG (21), d(ρ_G R_G v_G)/dx = M_GL (22); soma (24):
  vazão mássica total constante;
- **transferência de massa (25)**, de Adewumi & Mucharam (1990):
  M_GL = m (∂R_G^EQ/∂P · dP/dx + ∂R_G^EQ/∂T · dT/dx), com R_G^EQ a fração vaporizada de
  equilíbrio do flash a (P, T). A termodinâmica entra aqui e só aqui no modelo com
  escorregamento;
- momento por fase (26–27) com R_k dP/dx, atrito com a parede Γ_k/A, atrito interfacial
  Γ_kj/A e gravidade; **pressão única** na seção (sem pressão interfacial, §2.2.2, p. 48);
  momento da mistura (29) sem os termos interfaciais;
- energia da mistura (30): d(ρ_L R_L v_L H_L + ρ_G R_G v_G H_G)/dx = Q_w ∓ g(…) sin θ, sem
  energia cinética, sem geração, sem termos viscosos (como TACITE e OLGA, p. 49).

Sinais a confirmar na reprodução (ver [03](03-avaliacao-e-interpretacao.md#inconsistências-e-ambiguidades-detetadas-no-texto)):
o termo gravítico de (29) está impresso com (ρ_L R_L − ρ_G R_G); o de (30) com sinal
negativo e o de (160/182) com sinal positivo e sem as frações R_k.

### Padrões de escoamento (§2.3)

Horizontal/inclinado (Taitel & Dukler 1976): parte-se de estratificado e testam-se
transições: Kelvin–Helmholtz (32) → não estratificado; h_L/D < 0,5 (33) → anular; critério
de bolhas (34) → bolhas; senão intermitente. Vertical ascendente (Taitel et al. 1980):
anular (35); bolhas (36) com a condição de diâmetro (37) ou bolhas dispersas (38); senão
intermitente. Chave horizontal/vertical a **45°** (p. 51). As comparações booleanas são
substituídas por **sigmoides** «como sugerido por Souza (2010)» (p. 56). Fluxograma na
Figura 4 (p. 57; os rótulos dos losangos estão ambíguos na figura, a Tabela 1 é a que
vale). Validação: mapas ar–água de 1 polegada (Figuras 22–23, §5.3).

### Fechamentos mecânicos por padrão (§2.4.1)

Todos têm a mesma estrutura: um balanço de momento por região em regime permanente, do qual
se elimina dP/dx para obter uma equação algébrica no holdup, e depois a perda de carga.

| padrão | equilíbrio (holdup) | perda de carga | correlações |
|---|---|---|---|
| estratificado (só horizontal) | (41): τ_wL S_L/A_L − τ_wG S_G/A_G − τ_i S_i (1/A_L + 1/A_G) + (ρ_L − ρ_G) g sin θ = 0 | (42) | geometria (43–48); τ_wk com Churchill (49–52); D_L = 4A_L/S_L, D_G = 4A_G/(S_L + S_G) (53–54, ver nota); **τ_i com f_i = 0,0142 constante** (Xiao et al. 1990) (56) |
| anular | (59) filme × core | (60) | S_i = πD√R_G (63); entrainment de Wallis (64–65); R_C (66); propriedades do core (67–68); v_F, v_C (69–70); τ_i (71) com f_i = f_CS·I_θ, I_θ = I_H cos²θ + I_V sin²θ, I_H Henstock–Hanratty (74–75), I_V Wallis (76) (unificação de Gomes 2002) |
| bolhas | drift-flux (77): v_G = C_0 v_M + v_0 sin θ H_L^0,5, C_0 = 1,15 (Chokshi), v_0 Harmathy (78); H_L implícito em (79) | (80) homogéneo com ρ_M (81) | f da mistura com Churchill |
| intermitente | (82) filme (como estratificado) | (83) ponderada por L_S/L_U e L_F/L_U | H_LS Gomez et al. 2000 (84); v_TB Bendiksen (85); v_BS (86); L_S Scott/Xiao (87); velocidades por conservação de massa (88–93); τ_wS com propriedades médias (94–96); L_U e H_L (97–99) |

Nota a (54): Taitel & Dukler usam D_G = 4A_G/(S_G + S_i); a dissertação imprime S_L + S_G.
A ser confirmado com o código (ver [05](05-questoes-abertas.md)).

### Fechamento energético (§2.4.2)

Q_w = −U π D (T − T_amb) (100), com U dado por trecho (7, 5 e 5 W/m²K no caso 5.5; 1, 3 e
6 no caso 5.6).

### Termodinâmica (§2.4.3–2.5)

- **Peng–Robinson** (101–112): forma cúbica em Z (102); a, b por componente (106–109);
  regras de mistura clássicas (110–112) com δ_ij (valores não dados; presume-se zero).
- **Flash P-T** (113–125): Rachford–Rice (118) + igualdade de fugacidades φ-φ (119–122);
  ln φ̂_k de PR (121); substituições sucessivas K_i ← K_i f̂_L/f̂_V (124); critério de paragem
  Σ(f̂_L/f̂_V − 1)² (123); estimativa inicial de Wilson (125). **Sem teste de estabilidade**
  (Michelsen): o número de fases é decidido pela fração vaporizada do próprio flash
  (p. 76).
- **Propriedades**: ρ_k da EoS (126; a expressão impressa está invertida, deve ser
  P·MW/(Z R T)); μ_G por Pedersen et al. (1984/1987) com metano de referência e
  Hanley–McCarty–Haynes (127–136, Tabela 2); μ_L por Allan & Teja (1991) com número de
  carbonos efetivo (137–141); σ por Brock & Bird (142–143) com média molar (144);
  entalpia = gás ideal (148, coeficientes de Smith et al. 2018, Tabela 8) + residual de PR
  (146–147, forma do IDAES).

## Capítulo 3: abordagem de solução numérica

Arquitetura dos dados (Figura 9, p. 86): condições de contorno (P, T, vazões nas
extremidades, T_amb) + propriedades do fluido (ρ, μ, H, σ, distribuição das fases no
equilíbrio) + tubulação (D, L, inclinação, rugosidade, U) → solver.

### 3.1 Método integrador (p. 87–93): modelo de não escorregamento

1. Toda variável M ∈ {v, ρ_k, R_k, H_k} é escrita por derivada total em (P, T) (150).
2. Massa (153–155), momento (156–158) e energia (160–162) ficam lineares em dP/dx e dT/dx:
   E_1 dP/dx + E_2 dT/dx = I_1 (159), E_3 dP/dx + E_4 dT/dx = I_2 (163). A massa (155) não
   entra no sistema final porque m é constante (Tabela 3).
3. Sistema 2×2 resolvido por fórmula fechada (166–167) ou `A x = b` (168).
4. **Derivadas ∂M/∂P, ∂M/∂T por diferenças de 1.ª ordem** (169–170; o texto diz «para
   frente/upwind» e escreve M(P) − M(P − ΔP); a Figura 10 diz P + dP). ΔP e ΔT não são
   dados. Cada avaliação do lado direito custa **três flashes** com propriedades: (P, T),
   (P + dP, T), (P, T + dT) (Figura 10, p. 93).
5. Integração em x com **LSODA** (`scipy.integrate.solve_ivp`), passo e rigidez automáticos.
6. Graus de liberdade (3.1.5): só PVI; especificar vazão global, composição, P e T **na
   entrada**. Fração volumétrica das fases do flash: R_G = (β_G/ρ_G) / (β_G/ρ_G + β_L/ρ_L) com
   β em base mássica (relação implícita, não escrita na dissertação).
7. Desempenho (Tabela 34): 7 s no sistema de 5.5, contra 16–140 s do mesmo modelo por
   diferenças finitas.

### 3.2 Diferenças finitas (p. 94–101): modelo com escorregamento

1. Malha uniforme de n pontos; derivadas por `numpy.gradient` (2.ª ordem central, avançada
   e recuada nos extremos, Tabela 4; nota: a expressão «para frente» da Tabela 4 está
   impressa com os índices da recuada, e o `gradient` do NumPy só é de 2.ª ordem nos
   extremos com `edge_order=2`).
2. Forma matricial (1/Δx) M Y = B (171–172); vetores Y e B por equação:
   massa (177–178, Y = [m_G; m_L], B = [M_GL; M_LG]), momento (180–181,
   Y = ρ_L R_L v_L² + ρ_G R_G v_G² + P, B = −F(padrão)), energia (183–184).
3. Resíduos (185–190). Incógnitas: m_G, m_L, P, T em todos os nós menos os especificados
   (P e T «tamanho n − 1»); vazões por fase sempre especificadas na entrada (p. 96).
4. Dois conjuntos de condições de contorno (3.2.5): (a) P, m_k e T na entrada; (b) T na
   entrada, m_k e P **na saída**. Esta flexibilidade é a vantagem sobre 3.1.
5. **Holdup e flash aninhados no resíduo** (p. 100): «a fração volumétrica não aparece
   sendo resolvida explicitamente»; em cada avaliação resolve-se o equilíbrio por padrão
   (41/59/79/82) e o flash.
6. Solução por raízes com o **Powell híbrido modificado** do SciPy (`root`, MINPACK
   `hybrd`), sem Jacobiana analítica; precisa de estimativa inicial em todos os nós
   (Figura 11: v_L, v_G, P, T; a Tabela 5 diz m_G, m_L).
7. Limitações declaradas: convergência problemática «talvez pela presença de
   descontinuidades, como … aparecimento e desaparecimento de uma fase» (p. 99); custo
   que cresce «de forma exponencial» com n (Tabela 35: 37, 47, 230 s para 12, 30, 78 pontos
   no modelo com escorregamento).
8. A dissertação não mostra estudo de independência de malha nem diz com que n foram
   gerados os resultados de 5.4 e 5.5.

## Capítulo 5: o que valida o capítulo 3

| secção | o que compara | referência | dados no PDF | reprodutível como |
|---|---|---|---|---|
| 5.1 termodinâmica | ρ_V, ρ_L, H_V, H_L, fração vaporizada de um fluido de 11 componentes (Tabela 6) em 159 pontos (P, T) dentro do envelope | HYSYS com PR | Tabelas 6–9 completas (composição, Tc, Pc, ω, MW, H_form, A–D do Cp, T_b); só R² (Tabelas 10–11: 0,997/0,992/0,982/0,997/0,978) e figuras de paridade | contra uma referência independente de PR (p. ex. CoolProp ou `thermo`) e contra o HYSYS se o dono tiver os números |
| 5.2 μ e σ | viscosidades e tensão superficial | HYSYS (correlações diferentes) | R² 0,941/0,964/0,863 (Tabela 12) | idem |
| 5.3 padrões | mapas ar–água horizontal e vertical, D = 25 mm, ε = 1,53 mm, v_LS 0,1–50, v_GS 0,1–500 m/s | curvas de Taitel–Dukler 1976 e Taitel 1980 | Tabelas 13–14; Figuras 22–23 | exato: os critérios são fórmulas fechadas |
| **5.4 perda de carga** | 100 linhas horizontais + 100 verticais, L = 100 m, D 4–10 in (4), P_in 20–160 bar (5), m 2–10 kg/s (5), T_in 40 °C, U = 0, fluido CH₄ 70 % + C7+ 30 % | Tulsa (Zhang 2003), HTFS, Gregory, Aziz, Beggs & Brill no HYSYS | **Apêndice A: Tabelas 36–37 com os 200 valores da dissertação e dos modelos (3 algarismos)**; Tabela 18 (P → R_G na entrada) | nível B (tolerância ~1 %) nos valores «Dissertação» |
| 5.5 sistema offshore | poço 1500 m vertical (D 0,127 m), linha 6000 m horizontal (0,1524 m), riser 300 m (0,1524 m); 170 bar, 80 °C, 10 kg/s; fluido de 9 componentes com C7+ (Tabelas 19–22) | Tulsa e Beggs & Brill no HYSYS; os dois modelos da dissertação | só figuras 32–37 (P, T, holdup, fração de líquido flash × mecanicista, CO₂ por fase, padrões) | digitalização das figuras; comparação entre os dois modelos próprios |
| 5.6 condensado | gasoduto 8 km, 12 in, 70 bar, 15 kg/s, gás de 5 componentes (Tabela 23); 6 casos T_in 60/80 °C × U 1/3/6 W/m²K (Tabela 25); **modelo de não escorregamento** | nenhuma externa | Figuras 38–40; Tabela 26 (composição na saída) | digitalização; Tabela 26 numérica |
| 5.8 desempenho | tempos de 3.1 e 3.2 (Tabelas 34–35); flash = 90 % do custo | Intel i5 2,5 GHz | numérico | comparação de ordem de grandeza |

Resultados a reter: no horizontal a dissertação fica dentro da faixa dos modelos da
literatura em 63 % dos casos e, quando sai, fica **abaixo** (holdup menor a vazões
< 6 kg/s, p. 126); no vertical só 4 % ficam abaixo da faixa; os pontos 4–5 verticais caem na fronteira
anular/intermitente (Figura 30). No caso 5.5 os perfis de P e T coincidem com o Tulsa, o
holdup diverge entre modelos, e há **saltos de holdup nas mudanças de inclinação** por
troca do mapa de padrões (p. 133), o problema que a issue 008 do mfsim resolveu com
mistura por ângulo.

## Adiado: modelo substituto (feito em 2026-10-15, issue 009)

Capítulo 4 (amostragem por hipercubo latino com SMT; redes multicamada em PyTorch; 70/30;
Adam; busca em grelha de 135 configurações), secção 5.7 (300 amostras por trecho do caso
5.5 com o modelo de não escorregamento, Tabelas 27–33, Figuras 41–58; R² ≥ 0,9 em P e T;
0,07 s por avaliação) e apêndices B e C (amostras e hiperparâmetros). O plano em
[04](04-plano-de-reproducao.md#o-que-fica-adiado-e-como-o-plano-o-prepara) deixa a porta
aberta (um `vmap` sobre o solver gera a base de amostras em segundos).
