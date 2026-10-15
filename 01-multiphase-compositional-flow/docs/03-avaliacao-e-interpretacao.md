# Avaliação e interpretação

*Parte do [estudo](README.md) de 2026-10-15. Opinião técnica, escrita para decidir se e como
reproduzir; os pontos marcados «a confirmar» dependem do código ou do autor
([05](05-questoes-abertas.md)).*

## O que a dissertação é de facto

Apesar do título e da secção 2.2 («modelo de dois fluidos»), o que está implementado é um
**modelo mecanicista unificado em regime permanente** na linhagem Xiao, Shoham & Brill (1990)
e Gomez et al. (2000): uma equação de momento da mistura, holdup por equilíbrio mecânico
local de cada padrão, identificação de padrão de Taitel–Dukler, energia da mistura, e, como
contribuição própria, a **termodinâmica composicional acoplada** (PR + flash em cada ponto, com
a transferência de massa à taxa do equilíbrio, de Adewumi & Mucharam 1990) e a comparação
sistemática com o HYSYS. O modelo de não escorregamento é o modelo homogéneo em equilíbrio
(HEM) com energia. Isto não é uma crítica: é exatamente a classe de modelo que os simuladores
de regime permanente da indústria usam, e é a escolha certa para o que a dissertação quer
(perfis de P e T, formação de condensado, base para substitutos). Mas muda o mapeamento para
o mfsim: o parente mais próximo não é o `two_fluid` de 4 equações, é o `drift_flux_bb` (3
equações, holdup por correlação) com um fechamento mecanicista no lugar do Beggs & Brill.

A ligação genealógica é direta: a dissertação cita Souza (2010) para o conjunto de
fechamentos e para as sigmoides da identificação de padrão, e usa Churchill, Taitel–Dukler e
a mesma geometria do estratificado que o mfsim. Reproduzi-la com a estrutura do `doutorado`
é natural; a maior parte do esforço vai para o que o mfsim não tem (termodinâmica, energia,
solver em x).

## Pontos fortes

- **Documentação completa dos dados de entrada**: composições, propriedades críticas,
  coeficientes de Cp, T_b, geometrias, U, T_amb, condições de contorno, faixas da análise
  paramétrica. Tudo o que é preciso para montar os casos está no texto.
- **Apêndice A**: 200 valores numéricos próprios e dos cinco modelos do HYSYS, o que dá um
  alvo quantitativo de reprodução que a maioria das dissertações não dá.
- **Dois métodos para o mesmo problema** com tempos medidos (Tabelas 34–35), que permitem
  medir o ganho das melhorias.
- **Casos de aplicação realistas** (poço–linha–riser; gasoduto com condensação) com
  análise composicional por fase (CO₂, Tabela 26).
- Honestidade sobre limitações: PVI só com entrada especificada; convergência do método
  3.2; dispersão de 375 % entre os próprios modelos da literatura no horizontal (p. 121).

## Lacunas de método

1. **Derivadas numéricas de 1.ª ordem** (169–170) com ΔP, ΔT não documentados, dentro de um
   integrador com controlo de erro: o erro de truncatura das derivadas não é controlado e
   custa três flashes por avaliação (Figura 10).
2. **Flash sem teste de estabilidade** e com substituições sucessivas apenas: a deteção de
   fase única é feita «pelo próprio algoritmo de flash» (p. 76). Perto do ponto de orvalho
   (que é precisamente o que 5.6 quer detetar) e perto do crítico isto é frágil; as
   substituições sucessivas são lentas perto da fronteira de fases, o que explica os 90 % do
   custo (p. 158).
3. **Método 3.2 sem Jacobiana**: o Powell híbrido avalia a Jacobiana por diferenças (O(n)
   avaliações do resíduo, cada uma com n flashes e n equilíbrios de holdup) e o custo cresce
   super-linearmente (Tabela 35). O «exponencial» da p. 157 é isto, não uma propriedade do
   problema.
4. **Sem estudo de malha**: só tempos para 12, 30 e 78 pontos; não se sabe com que n foram
   gerados o Apêndice A e as figuras de 5.5.
5. **Holdup aninhado** sem continuidade garantida entre padrões: as sigmoides suavizam a
   escolha, mas a dissertação relata saltos de holdup nas mudanças de inclinação (p. 133)
   e o ponto 5 vertical na fronteira anular/intermitente (Figura 30). O mfsim já passou por
   isto (issues 008, 020, 021).
6. **Estimativa inicial** do método 3.2 não descrita; a solução do método 3.1 seria a
   estimativa óbvia e não é usada.
7. **Validação só contra o HYSYS**, sem dados experimentais; o próprio texto mostra que os
   modelos da literatura dispersam entre si até 1200 %.
8. Sem código, sem resultados gravados além do Apêndice A: a reprodução bit a bit, que foi
   o alicerce do doutorado, não é possível.

## Inconsistências e ambiguidades detetadas no texto

Todas «a confirmar» (gralhas de edição ou o que o código faz). Devem virar issues de tipo
*bugfix* ou *estudo* no projeto novo, com a decisão registada.

| onde | o que está impresso | o que se esperava | consequência |
|---|---|---|---|
| eq. (29) | termo gravítico −(ρ_L R_L − ρ_G R_G) g sin θ | −(ρ_L R_L + ρ_G R_G) g sin θ (como em 156) | gralha provável; (156) está certa |
| eq. (30) vs (160)/(182) | (30): Q_w − g(ρ_L R_L v_L + ρ_G R_G v_G) sin θ; (160): Q_w + g(ρ_L v_L + ρ_G v_G) sin θ | o trabalho da gravidade com H sem energia potencial é −g sin θ Σ ρ_k R_k v_k | sinal e frações R_k diferem; afeta o perfil de T em trechos verticais |
| eq. (25) | R_G^EQ «fração vaporizada» com m = «massa total»; lista de símbolos: R = fração volumétrica | (25) integra para m_G = m R_G^EQ, logo R_G^EQ tem de ser o título **mássico** w_G; com a fração volumétrica λ_G a repartição de massa fica errada pela ponderação das densidades | ambiguidade de base (molar/mássica/volumétrica) que muda m_G, m_L e o holdup |
| eq. (46) | S_G = (2π − α) D | (2π − α) D/2 | gralha (S_L = αD/2 está certa) |
| eq. (54) | D_G = 4A_G/(S_L + S_G) | 4A_G/(S_G + S_i) (Taitel & Dukler) | muda Re_G e τ_wG no estratificado |
| eq. (84) | H_LS = exp(−7,85·10³ θ + 2,48·10⁻⁶ Re_SS) | Gomez et al. (2000): exp[−(7,85·10⁻³ θ + 2,48·10⁻⁶ Re)] | expoente e sinal |
| eq. (89) | v_LS aparece dos dois lados | v_LS = [v_M − v_BS (1 − H_LS)]/H_LS | gralha |
| eq. (126) | ρ_k = Z_k R T MW_k / P | ρ_k = P MW_k /(Z_k R T) | gralha (a validação de 5.1 mostra que o código está certo) |
| eq. (147) | da/dT com a_i sob as duas raízes | forma padrão de PR: −(R/2) ΣΣ y_i y_j (1 − k_ij) [κ_j √(a_i T_c,j/P_c,j) + κ_i √(a_j T_c,i/P_c,i)] | verificar na implementação |
| Tabela 4 | «para frente»: −3f(x) + 4f(x − Δx) − f(x − 2Δx) | −3f(x) + 4f(x + Δx) − f(x + 2Δx) | gralha; e `numpy.gradient` só é de 2.ª ordem nos extremos com `edge_order=2` |
| eq. (169–170) vs Figura 10 | texto: M(P) − M(P − ΔP) («para frente»); figura: P + dP | — | lado da diferença |
| Figura 11 vs Tabela 5 | figura: estimativa de v_L, v_G, P, T; eq. 189–191 | Tabela 5: m_G, m_L, P, T; eq. 188–190 | numeração/variáveis |
| Figura 4 | rótulos dos losangos («Anular?» → NÃO → «Anular») | a Tabela 1 e o texto | seguir a Tabela 1 |
| Tabela 26 | título «caso 1», texto «caso 2» | — | identificar pelo perfil da Figura 40 |
| eq. (141) | unidades de η_L e de A, B, C de Allan & Teja não ditas | — | precisa da fonte |

## O que a reprodução moderna muda

Cada item é uma **opção com bandeira**, com o comportamento da dissertação disponível como
default (regra do `doutorado`), e uma issue com antes, depois e benefício medido.

| onde | dissertação | reprodução | ganho esperado |
|---|---|---|---|
| ∂M/∂P, ∂M/∂T (3.1) | diferenças de 1.ª ordem, 3 flashes | diferenciação automática (JAX) com 1 flash; derivadas do flash pelo teorema da função implícita (`custom_root`) | exatidão; ~3× menos flashes; sem ΔP, ΔT |
| flash | substituições sucessivas, Wilson, sem estabilidade | o mesmo como default; opções: aceleração (Newton nas K ou DEM), teste de estabilidade de Michelsen, `while_loop` compilado | robustez no orvalho/bolha; o custo de 90 % cai com `jit` |
| método 3.2 | Powell sem Jacobiana | Newton com Jacobiana exata (`jacfwd`), esparsa por blocos, busca linear e limites (`steady.newton_steady`); pseudo-transiente como fallback | custo linear em n em vez de «exponencial»; convergência com fases a aparecer |
| estimativa inicial (3.2) | não descrita | solução do método 3.1 (não escorregamento) | fragilidade declarada resolvida |
| repartição de massa (3.2) | EDO dm_k/dx = ±M_GL com derivadas do flash por diferenças | forma algébrica m_G = m·w_G(P, T): incógnitas só P e T; abre a marcha 3.1 ao modelo com escorregamento (hipótese) | metade das incógnitas; método rápido com holdup mecanicista |
| condições de contorno | 3.1 só entrada; 3.2 entrada ou saída | *shooting*: marcha 3.1 + raiz na P de entrada, com derivada por AD através do integrador | pressão na saída também no método rápido |
| integrador em x | LSODA (SciPy) | LSODA como caminho fiel; integrador em JAX (Diffrax ou RK/BDF próprio) para `jit`/`vmap` | 200 casos do Apêndice A numa chamada |
| padrões | sigmoides com k fixo; chave a 45° | IPE suave e local (issues 020–021), mistura por ângulo (008), rampas de fração baixa/alta (016–017) | sem saltos de holdup nas mudanças de inclinação |
| malha (3.2) | n fixo, sem estudo | estudo de independência com malha graduada (009) | n documentado por caso |
| derivadas em geral | diferenças | AD em todo o lado, com as lições das issues 037 (tangentes NaN) e 038 | Jacobianas sem ajuste de passo |
| validação | R² contra o HYSYS | referência independente de PR (CoolProp/`thermo`) a 1e-10 nas propriedades; Apêndice A a ~1 %; figuras digitalizadas; goldens A entre versões próprias | reprodutibilidade |
| gralhas | — | cada uma como issue *bugfix* com as duas formas disponíveis e o efeito medido | rastreabilidade |

## Nível de reprodução alcançável

| nível | o que é | aplicável a |
|---|---|---|
| A (bit a bit) | hash dos arrays | só entre versões do projeto novo (confirmado em 2026-10-15: não há código nem acesso ao autor) |
| B (tolerância) | ~1 % (3 algarismos impressos) | Apêndice A: 100 + 100 casos «Dissertação»; Tabela 26; Tabela 18 |
| C (qualitativo/quantitativo por figura) | curvas digitalizadas, 2–5 % | 5.1–5.2 (paridade), 5.3 (mapas), 5.5 (perfis), 5.6 (perfis) |
| termodinâmica | 1e-8 a 1e-10 contra outra implementação de PR | 5.1 (159 pontos) e os fluidos de 5.4–5.6 |

Dado que o horizontal de 5.4 depende do holdup do estratificado e o vertical do padrão
identificado, a reprodução do Apêndice A é também um teste indireto das gralhas (54) e (84):
se a dissertação implementou (54) tal como impressa, só esse D_G reproduzirá a coluna
«Dissertação». Isto é um «estudo» no sentido do `doutorado`: usar os números para descobrir o
que o código fazia.

## Valor estratégico

- Para o doutorado: o módulo termodinâmico e a energia são os dois degraus que separam o
  mfsim dos temas de `docs/future/` (composicional, CO₂, criogénico). Construí-los aqui, em
  permanente e com backend injetado, é o caminho de menor risco.
- Para a dissertação: uma versão reexecutável, com derivadas exatas, Jacobiana exata,
  robustez no flash e mapa de padrões contínuo, que responde às suas próprias sugestões de
  trabalho futuro (desempenho do equilíbrio acoplado ao escoamento; extensão a transiente,
  que seria o mfsim).
- Para o modelo substituto (adiado): com `vmap` e `jit`, as 300 amostras por trecho de 5.7
  custam segundos, e a derivada do solver em relação às entradas (AD) permite treinar
  substitutos com gradientes (PINN, que a dissertação lista como futuro).
