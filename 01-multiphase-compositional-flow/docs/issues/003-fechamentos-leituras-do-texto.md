# 003 · Fechamentos e critérios de padrão: leituras do texto e opções

| | |
|---|---|
| tipo | hipótese / bugfix |
| área | fechamentos mecânicos (§2.4.1) e identificação de padrão (§2.3) |
| estado | em vigor (2026-10-15); as opções decidem-se pelas Figuras 22–23 digitalizadas e pelo Apêndice A |
| onde | `cmfsim/closures/` (`ClosureOptions`), `docs/03-avaliacao-e-interpretacao.md` |
| dados | dissertação eqs. 32–99; Taitel & Dukler (1976); Taitel et al. (1980); Gomez et al. (2000); Henstock & Hanratty (1976) |

## Antes

Várias equações impressas têm gralhas ou ambiguidades: (34) com ρ_G/ρ_L; (36) sem o fator 3 e com
ρ_G²; (37) com ≤ 4,36; (38) sem o limite de fração de vazio 0,52; (46) S_G sem o /2; (50) na forma
de Darcy; (54) D_G com S_L + S_G; (75) «v_L/v_G»; (84) com 7,85·10³; (85) com cos θ dentro da
raiz; (86) com H_LS^0,1; (89) com v_LS dos dois lados; f_i do filme do intermitente não dito; a
viscosidade da mistura homogénea de bolhas não dita; Tabela 13 com σ = 0,0072 N/m.

## Depois

| equação | implementado por omissão | alternativa (bandeira) |
|---|---|---|
| (34) | (ρ_L − ρ_G)/ρ_L (Taitel & Dukler) | `dispersed_density="as_printed"` (ρ_G/ρ_L) |
| (36) | v_SL > 3 v_SG − 1,15[gΔρσ/ρ_L²]^¼ (Taitel 1980) | `bubble_slug="as_printed"` (sem o 3) |
| (37) | ≤ 4,36 como impresso (bolhas existem em tubos pequenos) | `bubble_existence="taitel"` (≥ 4,36) |
| (38) | sem limite de vazio, como impresso | `dispersed_max_void=0.52` (Taitel 1980) |
| (46) | S_G = (2π − α)D/2 | — (gralha certa) |
| (49–50) | τ = f_Fanning ρ v\|v\|/2 com Churchill em Fanning (o mfsim) | — |
| (54) | D_G = 4A_G/(S_G + S_i) (Taitel & Dukler) | `stratified_DG="as_printed"` (S_L + S_G) |
| (56) | f_i = 0,0142 | `stratified_interfacial="taitel_dukler"` ou `"andritsos_hanratty"` |
| (75) | ν_L/ν_G (Henstock & Hanratty) | — |
| (84) | exp(−(7,85·10⁻³ θ + 2,48·10⁻⁶ Re_SS)) (Gomez 2000) | — (a forma impressa dá H_LS > 1) |
| (85) | 0,542√(gD) cos θ + 0,351√(gD) sin θ (Bendiksen) | — |
| (86) | H_LS^0,1 como impresso | — |
| (89) | v_LS = [v_M − v_BS(1 − H_LS)]/H_LS | — |
| filme do intermitente | f_i = 0,0142 (p. 71) | `intermittent_fi_const` |
| bolhas (80) | μ_M = μ_L H_L + μ_G (1 − H_L) | — |
| sigmoides | relativas, k = 50 | `k_switch=None` (degraus) |
| 45° | degrau | `blend_deg > 0` (rampa C¹, issue 008 do mfsim) |

A correlação de H_LS de Gomez (84) só vale até Re_SS ≈ 2e5: acima disso H_LS → 0 e o pistão deixa de
existir (o piso de 0,05 em `slug_holdup` evita divisões por zero, não é física); o que a torna
sensível à viscosidade do líquido (issue 001, Nc_eff do C7+).

L_S (87) não afeta nem o holdup nem a perda de carga (as razões L_S/L_U e L_F/L_U não dependem de
L_S), logo a ambiguidade de unidades de (87) é inócua.

## Decisões pelas Figuras 22–23 digitalizadas (2026-10-15)

Classificação ponto a ponto dos mapas ar–água de 2,5 cm (395 pontos no horizontal, 398 no
vertical; `references/figures/data/fig_22.json`, `fig_23.json`) com o `ClosureOptions` em degraus duros:

| opção | acerto horizontal | acerto vertical | decisão |
|---|---|---|---|
| base (KH com v_G real, (34) de Taitel, D_G Taitel–Dukler, f_i 0,0142, σ 0,072) | 91,4 % | 83,2 % | — |
| + `dispersed_max_void = 0.52` (limite de Taitel 1980, omitido no texto) | 91,4 % | **95,5 %** | **default** |
| `bubble_existence = "taitel"` (≥ 4,36) | 91,4 % | 86,4 % | fica «como impresso» |
| σ = 0,0072 (Tabela 13 como impressa) | 91,4 % | 93,0 % | a dissertação usou 0,072 |
| (34) com ρ_G/ρ_L como impresso | 79 % (intermitente 0 %) | — | fica Taitel |
| KH com v_SG («se o gás escoasse sozinho», p. 53) | 88 % (intermitente 42 %) | — | fica v_G real |
| D_G como impresso (S_L + S_G) | 92,7 % | — | diferença dentro do ruído; fica Taitel–Dukler |
| f_i = f_G (Taitel & Dukler) | 93,2 % | — | idem; fica 0,0142 (explícito no texto) |

O erro restante no horizontal está na fronteira intermitente/anular (h/D < 0,5 avaliado no
equilíbrio estratificado): 73–76 % dos pontos intermitentes da dissertação.

## Benefício

Mapas ar–água de 2,5 cm com a estrutura de Taitel–Dukler e Taitel (`tests/test_closures.py`): estratificado a
baixa v_SL, intermitente, anular acima de v_SG ≈ 12,5 m/s no vertical (3,1[σgΔρ]^¼/ρ_G^½), bolhas dispersas
a alta v_SL; resíduos das equações de equilíbrio < 1e-6 nas raízes; paridade numpy × JAX 1e-9; derivadas de
H_L e F por AD iguais às diferenças centrais a 1e-4. A comparação quantitativa com as Figuras 22–23 fica
para quando as digitalizações estiverem prontas.
