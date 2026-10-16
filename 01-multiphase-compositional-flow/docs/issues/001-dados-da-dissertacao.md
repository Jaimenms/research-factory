# 001 · Dados em falta ou ambíguos na dissertação e as hipóteses adotadas

| | |
|---|---|
| tipo | hipótese |
| área | dados de entrada (termodinâmica) |
| estado | em vigor (2026-10-15); rever se surgirem dados do autor |
| onde | `dissertation/data/components.json` (`_assumptions`), `cmfsim/thermo/viscosity.py`, `cmfsim/thermo/pr.py` |
| dados | Tabelas 2, 7, 8, 9, 20 da dissertação; Pedersen & Fredenslund (1987); Smith et al. (2018) |

## Antes

A dissertação não dá: coeficientes de Cp de gás ideal do pseudocomponente C7+ (Tabela 8 só
tem os puros), parâmetros de interação binária k_ij, número de carbonos efetivo de Allan &
Teja para CO₂, N₂, H₂S e C7+, as regras de mistura e o parâmetro α do método de Pedersen, a
massa específica crítica do metano, o ΔT da tangente hiperbólica de (136), e imprime
«1,696986×10⁵» para s1 na Tabela 2 e «28,758» para B do propano na Tabela 8.

## Depois

| item | hipótese |
|---|---|
| Cp do C7+ | extrapolação linear da tendência C4–C8 da Tabela 8 até Nc ≈ 40 (MW 560): A = 21,4, B = 340e-3, C = −108,6e-6, D = 0 |
| H_form do C7+ | −1023 kJ/kg (Tabela 20) × 560 kg/kmol |
| k_ij | zero |
| Nc_eff | n-alcanos = n; CO₂, N₂, H₂S = 1; C7+ = **20** (revisto em 2026-10-15: com 39,9 = (MW − 2)/14 os polinómios de Allan & Teja saem do domínio e dão μ_L = 1,3e-4 Pa·s, Re_SS ≈ 1,6e6 e H_LS de Gomez ≈ 0,02; com 15–30 dão 1,5–4 mPa·s e H_LS 0,7–0,87, e os casos verticais a 20 bar do Apêndice A passam de −83 % para a faixa dos outros) |
| Pedersen | Pedersen & Fredenslund (1987): Tc, Pc, MW da mistura; α = 1 + 7,378e-3 ρ_r^1,847 MW^0,5173; ρ_c(CH₄) = 162,84 kg/m³; ΔT = T − 91 K; densidade do metano por PR na fase estável |
| ρ de referência do C7+ (950 kg/m³, Tabela 20) | **não usado**: ρ_L vem da EoS (PR dá 656–664 kg/m³ no líquido da §5.5). A Figura 34 só se reproduz com ρ_L ≈ 840 kg/m³ ou com o Pc do C7+ ×1,25 (issue 004, «Achados nas Figuras 34 e 40»): hipótese a confirmar com o autor, nada adotado |
| s1 (Tabela 2) | 1,696985927 (Hanley et al. 1975) |
| B do propano | 28,785 (Smith et al. 2018) |
| Ω_a, Ω_b de PR | 0,45724 e 0,0778 como impressos (eqs. 106–107) |

## Benefício

Casos montáveis sem o autor; cada hipótese é um campo de dados ou uma constante nomeada, trocável sem
tocar no código do simulador.
