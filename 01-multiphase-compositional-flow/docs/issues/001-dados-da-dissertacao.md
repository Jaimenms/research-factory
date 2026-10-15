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
| Nc_eff | n-alcanos = n; CO₂, N₂, H₂S = 1; C7+ = (MW − 2)/14 = 39,9 |
| Pedersen | Pedersen & Fredenslund (1987): Tc, Pc, MW da mistura; α = 1 + 7,378e-3 ρ_r^1,847 MW^0,5173; ρ_c(CH₄) = 162,84 kg/m³; ΔT = T − 91 K; densidade do metano por PR na fase estável |
| s1 (Tabela 2) | 1,696985927 (Hanley et al. 1975) |
| B do propano | 28,785 (Smith et al. 2018) |
| Ω_a, Ω_b de PR | 0,45724 e 0,0778 como impressos (eqs. 106–107) |

## Benefício

Casos montáveis sem o autor; cada hipótese é um campo de dados ou uma constante nomeada, trocável sem
tocar no código do simulador.
