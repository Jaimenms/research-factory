# Índice das equações da dissertação

*Parte do [estudo](README.md) de 2026-10-15. Numeração da dissertação; «módulo» é o
planeado em [04](04-plano-de-reproducao.md) (pacote `cmfsim`, nome a decidir). As marcadas
com ⚠ têm uma gralha ou ambiguidade registada em [03](03-avaliacao-e-interpretacao.md).*

## Capítulo 2: modelagem

| eq. | conteúdo | secção | módulo |
|---|---|---|---|
| 1–4, 5–7 | equação geral de conservação (Ishii & Hibiki); continuidade; permanente 1-D | 2.1.1 | — (dedução) |
| 8–13 | momento; forma 1-D com Γ/A e g sin θ | 2.1.2 | — |
| 14–20 | energia em entalpia; forma 1-D | 2.1.3 | — |
| 21–22 | massa por fase com M_LG, M_GL | 2.2.1 | `models/mechanistic.py` |
| 23–24 | M_LG = M_GL; continuidade da mistura (m constante) | 2.2.1 | idem; `models/noslip.py` |
| 25 ⚠ | transferência de massa: M_GL = m (∂R_G^EQ/∂P P' + ∂R_G^EQ/∂T T') | 2.2.1 | `models/mechanistic.py` (derivadas de `thermo/flash.py`) |
| 26–27 | momento por fase (pressão única, atrito parede e interface, gravidade) | 2.2.2 | — (origem dos fechamentos) |
| 28 | Γ_LG = Γ_GL | 2.2.2 | — |
| 29 ⚠ | momento da mistura | 2.2.2 | `models/mechanistic.py` |
| 30 ⚠ | energia da mistura com Q_w | 2.2.3 | `models/*.py` (`energy`) |
| 31 | v_L = v_G (não escorregamento) | 2.2.4 | `models/noslip.py` |
| 32 | Kelvin–Helmholtz: estratificado → não estratificado | 2.3.1 | `closures/regime.py` |
| 33 | h_L/D < 0,5: intermitente → anular | 2.3.1 | idem |
| 34 | intermitente → bolhas (horizontal) | 2.3.1 | idem |
| 35 | anular (vertical) | 2.3.2 | idem |
| 36–37 | bolhas (vertical) e condição de diâmetro | 2.3.2 | idem |
| 38 | bolhas dispersas (vertical) | 2.3.2 | idem |
| Tabela 1, Figura 4 | resumo das transições; fluxograma; sigmoides (Souza 2010); 45° | 2.3.3 | idem |
| 39–40 | momento por fase no estratificado (sem aceleração) | 2.4.1.1 | `closures/stratified.py` |
| 41 | equilíbrio do estratificado (holdup) | 2.4.1.1 | `closures/holdup.py` |
| 42 | perda de carga do estratificado | 2.4.1.1 | `closures/stratified.py` |
| 43–44 | h_L/D e ângulo molhado α | 2.4.1.1 | `closures/geometry.py` (cópia do mfsim) |
| 45–47 ⚠(46) | S_L, S_G, S_i | 2.4.1.1 | idem |
| 48 | dA_L/dh_L = S_i | 2.4.1.1 | idem |
| 49 | τ_wk = f_k ρ_k v_k \|v_k\| / 2 | 2.4.1.1 | `closures/friction.py` |
| 50–52 | Churchill | 2.4.1.1 | `closures/friction.py` (cópia do mfsim) |
| 53–54 ⚠(54) | D_L, D_G | 2.4.1.1 | `closures/stratified.py` (opção) |
| 55 | Re_k | 2.4.1.1 | `closures/friction.py` |
| 56 | τ_i com f_i = 0,0142 (Xiao) | 2.4.1.1 | `closures/stratified.py` (opção `xiao_const`) |
| 57–58 | momento filme e core (anular) | 2.4.1.2 | `closures/annular.py` |
| 59 | equilíbrio do anular (holdup) | 2.4.1.2 | `closures/holdup.py` |
| 60 | perda de carga do anular | 2.4.1.2 | `closures/annular.py` |
| 61–63 | S_L = πD, S_G = 0, S_i = πD√R_G | 2.4.1.2 | idem |
| 64–65 | entrainment de Wallis (FE, φ) | 2.4.1.2 | idem |
| 66 | R_C | 2.4.1.2 | idem |
| 67–68 | μ_C, ρ_C | 2.4.1.2 | idem |
| 69–70 | v_F, v_C | 2.4.1.2 | idem |
| 71 | τ_i do anular | 2.4.1.2 | idem |
| 72–76 | f_i = f_CS I_θ; I_H (Henstock–Hanratty, F_A); I_V (Wallis) | 2.4.1.2 | idem |
| 77 | drift-flux: v_G = C_0 v_M + v_0 sin θ H_L^0,5, C_0 = 1,15 | 2.4.1.3 | `closures/bubble.py` |
| 78 | v_0 de Harmathy | 2.4.1.3 | idem |
| 79 | v_GS = v_G (1 − H_L) (holdup implícito) | 2.4.1.3 | `closures/holdup.py` |
| 80–81 | perda de carga homogénea; ρ_M | 2.4.1.3 | `closures/bubble.py` |
| 82 | equilíbrio do filme (intermitente) | 2.4.1.4 | `closures/holdup.py` |
| 83 | perda de carga ponderada por L_S/L_U, L_F/L_U | 2.4.1.4 | `closures/intermittent.py` |
| 84 ⚠ | H_LS (Gomez et al. 2000) | 2.4.1.4 | idem |
| 85 | v_TB (Bendiksen 1984) | 2.4.1.4 | idem (partilhável com o mfsim) |
| 86 | v_BS | 2.4.1.4 | idem |
| 87 | L_S (Scott et al.; Xiao et al.) | 2.4.1.4 | idem |
| 88–93 ⚠(89) | v_LS, v_LF, v_GF por conservação de massa | 2.4.1.4 | idem |
| 94–96 | τ_wS, ρ_S, μ_S | 2.4.1.4 | idem |
| 97–99 | L_U e H_L da unidade | 2.4.1.4 | idem |
| 100 | Q_w = −UπD(T − T_amb) | 2.4.2 | `models/*.py` (`energy`) |
| 101–105 | Peng–Robinson; cúbica em Z; A, B | 2.4.3 | `thermo/eos_pr.py` |
| 106–109 | a, b, f(ω, T_r), κ | 2.4.3 | idem |
| 110–112 | regras de mistura clássicas, δ_ij | 2.4.3 | idem |
| 113–117 | balanços do flash; K_i; x_i, y_i | 2.4.4 | `thermo/flash.py` |
| 118 | Rachford–Rice | 2.4.4 | idem |
| 119–120 | φ-φ; definição de φ̂ | 2.4.4 | idem |
| 121 | ln φ̂_k de PR | 2.4.4 | `thermo/eos_pr.py` |
| 122 | K_i = φ̂_L/φ̂_V | 2.4.4 | `thermo/flash.py` |
| 123–124 | critério de paragem; atualização de K | 2.4.4 | idem |
| 125 | estimativa de Wilson | 2.4.4 | idem |
| 126 ⚠ | ρ_k da EoS | 2.5.1 | `thermo/eos_pr.py` |
| 127–128 | Pedersen: estados correspondentes, metano de referência | 2.5.2 | `thermo/viscosity.py` |
| 129–136, Tabela 2 | Hanley–McCarty–Haynes para o metano; η_0, η_1, Δη', Δη'', F_1, F_2 | 2.5.2 | idem |
| 137–141 ⚠(141) | Allan & Teja: Nc_m, A, B, C, η_L | 2.5.2 | idem |
| 142–143 | Brock & Bird | 2.5.3 | `thermo/surface_tension.py` |
| 144 | média molar de σ | 2.5.3 | idem |
| 145 | H_R = H − H_GI | 2.5.4 | `thermo/enthalpy.py` |
| 146–147 ⚠(147) | H_R de PR; da/dT | 2.5.4 | `thermo/eos_pr.py` |
| 148 | H_GI (Smith et al. 2018, Tabela 8) | 2.5.4 | `thermo/enthalpy.py` |
| 149 | H = H_R + H_GI | 2.5.4 | idem |

## Capítulo 3: abordagem numérica

| eq. | conteúdo | secção | módulo |
|---|---|---|---|
| 150–152 | derivada total em (P, T) | 3.1 | `models/noslip.py` |
| 153–155 | massa da mistura expandida | 3.1.1 | idem (não entra no sistema final) |
| 156–158 | momento expandido | 3.1.2 | idem |
| 159 | E_1 P' + E_2 T' = I_1 | 3.1.2 | idem |
| 160–162 ⚠(160) | energia expandida | 3.1.3 | idem |
| 163 | E_3 P' + E_4 T' = I_2 | 3.1.3 | idem |
| 164–165 | sistema final 2×2 | 3.1.4 | idem |
| 166–167 | solução fechada para P', T' | 3.1.4 | idem |
| 168 | forma A x = b | 3.1.4 | idem |
| Tabela 3 | variáveis P, T; equações momento e energia | 3.1.5 | — |
| 169–170 ⚠ | ∂M/∂P, ∂M/∂T por diferenças de 1.ª ordem | 3.1.6 | `models/noslip.py` (opção `finite_difference` \| `autodiff`) |
| Figura 10 | laço do `solve_ivp` (LSODA): 3 flashes por avaliação | 3.1.6 | `solvers/march.py` |
| Tabela 4 ⚠, 171–172 | diferenças de 2.ª ordem; (1/Δx) M Y = B | 3.2 | `solvers/fd.py` |
| 173–176 | massa por fase em vazões: dm_k/dx = M | 3.2.1 | `models/mechanistic.py` |
| 177–178 | Y_massa, B_massa | 3.2.1 | idem |
| 179 | momento da mistura com F(padrão) | 3.2.2 | idem |
| 180–181 | Y_momento, B_momento | 3.2.2 | idem |
| 182–184 ⚠(182) | energia; Y_energia, B_energia | 3.2.3 | idem |
| 185–187 | resíduos | 3.2.4 | idem |
| 188–190 | resíduos discretizados | 3.2.4 | `solvers/fd.py` |
| Tabela 5 | variáveis m_G, m_L, P, T; equações massa G, massa L, momento, energia | 3.2.5 | — |
| Figura 11 ⚠ | estimativa inicial → raízes (Powell híbrido) | 3.2.6 | `solvers/fd.py` (opção `hybr` \| `newton_exact`) |

## Capítulo 5: dados de entrada que viram casos

| tabela | conteúdo | ficheiro planeado |
|---|---|---|
| 6 | fluido de validação termodinâmica (11 componentes) | `cases/thermo/fluid_table6.json` |
| 7–9 | propriedades dos puros (MW, P_c, T_c, ω; H_form, A–D; T_b) | `dissertation/data/components_tables_7_8_9.json` |
| 13–14 | ar–água, tubo de 25 mm, ε 1,53 mm, horizontal e vertical | `cases/patterns/air_water_{h,v}.json` |
| 15–18 | linhas de 100 m; D, P_in, m; CH₄/C7+; P → R_G | `generators/dp_comparison.py` → `cases/dp/` |
| 19–22 | sistema offshore (fluido, C7+, tubulação, contorno) | `cases/offshore/5_5_{noslip,slip}.json` |
| 23–25 | gasoduto com condensado (gás, tubulação, 6 casos) | `generators/condensate.py` → `cases/condensate/` |
| 26 | composições na saída (alvo) | `dissertation/data/table26.json` |
| 34–35 | tempos (alvo de ordem de grandeza) | `docs/issues/` (benefício medido) |
| 36–37 (Apêndice A) | 200 perdas de carga (alvo principal) | `dissertation/data/appendix_a_{horizontal,vertical}.csv` |
