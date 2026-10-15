# Mapeamento: dissertação × mfsim (doutorado)

*Parte do [estudo](README.md) de 2026-10-15. «mfsim» é o simulador em
`/Users/jaimenms/code/personal/doutorado`; «projeto novo» é a reprodução proposta em
[04](04-plano-de-reproducao.md). Caminhos de módulo referem-se ao `doutorado`.*

## Diferenças de natureza

| | dissertação (2022) | mfsim (tese de 2010, reimplementada em 2026) |
|---|---|---|
| regime | permanente (d/dx) | dinâmico (∂/∂t + ∂/∂x), BDF implícito próprio |
| modelo de escoamento | momento da mistura + holdup mecanicista por padrão (com escorregamento) ou homogéneo em equilíbrio (sem) | dois fluidos a 4 equações (massa e momento por fase), mistura Beggs & Brill a 3, monofásicos |
| pressão | única na seção, sem termo interfacial | única com correções de pressão por fase (hidrostática no estratificado) e opções de regularização (Bestion, massa virtual) |
| fluido | composicional: Peng–Robinson + flash P-T, μ, σ, H por correlações; T é incógnita | politrópico por fase, isotérmico, μ e σ constantes |
| energia | sim (entalpia da mistura, U·(T − T_amb)) | não |
| transferência de massa | à taxa do equilíbrio (eq. 25, flash) | nenhuma (fases fixas); plano futuro: Γ explícito por padrão, **não** flash de célula |
| topologia | tubos em série (poço → linha → riser) | grafo (vértices e arestas: fronteira, junção, tanque; duto, choke, reservatório, retenção) |
| incógnitas | P, T (3.1) ou m_G, m_L, P, T por nó (3.2) | α_G, P/P_R, q_L, q_G por nó (+ H_G, P_T por tanque) |
| métodos | PVI em x (LSODA) ou diferenças finitas + Powell sem Jacobiana | Galerkin / diferenças / volumes finitos escalonados em x; BDF em t; estacionário por Newton (Jacobiana colorida ou exata JAX) e pseudo-transiente |
| referência de reprodução | tabelas do Apêndice A e figuras; **sem código nem ficheiros** | código MATLAB e `.mf2` com estados e coeficientes por nó (reprodução bit a bit) |
| validação | contra correlações do HYSYS | contra o MATLAB original e dados de laboratório |

## Conceito a conceito

Legenda da coluna «relação»: **igual** (mesma fórmula), **particular** (a dissertação é um caso
particular do que o mfsim tem, ou vice-versa), **diferente** (correlações distintas para o
mesmo papel), **inexistente** (não há equivalente no mfsim).

| conceito | dissertação | mfsim | relação | decisão proposta |
|---|---|---|---|---|
| fator de atrito | Churchill (50–52), forma Darcy/8 | `friction.fanning` (Churchill, Fanning; regularização `V_EPS`) | igual (f_Fanning = f_Darcy/4) | reutilizar; a regularização em v = 0 é opcional (irrelevante em permanente) |
| geometria do estratificado | h_L/D, ângulo molhado, S_L, S_G, S_i, dA_L/dh_L (43–48) | `geometry.stratified_level` (Newton em sin β − β + 2π α_L = 0) | igual (α da dissertação = β do mfsim) | reutilizar |
| estratificado: atrito interfacial | f_i = 0,0142 constante (Xiao) (56) | Andritsos & Hanratty com `fi_vgt`; `fi_vgt = inf` → Taitel & Dukler f_I = f_G; `fi_mult` | diferente | acrescentar opção `interfacial="xiao_const"` ao fechamento existente; default fiel à dissertação no projeto novo |
| estratificado: diâmetros hidráulicos | D_L = 4A_L/S_L, D_G = 4A_G/(S_L + S_G) (53–54) | D_G = 4A_G/(S_G + S_i) (Taitel & Dukler) | igual se (54) for gralha; diferente se for o que o código faz | confirmar com o autor; implementar as duas como opção |
| estratificado: correções de pressão | nenhuma (pressão única, §2.2.2) | dP_L, dP_G hidrostáticas e derivadas (`closures/stratified.py`) | particular (dissertação = termos nulos) | o modelo permanente não precisa delas; manter a possibilidade de as ligar se o módulo for partilhado |
| equilíbrio do estratificado (holdup) | eq. (41) resolvida em h_L | `closures.equilibrium_fractions` (equilíbrios de Taitel/Pots para a IPE e para `initial_guess`) | particular (mesma equação, usos diferentes) | reutilizar o solver de equilíbrio, com `custom_root` para derivadas em JAX |
| anular | filme + core com gotas (Wallis), f_i = f_CS·I_θ (Henstock–Hanratty, Wallis) (57–76) | `closures/annular.py`: Kowalski no filme, f_I de Andritsos–Hanratty, sem arraste de gotas (issue 024 avaliou Ishii–Mishima e concluiu desprezível no severe slug) | diferente | novo fechamento no projeto novo; comparar os dois numa issue de estudo |
| bolhas | drift-flux C_0 = 1,15 + Harmathy (77–79); atrito homogéneo (80–81) | `closures/bubble.py`: força de arrasto com diâmetro de Hinze e v_b (tese/físico; issue 015) | diferente (algébrico em permanente × força em dinâmico) | novo (simples); em permanente o drift-flux dá o holdup diretamente |
| intermitente | unidade de slug de Gomez et al. 2000: H_LS (84), v_TB Bendiksen (85), L_S Scott/Xiao (87) (82–99) | `closures/intermittent.py`: H_LS Barnea–Brauner, frequência Zabaras, v_T Bendiksen C_0 = 1,2, filme estratificado, pistão como bolhas | parcialmente igual (estrutura e Bendiksen) | novo com reutilização da estrutura; `vTB` partilhado |
| identificação de padrão | Taitel & Dukler 1976 + Taitel 1980, sigmoides de Souza (2010), chave a 45° (Tabela 1, Figura 4) | `closures/regime.py`: equilíbrios + critérios com sigmoides (porta da tese de 2010), issues 020 (quatro defeitos) e 021 (suave e local); `vertical_regime`/`vertical_angle` (45° por omissão)/`vertical_blend` (issue 008) | particular (mesma linhagem; verificar critérios verticais 35–38) | reutilizar a estrutura e as correções; validar contra as Figuras 22–23 |
| salto de holdup na mudança de inclinação | observado e não tratado (p. 133) | `vertical_blend` (mistura C¹ por ângulo, issue 008) | inexistente na dissertação | aplicar como melhoria com bandeira |
| fase que desaparece / aparece | «problemas de convergência» (p. 99) | issue 003 (limitadores), `low_void_regime`/`high_void_regime` (issues 016–017) | inexistente na dissertação | aplicar as ideias (rampas C¹ nos extremos de α) à versão permanente |
| fluido | PR + flash + correlações; composição z | `fluid.Fluid` politrópico, `props(P) → (ρ_L, ρ_G, C_L, C_G)` | inexistente | **novo módulo `thermo/`** com interface `props(P, T, z) → estado de fase` e derivadas; desenhado backend-agnóstico para ser reutilizável pelo mfsim |
| energia | eq. (30), Q_w = −UπD(T − T_amb) (100) | nenhuma | inexistente | **nova** no projeto novo; é a forma mais simples da energia que `docs/future/` do mfsim pede |
| transferência de massa | M_GL = m (∂R_G^EQ/∂P P' + ∂R_G^EQ/∂T T') (25) | nenhuma; plano futuro: Γ_i = k·a·(c_int − c_seio) por padrão | diferente em filosofia (equilíbrio local × cinética) | em permanente o equilíbrio local é a escolha certa; o módulo `thermo/` dá a composição de interface que o Γ explícito do mfsim precisará |
| método 3.1 (PVI em x) | derivadas totais, sistema 2×2, LSODA | nada equivalente (`initial_guess.py` é uma marcha hidrostática grosseira) | inexistente | **novo solver `march`**: caminho fiel com `solve_ivp(LSODA)`; caminho moderno com derivadas por AD e integrador em JAX (decisão em [05](05-questoes-abertas.md)) |
| método 3.2 (diferenças finitas + raízes) | `numpy.gradient` + Powell híbrido, sem Jacobiana | `steady.newton_steady` (Jacobiana esparsa colorida ou exata por `jacfwd`, busca linear, limites) + `steady.pseudo_transient` | particular (mesma formulação de resíduo; solver muito melhor) | **reutilizar o padrão**: resíduo puro `residual(x, params)` + Newton com Jacobiana exata; manter `scipy.optimize.root(hybr)` como opção fiel |
| estimativa inicial para 3.2 | «necessária em todos os pontos» (não diz como) | `initial_guess.py`; receitas em `initial/recipes.py` | inexistente | usar a solução do método 3.1 (não escorregamento) como estimativa inicial do 3.2: resolve a fragilidade declarada |
| backend numérico | NumPy puro | `backend/` (ABC com `while_loop`, `fori_loop`, `safe_sqrt`, `safe_pow`, `sigmoid` com `custom_jvp`, `masked_apply`, pytrees) + `compiled.py` (`jit` + `jacfwd`) | inexistente | copiar o padrão tal e qual; o flash por substituições sucessivas é um `while_loop`; derivadas do flash por `custom_root`/teorema da função implícita |
| casos | argumentos de função; tabelas no texto | JSON `mfsim-case/1` com schema, `$extends`, `--set`; `cases/`, `generators/` | inexistente | schema `cmfsim-case/1` ([esboço](04-plano-de-reproducao.md#o-caso-em-json)); gerador paramétrico para os 200 pontos de 5.4 |
| corridas e lotes | não há | npz v2 com `case_json`; `batches/*.json`; `mfsim batch` | inexistente | copiar o padrão |
| referências de regressão | não há | `tests/golden` níveis A/B/C, `tools/golden_generate.py`, `MANIFEST.sha256` | inexistente | copiar; nível A só entre versões próprias, nível B contra o Apêndice A |
| documentação de alterações | não há | `docs/issues/NNN-*.md` + `INDEX.md` + teste de sincronia | inexistente | copiar o formato (tipo, área, estado, onde, antes, depois, benefício medido) |
| notebooks | não há | `notebooks_src/{thesis,issues}` que só leem corridas | inexistente | copiar: um notebook por secção 5.x |
| validação | HYSYS | MATLAB + laboratório (Jansen, Fabre, Tengesdal) | — | sem HYSYS, a referência termodinâmica independente é outra implementação de PR (CoolProp/`thermo`), e a fluidodinâmica fica contra o Apêndice A e as figuras |

## O que se transfere do doutorado e o que não

Transfere-se como **padrão** (copiar a arquitetura, não necessariamente o código):
`backend/` e `compiled.py`; `io/` (schema, `extends`, validação); `runs.py`; `cli.py`;
`batches`; `tests/golden` e `tools/golden_generate.py`, `tools/manifest.py`; `docs/issues`;
`notebooks_src`; o método descrito em `docs/method_thesis_to_modern_code.md`; as regras do
`CLAUDE.md` (solver próprio; melhorias com bandeira e default fiel; nada apagado sem
aprovação; identificadores em inglês, textos em português; um só caminho de execução).

Transfere-se como **código** (backend-agnóstico, sem dependências do grafo): `friction.py`,
`geometry.stratified_level`, a função sigmoide com `custom_jvp`, `safe_sqrt`/`safe_pow`, os
critérios de Taitel em `closures/regime.py` (a verificar), `steady.newton_steady` como
esqueleto do solver do método 3.2.

**Não** se transfere (específico da dinâmica): BDF próprio, formulação conservativa (issue
004), viscosidade artificial (007), tolerâncias por faixa (034), fase virtual (006), a
montagem da rede em grafo (`network/`) na primeira versão (tubos em série bastam para a
dissertação; o grafo pode vir depois).

## O que o mfsim ganha com a reprodução (informativo: o dono não pretende aproveitar, decisão A11)

1. **Módulo termodinâmico composicional** (PR, flash, propriedades, derivadas por AD) que o
   plano `docs/future/compositional_mass_transfer.md` pressupõe na interface («a
   termodinâmica entra só na interface», composição de interface pelo flash a (P, T)) e que
   o plano do CO₂ precisa para as impurezas.
2. **Equação de energia** em permanente, primeiro degrau da energia que os planos
   criogénico e CO₂ listam como segundo passo.
3. **Estado estacionário composicional como condição inicial** de corridas dinâmicas: o
   caso 5.5 (linha 6000 m + riser 300 m) tem a geometria do severe slug; um perfil P, T,
   holdup em equilíbrio é uma receita de condição inicial melhor do que `guess`.
4. **Modelo homogéneo em equilíbrio** (não escorregamento) como o HEM que a literatura do
   CO₂ usa de base de comparação.
5. Um segundo caso de uso da mesma infraestrutura (JSON, lotes, goldens, issues), que
   testa se o padrão generaliza.
