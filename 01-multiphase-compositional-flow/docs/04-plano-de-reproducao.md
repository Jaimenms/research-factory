# Plano de reprodução (proposta, nada iniciado)

*Parte do [estudo](README.md) de 2026-10-15, atualizado no mesmo dia com as decisões do dono
([05](05-questoes-abertas.md#respostas-do-dono-2026-10-15)). Segue o método de
`doutorado/docs/method_thesis_to_modern_code.md` adaptado ao que a dissertação tem e não tem.
Pacote `cmfsim`, no monorepo `research-factory/`.*

## Princípios herdados do doutorado

1. **Solver próprio em tudo** (decisão A5): PR, flash, fechamentos, holdup, resíduos, Newton
   **e o integrador em x** (RK encaixado adaptativo; BDF próprio se a rigidez o exigir),
   escritos sobre o backend para compilar em JAX. Bibliotecas externas só como oráculo de
   teste (`thermo`, CoolProp). A fidelidade à dissertação vem de tolerâncias apertadas (a
   solução da EDO é a mesma), não de usar o LSODA; o Powell do SciPy e o `numpy.gradient`
   podem ficar como opções de comparação, não como caminho principal.
2. **Toda a melhoria é uma opção com bandeira**; o default reproduz a dissertação, gralhas
   incluídas até o estudo decidir.
3. **Um só caminho de execução**: `simulate(case)` lê tudo do JSON; notebooks só leem corridas
   de lotes.
4. **Física escrita uma só vez** sobre o backend injetado (numpy/JAX), sem ramos em valores,
   para ter `jit`, `jacfwd` e `vmap` de graça.
5. **Uma issue por alteração** (tipo, área, estado, onde, antes, depois, benefício medido),
   índice sincronizado por teste.
6. Identificadores e chaves JSON em inglês; comentários, documentação e mensagens em
   português; nada apagado sem aprovação; cada fase fecha com suíte verde, `black`, docs e
   commit com tag.

## Estrutura de pastas proposta

```
research-factory/                       # raiz do git (monorepo, decisão A4)
└── 01-multiphase-compositional-flow/
    ├── README.md  CLAUDE.md  LICENSE  pyproject.toml  uv.lock  .python-version  MANIFEST.sha256
    ├── references/                 # o PDF e as figuras digitalizadas (A7): figures/*.json, figures/README.md
    ├── cmfsim/                     # o simulador, genérico: só lê casos JSON (cmfsim-case/1)
│   ├── backend/                #   numpy | jax (base.py, numpy_backend.py, jax_backend.py: cópia do mfsim)
│   ├── thermo/                 #   components.py (base de dados, pseudo-componentes), eos_pr.py (Z, phî, H_R, rho),
│   │                           #   flash.py (Rachford–Rice + substituições sucessivas + Wilson; opções),
│   │                           #   viscosity.py (Pedersen; Allan–Teja), surface_tension.py (Brock–Bird),
│   │                           #   enthalpy.py (gás ideal + residual), state.py (PhaseState: beta, x, y, rho_k, mu_k, h_k, sigma)
│   ├── closures/               #   geometry.py, friction.py (cópia do mfsim); stratified.py, annular.py, bubble.py,
│   │                           #   intermittent.py; regime.py (Taitel–Dukler 1976, Taitel 1980, sigmoides, opções);
│   │                           #   holdup.py (equilíbrio por padrão -> holdup e dP/dx)
│   ├── models/                 #   base.py; noslip.py (E1..E4, I1, I2 da §3.1); mechanistic.py (resíduos de massa, momento, energia da §3.2)
│   ├── solvers/                #   march.py (PVI em x: integrador próprio, RK encaixado adaptativo | BDF próprio);
│   │                           #   fd.py (malha, diferenças, resíduos, Newton com Jacobiana exata | hybr como comparação);
│   │                           #   newton.py (de mfsim/steady.py); shooting.py (saída especificada no método 3.1)
│   ├── pipe.py                 #   um único tubo com parâmetros que variam axialmente (decisão A9): perfil por pontos de quebra
│   │                           #   em x (D, ângulo, rugosidade, U, T_amb constantes por troço); malha e eventos nos pontos de quebra
│   ├── io/                     #   spec.py, schema/cmfsim-case-1.schema.json, extends.py ($extends, $ref), validate.py
│   ├── compiled.py  results.py  runs.py  cli.py  plotting.py
├── dissertation/               # ponte com a dissertação, fora do simulador
│   ├── data/                   #   appendix_a_horizontal.csv, appendix_a_vertical.csv (extraídos do PDF por script),
│   │                           #   components_tables_7_8_9.json, pseudo_c7plus_table_20.json, fluids_tables_6_17_19_23.json,
│   │                           #   figures/ (digitalizações, se feitas), README.md (proveniência)
│   ├── extract.py              #   pymupdf -> csv/json, reprodutível e testado (200 linhas, somas de controlo)
│   └── notebook_utils.py
├── generators/                 # dp_comparison.py (Tabela 16: 4x5x5 x 2 orientações), condensate.py (Tabela 25), offshore.py
├── cases/                      # thermo/fluid_table6.json; patterns/air_water_h.json, air_water_v.json;
│                               # dp/ (gerados); offshore/5_5_noslip.json, 5_5_slip.json; condensate/5_6_case1..6.json
├── batches/                    # dissertation.json (todas as secções 5.x) ; issues.json
├── runs/                       # saídas dos lotes (fora do git, manifesto)
├── notebooks_src/{thesis,issues}/   notebooks/
├── tests/                      # rotina (minutos) e noturna; golden/ (níveis A/B/C, CHANGELOG.md)
├── tools/                      # golden_generate.py, manifest.py, make_nb.py
└── docs/                       # este estudo; issues/NNN-*.md + INDEX.md; future/
```

(A árvore acima está indentada sob `01-multiphase-compositional-flow/`; os níveis abaixo de
`cmfsim/` seguem o mesmo padrão.) `references/` já existe com o PDF e as figuras
digitalizadas; `src/` (vazia) torna-se `cmfsim/` ou é removida, à escolha do dono.

## O caso em JSON

Esboço do schema `cmfsim-case/1` (secções: `fluid`, `pipeline`, `boundary`, `model`,
`solver`, `outputs`, `meta`), com o caso 1 da secção 5.6 como exemplo:

```json
{
 "schema": "cmfsim-case/1",
 "name": "condensate_5_6_case1",
 "description": "Gasoduto de 8 km e 12 in, gás rico (Tabela 23), T_in 60 °C, U = 1 W/m²K (Tabela 25, caso 1).",
 "fluid": {
  "type": "compositional",
  "eos": {"type": "peng_robinson", "kij": "zero"},
  "components": [{"id": "C1", "z": 0.714}, {"id": "C2", "z": 0.152}, {"id": "C3", "z": 0.078},
                 {"id": "nC4", "z": 0.039}, {"id": "nC5", "z": 0.017}],
  "properties": {"viscosity_gas": "pedersen", "viscosity_liquid": "allan_teja",
                 "surface_tension": "brock_bird", "enthalpy": "ideal_gas_plus_residual"},
  "flash": {"method": "successive_substitution", "init": "wilson", "tol": 1e-10, "stability_test": false}
 },
 "pipe": {"profile": [
   {"x": 0.0, "D": 0.3048, "roughness": 4.57e-5, "angle_deg": 0.0, "U": 1.0, "T_amb": 298.15},
   {"x": 8000.0}],
  "n": 80},
 "boundary": {"inlet": {"P": 7.0e6, "T": 333.15, "m": 15.0}},
 "model": {"type": "noslip"},
 "solver": {"type": "march", "integrator": "rk45", "rtol": 1e-8, "atol": 1e-10,
            "derivatives": {"type": "finite_difference", "dP": 1000.0, "dT": 0.01}, "backend": "numpy"},
 "outputs": {"dx_out": 100.0},
 "meta": {"source": "dissertação §5.6, Tabelas 23–25, Figuras 38–40"}
}
```

O tubo é **um só** (decisão A9): `profile` é uma lista de pontos de quebra em x; cada entrada
fixa D, ângulo, rugosidade, U e T_amb desde o seu x até ao x da entrada seguinte, e a última
só diz onde o tubo acaba. O sistema offshore de §5.5 fica

```json
 "pipe": {"profile": [
   {"x": 0.0,    "D": 0.127,  "roughness": 2.0e-4, "angle_deg": 90.0, "U": 7.0, "T_amb": 313.15},
   {"x": 1500.0, "D": 0.1524, "roughness": 6.0e-4, "angle_deg": 0.0,  "U": 5.0, "T_amb": 285.15},
   {"x": 7500.0, "D": 0.1524, "roughness": 6.0e-4, "angle_deg": 90.0, "U": 5.0, "T_amb": 285.15},
   {"x": 7800.0}],
  "n": 78}
```

e a marcha trata os pontos de quebra como eventos (reinício do passo), a malha das
diferenças finitas coloca neles um nó. A mudança de diâmetro num ponto de quebra conserva
m e impõe continuidade de P e T (a dissertação não diz outra coisa).

Uma variante com escorregamento herda e traz só o que muda:

```json
{"schema": "cmfsim-case/1", "$extends": "../offshore/5_5_noslip.json", "name": "offshore_5_5_slip",
 "model": {"type": "mechanistic",
           "regime": {"type": "automatic", "switch": "sigmoid", "k": 20.0, "vertical_angle_deg": 45.0, "blend_deg": 0.0},
           "closures": {"stratified": {"interfacial": "xiao_const", "hydraulic_diameter_gas": "as_printed"},
                        "annular": {"entrainment": "wallis"}, "bubble": {"C0": 1.15},
                        "intermittent": {"slug_holdup": "gomez2000_as_printed"}}},
 "solver": {"type": "fd", "n": 78, "edge_order": 1, "root": "newton_exact", "initial_guess": "march_noslip"}}
```

Especificação na saída (só no método 3.2 na dissertação; no 3.1 por *shooting* como
melhoria): `"boundary": {"inlet": {"T": ...}, "outlet": {"P": ..., "m": ...}}`.

## Fases e portas de saída

| fase | conteúdo | alvo / porta de saída | sessões |
|---|---|---|---|
| 0 | rede de segurança: `git init` na raiz de `research-factory/`, `uv`, `pyproject`, esqueleto do pacote, backend copiado, schema mínimo, `tests/golden` e `tools/`, manifesto, `CLAUDE.md`, `docs/issues/INDEX.md`; `dissertation/extract.py` → CSV do Apêndice A e JSON das Tabelas 6–9, 17, 19, 20, 23; as figuras digitalizadas (A7) já estão em `references/figures/` | CSVs com 100 + 100 linhas conferidas contra o PDF (somas de controlo); suíte verde | 1 |
| 1 | `thermo/`: componentes, PR (raízes de Z com seleção de fase sem ramos, φ̂, H_R, ρ), flash, μ_G, μ_L, σ, H; backend-agnóstico; derivadas por AD com `custom_root` | 159 pontos do fluido da Tabela 6 contra outra implementação de PR (1e-8); paridade numpy × JAX (1e-12); AD contra diferenças; Figuras 14–21 refeitas; R² ≥ os das Tabelas 10–12 | 2 |
| 2 | padrões e fechamentos: geometria e Churchill copiados; critérios (32–38) com sigmoides; holdup por padrão (41, 59, 79, 82) e dP/dx (42, 60, 80, 83); regime fixo e automático | mapas ar–água das Figuras 22–23 (as curvas de transição são analíticas); testes unitários de cada fechamento contra cálculo à mão; igualdade a 1e-14 com o mfsim onde a fórmula é a mesma | 2 |
| 3 | modelo de não escorregamento + método 3.1 (`march`): derivadas totais, E/I, sistema 2×2, integrador próprio (RK encaixado adaptativo com eventos nos pontos de quebra; BDF próprio se preciso); energia com U; tubo único com perfil axial | §5.6: seis casos, Figuras 38–40 digitalizadas, Tabela 26; §5.5 sem escorregamento (Figuras 32–34); ordem de grandeza da Tabela 34 | 1–2 |
| 4 | modelo com escorregamento + método 3.2 (`fd`): diferenças de 2.ª ordem próprias (equivalentes ao `gradient`), resíduos, Newton com Jacobiana exata (JAX) como caminho principal e Powell do SciPy só como comparação; condições na entrada ou na saída; estimativa inicial pelo método 3.1; estudo da coluna Beggs & Brill (A10) | **Apêndice A**: 200 casos a ~1 % (gerador + lote); §5.5 com escorregamento (Figuras 32–37); estudo de malha; estudo das gralhas (54) e (84) pelos números | 2 |
| 5 | melhorias JAX: AD nas derivadas totais; `custom_root` no flash e no holdup; `jit`/`vmap` no lote; marcha com escorregamento (`mass_split = algebraic`); *shooting*; IPE suave e local; mistura por ângulo; teste de estabilidade | paridade com o caminho fiel; tabela de tempos contra as Tabelas 34–35; issues com benefício medido | 2 |
| 6 | consolidação: notebooks (um por secção 5.x), goldens, docs, síntese de uma página («o que a reprodução mostrou») | noturna verde; README; achados | 1 |

Total: 11–12 sessões. Ordem obrigatória 0 → 1 → 2 → 3 → 4; a fase 5 pode começar em
paralelo com a 4 (as derivadas por AD servem logo ao método 3.1).

## Alvos de validação e como obtê-los

| alvo | origem | extração | nível |
|---|---|---|---|
| 200 perdas de carga (Dissertação, Beggs & Brill, Tulsa, HTFS, Gregory/Aziz) | Tabelas 36–37, p. 168–172 | `pymupdf` → CSV, com teste de contagem e de monotonia (ΔP cresce com m) | B (~1 %) |
| propriedades de 11 componentes + C7+ | Tabelas 7–9, 20 | JSON transcrito, com teste contra o PDF | — |
| composições dos quatro fluidos | Tabelas 6, 17, 19, 23 | JSON | — |
| P → R_G na entrada (fluido de 5.4) | Tabela 18 | 5 pares | B |
| composição na saída do gasoduto | Tabela 26 | 15 valores | B |
| mapas de padrão ar–água | Figuras 22–23 | regenerar as curvas de Taitel–Dukler/Taitel (fórmulas fechadas) | exato |
| perfis P, T, holdup, fração de líquido, CO₂, padrões (5.5) | Figuras 32–37 | **digitalizadas** em `references/figures/` (A7) | C |
| perfis P, T, fração vaporizada (5.6) | Figuras 38–40 | **digitalizadas** | C |
| paridade termodinâmica (5.1–5.2) | Figuras 14–21 | **digitalizadas** (pares dissertação × HYSYS) + referência independente | termo |
| tempos | Tabelas 34–35 | — | ordem de grandeza |
| coluna Beggs & Brill do Apêndice A | HYSYS | o mfsim já tem Beggs & Brill (`drift_flux_bb`); estudo opcional | C |

## Melhorias como opções com bandeira

| chave (esboço) | default (dissertação) | alternativa (melhoria) | issue prevista |
|---|---|---|---|
| `solver.derivatives.type` | `finite_difference` (ΔP, ΔT) | `autodiff` | derivadas exatas no método 3.1 |
| `fluid.flash.method` | `successive_substitution` | `newton`, `accelerated` | flash rápido |
| `fluid.flash.stability_test` | `false` | `true` (Michelsen) | fase única robusta |
| `solver.root` (fd) | `newton_exact` (JAX) | `newton_colored`; `scipy_hybr` só para comparação | Jacobiana exata (A5) |
| `solver.initial_guess` (fd) | `flat` | `march_noslip` | estimativa inicial |
| `solver.integrator` (march) | `rk45` próprio (tolerância apertada) | `bdf` próprio; `scipy_lsoda` só para comparação | integrador próprio (A5) |
| `boundary.outlet` com `march` | não permitido | *shooting* | pressão na saída no método rápido |
| `model.regime.switch` | `sigmoid` (k fixo) | `smooth_local` (issues 020–021 do mfsim) | IPE suave |
| `model.regime.blend_deg` | `0` | `> 0` (issue 008) | saltos de holdup |
| `closures.stratified.interfacial` | `xiao_const` | `andritsos_hanratty`, `taitel_dukler` | sensibilidade |
| `closures.stratified.hydraulic_diameter_gas` | `as_printed` | `taitel_dukler` | gralha (54) |
| `closures.intermittent.slug_holdup` | `gomez2000_as_printed` | `gomez2000` | gralha (84) |
| `energy.gravity_term` | `eq30` | `eq160` | sinal do termo gravítico |
| `mass_transfer.basis` | `mass` | `mole`, `volume` | base de (25) |
| `solver.edge_order` (fd) | `1` (`numpy.gradient`) | `2` (Tabela 4) | ordem nos extremos |
| `pipeline.segments[].grading` | uniforme | graduada (issue 009) | malha |
| `energy.kinetic` | `false` | `true` | termo cinético |
| `model.mass_split` | `ode` (eq. 25 discretizada) | `algebraic` (m_G = m·w_G; permite `march` com `mechanistic`) | marcha com escorregamento |

## O que ficou adiado e como o plano o preparou

*Feito em 2026-10-15 (issue 008): `surrogate/` fora do simulador, com `march_many` a gerar a massa de dados,
rede e Adam em JAX, grelha de 135 redes e notebook 5.7; os Apêndices B e C extraídos para CSV servem de
comparação direta.* O texto original do plano:

Capítulo 4 e secção 5.7 (modelo substituto) e apêndices B–C. O plano deixa tudo pronto:
`vmap` sobre `simulate` gera as 300 amostras por trecho (Tabelas 27–29) em segundos; cada
amostra é uma corrida gravada com `case_json`; a derivada do solver em relação às entradas
(AD) fica disponível para substitutos com gradiente. Quando chegar a vez: `surrogate/` fora
do simulador (amostragem por hipercubo latino, rede multicamada em JAX ou PyTorch, busca em
grelha), um lote `batches/surrogate.json`, um notebook por trecho.

## Riscos

- **Ambiguidades sem código** (confirmado: não há código nem acesso ao autor): as gralhas de
  [03](03-avaliacao-e-interpretacao.md) só se resolvem pelos números do Apêndice A e pelas
  figuras digitalizadas, e a solução pode não ser única.
- **Dados em falta** para 5.5 (Cp do C7+, números de carbono efetivos, k_ij): o perfil de T
  do caso offshore pode não ser reprodutível exatamente ([05](05-questoes-abertas.md)).
- **Colunas do HYSYS** não são reproduzíveis (Tulsa, HTFS, Gregory, Aziz não serão
  implementados); só a coluna «Dissertação» e, opcionalmente, Beggs & Brill.
- **Flash perto do orvalho** (5.6) e do crítico: sensibilidade numérica; o teste de
  estabilidade é a mitigação.
- **Digitalização** de figuras: 2–5 % de erro inerente.
- **Deriva de âmbito** para o dinâmico: o grafo e o BDF ficam fora até a reprodução fechar.
