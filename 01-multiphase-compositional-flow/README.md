# cmfsim — escoamento bifásico composicional em tubulações, regime permanente

Reimplementação moderna (Python, numpy/JAX, arquitetura do `mfsim`) da dissertação de mestrado
de **Gustavo Alves de Carvalho** (UERJ, 2022), *Modelagem e simulação composicional do
escoamento multifásico em tubulações com geração de modelos substitutos*: termodinâmica
composicional (Peng–Robinson, flash P-T, viscosidades, tensão superficial, entalpia), modelo
de não escorregamento e modelo mecanicista com escorregamento, cada um resolúvel por marcha em x
(RK45), diferenças finitas ou volumes finitos, em numpy ou JAX, com balanços globais em toda a
corrida; identificação de padrão de Taitel–Dukler,
validação contra os 200 casos do Apêndice A e as figuras digitalizadas.

O estudo que precedeu o código, o mapeamento com o `mfsim`, a avaliação crítica, o plano e as
decisões estão em [docs/README.md](docs/README.md). O guia de sessão é [CLAUDE.md](CLAUDE.md).

## Começar

```bash
git submodule update --init                                    # código original da dissertação em legacy/masters_code (só consulta)
uv sync
uv run pytest                                                  # rotina (~2 min)
uv run python -m generators.condensate                         # casos JSON de §5.6
uv run cmfsim validate cases/condensate/*.json
uv run cmfsim run cases/condensate/condensate_5_6_case2.json --out runs/case2.npz --backend jax
uv run cmfsim batch batches/dissertation.json                  # lote §5.6 -> runs/dissertation
uv run cmfsim batch batches/solver_matrix.json                 # 2 modelos x 3 solvers no §5.5 e no caso 3 da §5.6 -> runs/solver_matrix
uv run cmfsim batch batches/convergence.json                   # convergência de malha (n = 6..192) -> runs/convergence
uv run cmfsim run cases/offshore/offshore_5_5_slip.json --set solver.type=fv --backend jax   # qualquer caso com march | fd | fv
uv run python -m generators.dp_comparison && uv run cmfsim batch batches/dp_comparison.json   # §5.4: 400 corridas -> runs/dp_comparison
uv run python notebooks_src/thesis/nb_5_6_condensate.py        # comparação com as Figuras 38–40 (lê runs/)
uv run python notebooks_src/thesis/nb_5_4_dp_comparison.py     # comparação com o Apêndice A
uv run python -m dissertation.extract                          # regenera os CSV dos Apêndices A, B e C a partir do PDF
uv run python -m generators.surrogate                          # §5.7: casos dos três trechos e do sistema com o fluido da Tabela 6
uv run python -m surrogate.pipeline                            # §5.7: massa de dados (march_many), grelha de 135 redes e modelos finais -> runs/surrogate, models/surrogate
uv run python -m surrogate.predict models/surrogate/well.json models/surrogate/line.json models/surrogate/riser.json --P 170 --T 353.15 --m 20
uv run python -m surrogate.pipeline --steps profiles,profile_models,pinn   # perfis finos, substituto de perfil (x como entrada) e estudo PINN (issue 010)
make -C papers/01-compositional-steady-state                   # artigo 1: figuras (de runs/) + PDF em build/main.pdf
```

## Notebooks de reprodução (um por secção do capítulo 5)

Fontes em `notebooks_src/thesis/` (células `# %%`), executados por `uv run cmfsim nb <fonte> --execute`
para `notebooks/thesis/*.ipynb`; nenhum simula: leem as corridas dos lotes (`runs/`), as figuras
digitalizadas (`references/figures/data/`) e os dados da dissertação (`dissertation/data/`), e
reproduzem todas as figuras e tabelas da secção com o cmfsim sobreposto.

| notebook | secção | reproduz | corridas necessárias |
|---|---|---|---|
| `nb_5_1_thermo` | 5.1 | Figura 13 (envelope e 159 pontos), Figuras 14–18 (quantis contra o eixo HYSYS e o eixo da dissertação), Tabelas 10–11 (R² contra o oráculo `thermo`) | nenhuma (cálculo termodinâmico instantâneo) |
| `nb_5_2_viscosity_tension` | 5.2 | Figuras 19–21 (quantis contra o eixo HYSYS, com a fusão de marcadores simulada), Tabela 12, regras de mistura de Allan & Teja (issue 008) | nenhuma |
| `nb_5_3_patterns` | 5.3 | Figuras 22–23 (mapas ar–água ponto a ponto e acerto por classe), Tabelas 13–14, sensibilidade às leituras | nenhuma |
| `nb_5_4_dp_comparison` | 5.4 | Figuras 24–30, Tabelas 15–18, Apêndice A (desvios por modelo) | `batches/dp_comparison.json` |
| `nb_5_5_offshore` | 5.5 | Figuras 31–37, Tabelas 19–22, matriz de métodos (6 caminhos sobrepostos, balanços, paridade numpy × JAX) | `batches/dissertation.json` (offshore), `batches/solver_matrix.json` |
| `nb_5_6_condensate` | 5.6 | Figuras 38–40, Tabelas 23–26, ajuste do fator térmico, salto da Figura 40, matriz de métodos no caso 3 | `batches/dissertation.json`, `batches/solver_matrix.json` |
| `nb_5_7_surrogate` | 5.7 | Tabelas 27–33, Figuras 41–58, Apêndices B e C (amostras do autor simuladas pelo cmfsim; grelha de hiperparâmetros) | `surrogate.pipeline` (`runs/surrogate/`, `models/surrogate/`) |
| `nb_5_8_performance` | 5.8 | Tabelas 34–35, Figura 59, tempos da matriz de métodos e convergência de malha (n = 6 … 192) | `batches/performance.json`, `batches/solver_matrix.json`, `batches/convergence.json` |
| `studies/nb_surrogate_profile_pinn` | além da dissertação | substituto de perfil (s = x/L como entrada) e PINN com o resíduo do modelo em dados escassos (issue 010) | `surrogate.pipeline --steps profiles,profile_models,pinn` |

O modelo substituto (capítulo 4, §5.7) vive em [surrogate/](surrogate/), fora do simulador: amostragem por
hipercubo latino, massa de dados gerada por `march_many` (uma compilação, `vmap` sobre as amostras),
rede multicamada com Adam em JAX, busca em grelha e predição encadeada poço → linha → riser; os modelos
empíricos treinados ficam em [models/surrogate/](models/surrogate/) (JSON, carregados por `surrogate.model.MLP.load`).
Duas extensões (issue 010): `surrogate.profile` recebe também a posição axial e devolve o perfil P(x), T(x) com a
entrada exata por construção (`ProfileMLP`), e `surrogate.pinn` treina esse modelo com o resíduo das equações do
não escorregamento em pontos de colocação não simulados, sem retropropagar pelo flash.

Um caso é um ficheiro JSON `cmfsim-case/1` autocontido (fluido com as propriedades dos componentes,
tubo único com perfil axial, condições de entrada, modelo, solver); variantes herdam com `$extends`
e `--set caminho=valor` varia sem ficheiro novo. Uma corrida gravada (`runs/*.npz`) traz os perfis e
o caso completo (`cmfsim.load_run`).

Os artigos para revistas ficam em [papers/](papers/README.md), um por pasta, construídos com `make`
(LaTeX Elsevier CAS, figuras geradas por código a partir das corridas); a revista escolhida e o seu guia de
autores estão em [papers/00-escolha-da-revista.md](papers/00-escolha-da-revista.md).

## Estado

| fase | conteúdo | estado |
|---|---|---|
| 0 | rede de segurança: git (raiz do monorepo), `uv`, esqueleto, backend copiado, Apêndice A e tabelas em `dissertation/data`, figuras digitalizadas em `references/figures` | feito (2026-10-15) |
| 1 | `cmfsim/thermo`: PR, flash, propriedades, derivadas exatas em JAX; validação contra a biblioteca `thermo` | feito (issues 001–002) |
| 2 | padrões e fechamentos | feito (issue 003) |
| 3 | modelo de não escorregamento + marcha em x (integrador próprio); caso 5.6 contra as Figuras 38–40 | feito (issue 004; perfis térmicos da §5.6 reproduzidos com U_eff = 3,42 × U da Tabela 25) |
| 4 | modelo mecanicista com escorregamento + diferenças finitas + Newton com Jacobiana colorida; consistência com a marcha; lote dos 200 casos do Apêndice A (`batches/dp_comparison.json`) | feito (issue 005); comparação com o Apêndice A em curso |
| 5 | melhorias JAX (AD, `vmap`, IPE suave); parâmetros como argumentos e arranque quente do flash (issues 006–007); matriz de métodos 2 modelos × 3 solvers (marcha, diferenças finitas, volumes finitos) × 2 backends com balanços globais (issue 011) | feito; IPE suave por fazer |
| 6 | notebooks de reprodução por secção do capítulo 5 (5.1–5.8), goldens, síntese | notebooks feitos; goldens de texto por fazer |
| 7 | modelo substituto (cap. 4, §5.7): `surrogate/` (amostragem, massa de dados por `march_many`, MLP + Adam em JAX, grelha, encadeamento), casos e modelos treinados, notebook 5.7 | feito (issue 009) |

## Licença

MIT ([LICENSE](LICENSE)).
