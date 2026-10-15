# cmfsim — escoamento bifásico composicional em tubulações, regime permanente

Reimplementação moderna (Python, numpy/JAX, arquitetura do `mfsim`) da dissertação de mestrado
de **Gustavo Alves de Carvalho** (UERJ, 2022), *Modelagem e simulação composicional do
escoamento multifásico em tubulações com geração de modelos substitutos*: termodinâmica
composicional (Peng–Robinson, flash P-T, viscosidades, tensão superficial, entalpia), modelo
de não escorregamento resolvido como problema de valor inicial em x, modelo mecanicista com
escorregamento resolvido por diferenças finitas, identificação de padrão de Taitel–Dukler,
validação contra os 200 casos do Apêndice A e as figuras digitalizadas.

O estudo que precedeu o código, o mapeamento com o `mfsim`, a avaliação crítica, o plano e as
decisões estão em [docs/README.md](docs/README.md). O guia de sessão é [CLAUDE.md](CLAUDE.md).

## Começar

```bash
uv sync
uv run pytest                                                  # rotina (~2 min)
uv run python -m generators.condensate                         # casos JSON de §5.6
uv run cmfsim validate cases/condensate/*.json
uv run cmfsim run cases/condensate/condensate_5_6_case2.json --out runs/case2.npz --backend jax
uv run cmfsim batch batches/dissertation.json                  # lote §5.6 -> runs/dissertation
uv run python -m generators.dp_comparison && uv run cmfsim batch batches/dp_comparison.json   # §5.4: 400 corridas -> runs/dp_comparison
uv run python notebooks_src/thesis/nb_5_6_condensate.py        # comparação com as Figuras 38–40 (lê runs/)
uv run python notebooks_src/thesis/nb_5_4_dp_comparison.py     # comparação com o Apêndice A
uv run python -m dissertation.extract                          # regenera os CSV do Apêndice A a partir do PDF
```

Um caso é um ficheiro JSON `cmfsim-case/1` autocontido (fluido com as propriedades dos componentes,
tubo único com perfil axial, condições de entrada, modelo, solver); variantes herdam com `$extends`
e `--set caminho=valor` varia sem ficheiro novo. Uma corrida gravada (`runs/*.npz`) traz os perfis e
o caso completo (`cmfsim.load_run`).

## Estado

| fase | conteúdo | estado |
|---|---|---|
| 0 | rede de segurança: git (raiz do monorepo), `uv`, esqueleto, backend copiado, Apêndice A e tabelas em `dissertation/data`, figuras digitalizadas em `references/figures` | feito (2026-10-15) |
| 1 | `cmfsim/thermo`: PR, flash, propriedades, derivadas exatas em JAX; validação contra a biblioteca `thermo` | feito (issues 001–002) |
| 2 | padrões e fechamentos | feito (issue 003) |
| 3 | modelo de não escorregamento + marcha em x (integrador próprio); caso 5.6 contra as Figuras 38–40 | feito (issue 004); achado térmico em aberto |
| 4 | modelo mecanicista com escorregamento + diferenças finitas + Newton com Jacobiana colorida; consistência com a marcha; lote dos 200 casos do Apêndice A (`batches/dp_comparison.json`) | feito (issue 005); comparação com o Apêndice A em curso |
| 5 | melhorias JAX (AD, `vmap`, marcha com escorregamento, IPE suave); parâmetros como argumentos (uma compilação por estrutura) já feito | em curso |
| 6 | notebooks, goldens, síntese | — |

## Licença

MIT ([LICENSE](LICENSE)).
