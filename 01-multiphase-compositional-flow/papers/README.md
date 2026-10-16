# Artigos do tema 01

Um artigo por pasta `NN-<slug>/`, todos com a mesma construção e a partilhar bibliografia, estilo de figuras e
regras de `make`. A revista alvo e o seu guia de autores estão em
[00-escolha-da-revista.md](00-escolha-da-revista.md) (*Geoenergy Science and Engineering*, Elsevier, modelo
CAS, citações autor–ano).

| pasta | conteúdo |
|---|---|
| `_common/` | `paper.mk` (regras de `make`), `figlib.py` (apoio às figuras: caminhos, estilo, paleta, gravação PDF+PNG), `cmfsim-paper.mplstyle`, `references.bib` (bibliografia canónica do tema), `latex/` (ícones da classe CAS) |
| `_template/` | esqueleto de artigo novo (`tools/new_paper.py` copia-o) |
| `01-compositional-steady-state/` | artigo 1: o simulador composicional em regime permanente com 2 modelos × 3 solvers verificados entre si e JAX em destaque, o modelo substituto (§5.7) e a extensão de perfil + PINN (issue 010); manuscrito completo, 9 figuras e 11 tabelas geradas |

Dentro de cada artigo:

| ficheiro | papel |
|---|---|
| `main.tex` | o manuscrito (classe `cas-sc`; inglês; comentários em português) |
| `refs.bib` | referências só deste artigo; as comuns vêm de `../_common/references.bib` |
| `figures/make_figures.py` | **um só guião** gera as figuras em `figures/generated/` (PDF vetorial + PNG) e os fragmentos de tabela em `tables/generated/` (`<nome>.tex` com o `tabular` e `<nome>_note.tex` com a nota, lidos por `\input` no `main.tex`) **só a partir de corridas gravadas** em `runs/`, dos modelos treinados em `models/`, das figuras digitalizadas e dos dados da dissertação; não simula. `figlib.run_cli(FIGURES, descrição, tables=TABLES)` trata a linha de comando (`--out`, `--tables`, `--only`) |
| `figures/static/` | figuras que não vêm de código (esquemas), no git |
| `figures/generated/`, `tables/generated/`, `build/` | saídas reproduzíveis, fora do git |
| `Makefile` | três linhas: `PAPER = main` e `include ../_common/paper.mk` |

## Construir

```bash
cd papers/01-compositional-steady-state
make check        # que motor LaTeX vai usar e onde está o projeto
make              # figuras e tabelas (se o guião mudou) + PDF em build/main.pdf
make refigures    # força as figuras e as tabelas (depois de novas corridas em runs/ ou modelos em models/)
make pdf-only     # só o LaTeX
make clean        # apaga build/ ; make distclean apaga também as figuras geradas
```

Motor: `latexmk` (TeX Live/MacTeX) se existir, senão `tectonic` (`brew install tectonic`; um binário que
descarrega os pacotes à primeira compilação e corre o BibTeX sozinho). As corridas de que as figuras do artigo 1
precisam: `uv run cmfsim batch batches/dissertation.json` (§5.5) e
`uv run python -m generators.dp_comparison && uv run cmfsim batch batches/dp_comparison.json` (Apêndice A),
`uv run cmfsim batch batches/solver_matrix.json` (matriz de solvers), `uv run cmfsim batch batches/performance.json` e
`uv run python -m surrogate.pipeline --steps datasets,reference,grid,models,profiles,profile_models,pinn` (substituto).
Sem corridas, as figuras saem com um aviso no lugar dos dados, as tabelas com «--», e o PDF compila na mesma.

## Artigo novo

```bash
uv run python tools/new_paper.py surrogate-models --title "Surrogate models for ..."
cd papers/02-surrogate-models && make
```

## Convenções

- As figuras do artigo são código: tudo o que aparece no PDF sai de `make_figures.py` + `figlib`, com a paleta
  por entidade de `figlib.COLORS` (a mesma entidade tem a mesma cor em todos os artigos) e as larguras de
  coluna da Elsevier (`SINGLE` 90 mm, `MID` 140 mm, `DOUBLE` 190 mm).
- A bibliografia canónica do tema fica em `_common/references.bib` (chave `apelido+ano`, DOI sempre que exista);
  `refs.bib` só para o que é específico do artigo.
- O guia da revista, resumido em `00-escolha-da-revista.md`, já está embutido no esqueleto (resumo ≤ 250
  palavras, 3–5 *highlights* ≤ 85 caracteres, 1–7 palavras-chave, SI, declarações, CRediT). Na submissão,
  além do `.tex`, vão: cada figura como ficheiro separado (`figures/generated/*.pdf`, renomeadas Figure_1…),
  os *highlights* num ficheiro à parte, as declarações e, se houver, o resumo gráfico.
