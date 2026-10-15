# cmfsim — guia para sessões do Claude

Leia por esta ordem: este ficheiro, [README.md](README.md), [docs/README.md](docs/README.md) (o estudo
que originou o projeto), [docs/04-plano-de-reproducao.md](docs/04-plano-de-reproducao.md) e
[docs/issues/INDEX.md](docs/issues/INDEX.md). O padrão de trabalho é o do `mfsim`
(`/Users/jaimenms/code/personal/doutorado`, `docs/method_thesis_to_modern_code.md`).

## O que é

Reimplementação em Python, com arquitetura do mfsim e backend numpy/JAX, da dissertação de
mestrado *Modelagem e simulação composicional do escoamento multifásico em tubulações com
geração de modelos substitutos* (Gustavo Alves de Carvalho, UERJ/IQ, 2022; orientadores
André L. H. Costa, Giovani C. Nunes e Jaime N. M. de Souza): simulador de regime permanente de
escoamento gás–líquido em tubulações com termodinâmica composicional (Peng–Robinson + flash),
modelo de não escorregamento (PVI em x) e modelo mecanicista com escorregamento (diferenças
finitas + Newton). Foco atual: capítulo 3 e as suas validações; o modelo substituto (cap. 4,
§5.7) vem depois e o desempenho (jit/vmap) é desenhado desde já a pensar nisso.

## Mapa do repositório

| pasta | papel |
|---|---|
| `cmfsim/` | o simulador, genérico: só lê casos JSON (`cmfsim-case/1`). `backend/` (cópia do mfsim), `thermo/`, `closures/`, `models/`, `solvers/`, `io/`. |
| `dissertation/` | ponte com a dissertação, fora do simulador: `extract.py` (Apêndice A → CSV), `data/` (tabelas transcritas, CSV). |
| `references/` | o PDF e as figuras digitalizadas (`figures/`, JSON + sobreposições). |
| `cases/`, `generators/`, `batches/`, `runs/` | casos JSON, geradores paramétricos, lotes, corridas (fora do git). |
| `tests/` | rotina (`uv run pytest`, ~4 min) e noturna (`-m nightly`, lê `runs/`); `reference` = precisa da biblioteca `thermo` (oráculo). |
| `notebooks_src/thesis/`, `notebooks/` | fontes `# %%` e notebooks executados, um por secção do capítulo 5; apoio em `dissertation/notebook_utils.py`; figuras em `notebooks/figs/`. |
| `tools/` | `make_nb.py` (fontes → notebooks). |
| `docs/` | o estudo (README + 01–06), `issues/NNN-*.md` + `INDEX.md`. |

## Comandos

```bash
uv sync
uv run pytest                                  # rotina
uv run pytest -m nightly                       # noturna
uv run black --check cmfsim tests tools generators dissertation
uv run python -m dissertation.extract          # Apêndice A -> CSV
```

## Regras acordadas com o dono (não negociar sem perguntar)

1. **Solver próprio em tudo** (PR, flash, holdup, Newton, integrador em x); `thermo`/CoolProp só como oráculo de teste.
2. **Toda a melhoria é uma opção com bandeira**; o default reproduz a dissertação (gralhas incluídas até o estudo decidir).
3. **Nada é apagado sem aprovação explícita.**
4. **Identificadores e chaves JSON em inglês; comentários, docs e mensagens em português.**
5. **Um só caminho de execução**: tudo o que `simulate` precisa está no JSON; notebooks só leem corridas de lotes.
6. **Física escrita uma só vez** sobre o backend injetado (`bk`), sem ramos em valores, formas estáticas: `jit`, `jacfwd` e `vmap` de graça. Desempenho conta: isto vai gerar dados para treinar redes neuronais.
7. **Cada alteração atómica** ao modelo ou ao solver ganha um ficheiro em `docs/issues/` e uma linha no `INDEX.md`.
8. Cada passo fecha com rotina verde, `black` limpo, docs atualizados e commit com `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (git na raiz de `research-factory/`).
9. Um único tubo com parâmetros que variam axialmente (`pipe.profile`); sem grafo por agora.
10. **Um notebook por secção do capítulo 5** (`notebooks_src/thesis/nb_5_X_*.py`), que reproduz
    todas as figuras e tabelas da secção sem simular (lê `runs/`, `references/figures/data/`,
    `dissertation/data/`); `uv run cmfsim nb fonte.py --execute`. O 5.7 vem com o substituto.

## Estado (2026-10-15)

Fases 0–4 feitas (issues 001–005): termodinâmica validada contra `thermo`, fechamentos e mapas de
padrão, marcha de não escorregamento com integrador próprio, modelo mecanicista por diferenças
finitas com Newton (backend de produção: JAX). Em aberto: comparação com o Apêndice A (lote
`batches/dp_comparison.json`), achado dos perfis térmicos da dissertação (issue 004), `vmap`
sobre casos, modelo substituto (adiado).

## Armadilhas conhecidas

- Shell: o diretório de trabalho alterna entre a raiz do monorepo e o projeto; usar caminhos
  absolutos (`P=/Users/jaimenms/code/personal/research-factory/01-multiphase-compositional-flow`)
  e `uv run --project "$P"`. Encadear `pytest` sem `| tail` antes de um commit (estado de saída).
- Método 3.2 em numpy: diferenças avançadas na Jacobiana; converge em malhas grossas (≤ 25 nós) e
  pode estagnar nas finas no vinco do ponto de orvalho; o backend de produção é o JAX (`jvp` exato).
- `_COMPILED` em `simulation.py` guarda as funções compiladas por estrutura (fluido, opções,
  malha): mudar só D, P_in, m, U, T_amb não recompila; mudar o fluido ou n recompila.
- JAX: `where` antes das funções com domínio restrito (sqrt, log, arccos, potências), para tangentes finitas (issue 037 do mfsim).
- Flash: o resultado das substituições sucessivas entra com `stop_gradient`; as derivadas vêm do polimento de Newton (teorema da função implícita). Com `polish_iter=0` não há derivadas corretas do flash em JAX.
- Constantes de PR: Ω_a = 0,45724 e Ω_b = 0,0778 como impressas na dissertação (a `thermo` usa as exatas: diferenças de ~1e-5).
