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
uv run pytest
uv run python -m dissertation.extract      # regenera os CSV do Apêndice A a partir do PDF
```

## Estado

| fase | conteúdo | estado |
|---|---|---|
| 0 | rede de segurança: git (raiz do monorepo), `uv`, esqueleto, backend copiado, Apêndice A e tabelas em `dissertation/data`, figuras digitalizadas em `references/figures` | em curso (2026-10-15) |
| 1 | `cmfsim/thermo`: PR, flash, propriedades, derivadas exatas em JAX; validação contra a biblioteca `thermo` | em curso |
| 2 | padrões e fechamentos | — |
| 3 | modelo de não escorregamento + marcha em x (integrador próprio) | — |
| 4 | modelo com escorregamento + diferenças finitas + Newton (Apêndice A) | — |
| 5 | melhorias JAX (AD, `vmap`, marcha com escorregamento, IPE suave) | — |
| 6 | notebooks, goldens, síntese | — |

## Licença

MIT ([LICENSE](LICENSE)).
