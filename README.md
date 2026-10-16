# research-factory

Repositório de **pesquisas em andamento** de Jaime Neiva Miranda de Souza. Cada pasta numerada é
uma linha de investigação autónoma, com o seu próprio código, documentação, histórico de
decisões e artigos em preparação. Nada aqui é um produto acabado: os resultados mudam à medida
que as hipóteses são testadas, e as discrepâncias encontradas ficam registadas em vez de
escondidas.

## Linhas de investigação

| pasta | tema | estado |
|---|---|---|
| [01-multiphase-compositional-flow](01-multiphase-compositional-flow/) | `cmfsim`: reimplementação moderna (Python, numpy/JAX) da dissertação de mestrado de Gustavo Alves de Carvalho (UERJ, 2022) sobre escoamento bifásico composicional em tubulações em regime permanente, com validação contra a biblioteca `thermo`, as figuras digitalizadas e o Apêndice A da dissertação; o modelo substituto (redes neuronais) vem a seguir | em curso |

## Como cada linha está organizada

- `README.md` e `CLAUDE.md` da pasta: o que é, como correr, regras de trabalho acordadas.
- `docs/`: o estudo que precede o código e, em `docs/issues/`, uma nota por alteração atómica
  em relação à fonte reproduzida (tipo, antes, depois, benefício medido, hipóteses assumidas).
- código genérico (só lê casos JSON), dados transcritos da fonte, casos, lotes, testes de rotina
  e noturnos, notebooks de reprodução (um por secção da fonte) e `papers/` com os artigos.
- Corridas (`runs/`) e saídas de LaTeX ficam fora do git; regeneram-se com os comandos indicados
  em cada pasta.

## Método

O padrão é «tese → código moderno»: ler a fonte, mapear equações e dados, reimplementar com
física escrita uma só vez sobre um backend numpy/JAX (jit, diferenciação automática, vmap),
validar contra referências independentes e registar cada desvio. O trabalho é feito pelo autor
com apoio de um agente de programação (Claude Code), com as decisões documentadas em
`docs/` e as alterações assinadas nos commits.

## Licença e citação

Cada pasta tem a sua licença (a 01 é MIT). Os documentos de terceiros reproduzidos para estudo
(por exemplo, a dissertação em `01-multiphase-compositional-flow/references/`) pertencem aos seus
autores e estão aqui apenas como referência de trabalho. Para citar resultados, use a issue ou o
artigo correspondente e indique o commit.
