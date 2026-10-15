# Ficheiros LaTeX partilhados

`thumbnails/` — ícones (e-mail, URL, ORCID, redes) que a classe Elsevier CAS (`cas-sc`/`cas-dc`) inclui ao lado
dos autores; vêm do pacote `els-cas-templates` do CTAN (LPPL) e não fazem parte do pacote da distribuição TeX,
por isso entram pelo caminho de procura que `paper.mk` passa ao motor (`-Z search-path` no tectonic,
`TEXINPUTS` no latexmk). As classes e o estilo bibliográfico `cas-model2-names.bst` vêm da própria distribuição.
