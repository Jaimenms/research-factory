# Índice de alterações em relação à dissertação

*Cada entrada é uma alteração atómica em relação ao que a dissertação descreve, com ficheiro
próprio nesta pasta (tipo, área, estado, onde, antes, depois, benefício medido). Tipos:
**feature** (opção nova; a da dissertação continua disponível), **bugfix** (gralha ou erro do
texto), **estudo** (sensibilidade ou verificação), **hipótese** (decisão tomada por falta de
informação, registada para ser revista).*

| n.º | tipo | título | antes → depois |
|---|---|---|---|
| [001](001-dados-da-dissertacao.md) | hipótese | Dados em falta ou ambíguos na dissertação e as hipóteses adotadas | sem Cp do C7+, k_ij, Nc_eff, constantes de Pedersen → valores assumidos e registados |
| [002](002-flash-polimento-newton.md) | feature | Polimento de Newton no flash e derivadas pelo teorema da função implícita | substituições sucessivas até 1e-12 → + 2 Newton: precisão de máquina e d(beta, x, y)/d(P, T) exatas em JAX |
