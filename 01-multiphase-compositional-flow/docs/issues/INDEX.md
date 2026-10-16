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
| [003](003-fechamentos-leituras-do-texto.md) | hipótese / bugfix | Fechamentos e critérios de padrão: leituras do texto e opções | 15 gralhas ou omissões nas eqs. 32–99 → forma adotada por omissão e alternativa com bandeira |
| [004](004-modelo-sem-escorregamento-e-marcha.md) | feature / hipótese / estudo | Modelo de não escorregamento e marcha em x: fluxos conservados, integrador próprio e o que as figuras mostram | derivadas por diferenças com 3 flashes e LSODA → J = ∂Φ/∂u por AD, Dormand–Prince próprio (175 ms/corrida); perfis térmicos da §5.6 explicados por um só fator 3,42 no termo de perda de calor (erro de apresentação de U na Tabela 25; casos com U_eff) |
| [005](005-modelo-mecanistico-diferencas-finitas.md) | feature / hipótese / estudo | Modelo mecanicista com escorregamento por diferenças finitas e Newton | Powell sem Jacobiana, convergência frágil nas mudanças de fase → Newton com Jacobiana colorida exata, forma algébrica da massa e fluxo m v/A suaves; consistência com a marcha a 4 Pa |
| [006](006-vmap-e-desempenho.md) | feature / estudo | Parâmetros como argumentos, `march_many` (jit + vmap) e o custo por avaliação | recompilação por caso → uma compilação por estrutura; 175 ms por marcha compilada; vmap idêntico ao individual; próximos ganhos identificados |
| [007](007-arranque-quente-do-flash.md) | feature / bugfix | Arranque quente do flash ao longo de x, Rachford–Rice nos extremos e novos defaults do flash | Wilson em todos os flashes → K do flash anterior, resultado idêntico, ganho modesto (o custo está no polimento, medido); bisseção de Rachford–Rice corrigida nos extremos; flash da entrada dentro do jit |
| [008](008-viscosidade-do-liquido-allan-teja.md) | hipótese / estudo | Viscosidade do líquido: a regra de mistura de Allan & Teja dá 3–4× menos do que a dissertação e o HYSYS | notebook 5.2 com 155 estados fora do envelope (kelvin) → corrigido; μ_L 3–4× abaixo nos 159 estados; μ_L × 3 sobe o Apêndice A horizontal de 0,43× para 0,52×; causa por identificar, nada adotado |
