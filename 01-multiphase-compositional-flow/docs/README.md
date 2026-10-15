# Dissertação de Gustavo Carvalho (UERJ, 2022) × mfsim: estudo de ligação e plano de reprodução

*Estudo feito em 2026-10-15 a pedido do dono do projeto, **sem nada implementado**. Foco
acordado: o capítulo 3 da dissertação (abordagem de solução numérica) e tudo o que ele exige
do capítulo 2 (modelos, fechamentos, termodinâmica), validado pelas secções 5.1 a 5.6 e 5.8.
O modelo substituto (capítulo 4, secção 5.7, apêndices B e C) fica registado e adiado.*

Fonte: [`references/tese-gustavo-carvalho.pdf`](../references/tese-gustavo-carvalho.pdf)
(201 f., revisão 4 de 24 abr 2023). Base de comparação: o simulador `mfsim` em
`/Users/jaimenms/code/personal/doutorado` (reimplementação da tese de doutorado de 2010, com
backend numpy/JAX, casos JSON, lotes, goldens e issues atómicas).

## Em uma página

**O que a dissertação é.** Um simulador em Python, em **regime permanente**, do escoamento
bifásico gás–líquido em tubulações em série (poço, linha, riser, gasoduto), com
**termodinâmica composicional** (Peng–Robinson, flash P-T por substituições sucessivas,
viscosidades, tensão superficial e entalpia por correlações) acoplada a **dois modelos de
escoamento**: o de **não escorregamento** (v_L = v_G, holdup do flash, incógnitas P e T) e o
**com escorregamento** (vazões por fase, P e T; holdup por balanços de momento mecanicistas
por padrão, à Xiao–Shoham–Brill / Gomez). A identificação de padrão é Taitel–Dukler (1976) e
Taitel et al. (1980) com chaves sigmoidais tomadas de Souza (2010). Apesar do nome
«dois fluidos», o modelo com escorregamento resolve **uma só equação de momento (da
mistura)**; o escorregamento entra por equilíbrio local algébrico entre as fases.

**Capítulo 3: dois métodos para dois modelos.** O *método integrador* (3.1) reescreve o
modelo de não escorregamento como um PVI em x nas variáveis P e T (derivadas totais via
∂/∂P e ∂/∂T por diferenças de 1.ª ordem; três flashes por avaliação; LSODA do SciPy); só
aceita especificação na entrada. O *método das diferenças finitas* (3.2) discretiza o modelo
com escorregamento em n pontos (`numpy.gradient`, 2.ª ordem) e resolve o sistema algébrico
não linear com o Powell híbrido do SciPy, sem Jacobiana analítica, com holdup e flash
aninhados no resíduo; aceita especificações na entrada ou na saída, mas precisa de
estimativa inicial e tem convergência frágil quando uma fase aparece ou desaparece. Resposta
à pergunta do dono: **sim, o capítulo 3 considera o escorregamento, mas só no método 3.2**;
o método 3.1 é, por construção, o modelo de não escorregamento
([detalhe](01-leitura-da-dissertacao.md#os-dois-modelos-e-os-dois-métodos)).

**Holdup e flash.** O flash dá a repartição de massa entre as fases do escoamento (título
w_G) e, com as densidades, a fração volumétrica de equilíbrio λ_L, que é o holdup do modelo de
não escorregamento; o holdup *in situ* H_L só sai do fechamento mecanicista,
H_L = S λ_L/(1 − λ_L + S λ_L) com S = v_G/v_L. A eq. (25) integra para m_G = m·w_G, logo as
equações de massa do método 3.2 são a repartição do flash disfarçada de EDO
([detalhe](01-leitura-da-dissertacao.md#holdup-de-líquido-e-fração-vaporizada-do-flash)).

**Ligação com o mfsim.** São projetos de natureza diferente (dinâmico a 4 equações,
isotérmico, politrópico, rede em grafo, contra permanente, com energia, composicional,
tubos em série), mas com uma interseção grande e aproveitável: Churchill, geometria do
estratificado e os critérios de Taitel–Dukler são os mesmos; a identificação de padrão da
dissertação descende da tese de 2010 (as correções das issues 020/021 e a mistura por ângulo
da 008 aplicam-se diretamente); o solver de estado estacionário do mfsim (Newton com
Jacobiana esparsa ou exata em JAX, mais pseudo-transiente) é o substituto natural do
Powell da dissertação; e toda a infraestrutura (backend injetado numpy/JAX, casos JSON com
schema e herança, lotes, corridas gravadas, goldens A/B/C, issues, notebooks que só leem)
transfere-se como padrão. O que a dissertação traz de **novo** para o ecossistema é o módulo
termodinâmico composicional e a equação de energia, exatamente as duas peças que
`docs/future/` do mfsim lista como pré-requisitos dos temas composicional, CO₂ e criogénico
([mapeamento](02-mapeamento-com-o-mfsim.md)).

**Reprodução.** Não há código-fonte nem ficheiros de resultados da dissertação (ao contrário
dos `.mf2` do doutorado); o único alvo numérico impresso são os 200 pontos de perda de carga
do Apêndice A (3 algarismos), mais figuras digitalizáveis e tabelas de entrada completas.
Logo o nível «bit a bit» não existe; o alcançável é tolerância (~1 %) no Apêndice A e
reprodução qualitativa/quantitativa das figuras. O plano propõe um projeto irmão do
`doutorado` em sete fases, 0 a 6 (rede de segurança; termodinâmica; padrões e fechamentos; modelo
de não escorregamento + método integrador; modelo com escorregamento + diferenças finitas;
melhorias JAX; consolidação), 11–12 sessões ([plano](04-plano-de-reproducao.md)).

**O que o JAX muda.** As derivadas ∂M/∂P, ∂M/∂T do método 3.1 passam de diferenças de
1.ª ordem com três flashes a derivadas exatas por diferenciação automática com um flash
(derivadas do flash pelo teorema da função implícita); o método 3.2 passa de Powell sem
Jacobiana (custo «exponencial» na dissertação) a Newton com Jacobiana exata e esparsa;
`jit` elimina os 90 % de custo do flash; `vmap` corre os 200 casos do Apêndice A (e, mais
tarde, a amostragem do modelo substituto) numa chamada ([avaliação](03-avaliacao-e-interpretacao.md)).

## Ficheiros desta pasta

| ficheiro | conteúdo |
|---|---|
| [01-leitura-da-dissertacao.md](01-leitura-da-dissertacao.md) | leitura estruturada: ficha, os dois modelos, equações, padrões, fechamentos, termodinâmica, os dois métodos do capítulo 3, os resultados que os validam, o que fica adiado |
| [02-mapeamento-com-o-mfsim.md](02-mapeamento-com-o-mfsim.md) | diferenças de natureza e tabela conceito a conceito (igual / caso particular / diferente / inexistente), com o que se reutiliza, o que é novo e o que o mfsim ganha |
| [03-avaliacao-e-interpretacao.md](03-avaliacao-e-interpretacao.md) | o que a dissertação é de facto, pontos fortes, lacunas e inconsistências detetadas no texto, o que a reprodução moderna muda, nível de reprodução alcançável |
| [04-plano-de-reproducao.md](04-plano-de-reproducao.md) | estrutura de projeto proposta (irmã do `doutorado`), esboço do caso JSON, fases com portas de saída, alvos de validação, melhorias como opções com bandeira |
| [05-questoes-abertas.md](05-questoes-abertas.md) | decisões pedidas ao dono, dados em falta, pontos a confirmar com o autor, riscos |
| [06-indice-de-equacoes.md](06-indice-de-equacoes.md) | as 190 equações da dissertação por bloco, com significado e módulo planeado |

## Decisões tomadas (2026-10-15)

O dono respondeu ao bloco A de [05-questoes-abertas.md](05-questoes-abertas.md): não há código
nem dados do HYSYS além do Apêndice A (reprodução de nível B/C); pacote `cmfsim` no monorepo
`research-factory/`; integrador próprio desde o início; referência termodinâmica
independente só como oráculo; **todas as figuras digitalizadas para JSON em
`references/figures/`**; código do mfsim por cópia; um único tubo com parâmetros que variam
axialmente; estudo da coluna Beggs & Brill; sem objetivo de reaproveitamento no doutorado;
termo cinético e pressão interfacial como opções. Sem acesso ao autor, as ambiguidades do
texto decidem-se por estudo e ficam como issues.

## Próximo passo

Fase 0 do [plano](04-plano-de-reproducao.md) quando o dono mandar avançar; a digitalização
das figuras (A7) já está feita.
