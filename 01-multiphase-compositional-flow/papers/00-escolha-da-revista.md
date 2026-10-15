# Escolha da revista para o artigo da dissertação

*Pesquisa feita em 2026-10-15. Objetivo: a revista indexada com maior semelhança ao tema do artigo 01
(simulador composicional de escoamento gás–líquido em tubulações em regime permanente: Peng–Robinson + flash,
modelo de não escorregamento e modelo mecanicista com escorregamento, Jacobianas exatas por diferenciação
automática, validação com os 200 casos do Apêndice A e os sistemas das §5.5–5.6; a seguir, modelos
substitutos).*

## Decisão

**Geoenergy Science and Engineering** (Elsevier; ISSN 2949-8910; até 2022 *Journal of Petroleum Science and
Engineering*, ISSN 0920-4105). Razões:

1. **É onde estão os artigos mais parecidos.** A pesquisa por «compositional two-phase flow pipeline/wellbore
   steady-state mechanistic model equation of state» devolve, como mais próximos, artigos do JPSE: o simulador
   composicional bifásico de poço totalmente implícito com equação de estado (2021), o estudo numérico de
   escoamento bifásico composicional com transferência de calor em poços verticais (2021) e o modelo
   composicional de poço acoplado ao reservatório (2009). Nenhum combina, como o nosso, regime permanente +
   composicional + fechamentos mecanicistas + equação de energia + Jacobianas exatas, o que é a lacuna a explorar.
2. **O âmbito cobre também o artigo seguinte.** A revista lista «application of artificial intelligence, machine
   learning and data analytics in energy industry» e «sustainable hydrocarbon exploration and production», ou
   seja, o artigo dos modelos substitutos (cap. 4 e §5.7 da dissertação) cabe na mesma revista.
3. **Métricas e processo.** CiteScore 12,1 e fator de impacto 4,4 (página da revista, 2026-10-15; confirmar no
   JCR do ano), Q1 no SJR; revisão *single anonymized*; a página «Insights» indica 17 dias até à primeira
   decisão, 74 dias de revisão, 167 dias da submissão à aceitação e 2 dias da aceitação à publicação em linha.
4. **LaTeX em primeira classe.** O guia de autores encoraja o modelo Elsevier CAS (`cas-sc`/`cas-dc`), pede o
   `.tex` como ficheiro-fonte e aceita coluna dupla só em LaTeX. Citações autor–ano.

## Candidatas comparadas

| revista | editora / acesso | âmbito e semelhança com o tema | modelo e estilo | observações |
|---|---|---|---|---|
| **Geoenergy Science and Engineering** (ex-JPSE) | Elsevier; híbrida | produção de hidrocarbonetos, poços e linhas, IA/ML na indústria da energia; os artigos mais próximos estão aqui | CAS/elsarticle; autor–ano | **escolhida**; CiteScore 12,1, IF 4,4 |
| Gas Science and Engineering (ex-*J. Natural Gas Sci. Eng.*) | Elsevier; híbrida | produção e processamento de gás, «analytical and numerical methods», IA; o âmbito antigo citava explicitamente «multiphase flow and flow assurance» | CAS/elsarticle; autor–ano | alternativa natural se o artigo for centrado no gasoduto com condensado (§5.6) |
| International Journal of Multiphase Flow | Elsevier; híbrida | «mass, momentum and energy exchange among phases»; modelação fundamental de escoamentos multifásicos | CAS/elsarticle (ExTra exige `elsarticle`) | CiteScore 7,0, IF 3,9; mais exigente na física do escoamento e menos na aplicação à produção; revisão com ≥ 2 revisores |
| SPE Journal | Society of Petroleum Engineers | engenharia de petróleo; modelos mecanicistas (Xiao, Gomez) saíram em veículos SPE | modelo Word; LaTeX só como PDF na revisão e fonte na revisão final | sem modelo LaTeX oficial; comunidade muito alinhada com o tema |
| Journal of Pipeline Science and Engineering | KeAi/Elsevier; acesso aberto | dutos de petróleo, gás, H₂, CO₂; «flow assurance» e «numerical modelling» entre os tópicos | guia KeAi; revisão duplamente cega | revista jovem (eISSN 2667-1433); boa para o lado «dutos», fraca para o lado termodinâmico |
| Computers & Chemical Engineering | Elsevier; híbrida | métodos numéricos, AD, substitutos; já publicou PINN para escoamento bifásico em dutos (2026) | CAS/elsarticle; numérico | encaixa se a contribuição for vendida como método numérico e não como aplicação |

Também consideradas e postas de lado: *Fluid Phase Equilibria* (só a termodinâmica), *Petroleum Science* e
*Journal of Petroleum Exploration and Production Technology* (acesso aberto, menor seletividade), *Brazilian
Journal of Chemical Engineering* (regional), *Energies*/*Processes* (MDPI).

## Guia de autores da revista escolhida (o que o esqueleto já cumpre)

Lido em 2026-10-15 a partir da página «Guide for authors» do ScienceDirect (a página bloqueia o acesso
automático direto; o texto foi obtido por um proxy de leitura). Pontos que afetam o manuscrito:

- **Ficheiros**: `.tex` como fonte (PDF não serve); modelo Elsevier CAS recomendado (`els-cas-templates`,
  classes `cas-sc` para uma coluna e `cas-dc` para duas); coluna dupla só em LaTeX.
- **Página de título**: título sem abreviaturas; nomes completos; afiliações com morada completa e país, por
  letra minúscula sobrescrita; autor correspondente com e-mail (publicado).
- **Resumo**: ≤ 250 palavras, autocontido, sem referências (se inevitáveis, autor e ano).
- **Palavras-chave**: 1 a 7, em inglês, de preferência de uma palavra. Até 6 códigos Inspec.
- **Highlights**: obrigatórios; 3 a 5 pontos, ≤ 85 caracteres cada (com espaços); ficheiro à parte com
  «highlights» no nome.
- **Resumo gráfico**: opcional.
- **Estrutura**: secções numeradas 1, 1.1, 1.1.1; o resumo fora da numeração; teoria/cálculo como extensão da
  introdução; agradecimentos em secção própria antes das referências; **CRediT** obrigatório (o autor
  correspondente declara os papéis de todos); apêndices A, B com Eq. (A.1), Tabela A.1, Fig. A.1.
- **Unidades**: SI (equivalente SI se outras aparecerem). Variáveis em itálico, `exp` para potências de e,
  equações apresentadas e numeradas pela ordem de citação.
- **Tabelas**: texto editável, legenda, notas por baixo, sem linhas verticais nem sombreados; sem duplicar dados.
- **Figuras**: ficheiros separados, nomeados Figure_1, Figure_2…; vetoriais em EPS/PDF com fontes embebidas;
  bitmaps a ≥ 300 dpi (fotografias) ou ≥ 1000 dpi (traço); não combinar várias imagens num ficheiro, exceto se
  são para comparar (painéis a, b); cores acessíveis a daltónicos; legenda com título breve e descrição.
- **Referências**: formato livre mas consistente na submissão (a revista aplica o seu estilo na prova); **no
  texto, autor–ano** («Jones, 2019; Allan, 2020», «Kramer et al. (2023)»); lista alfabética e depois
  cronológica; DOI sempre que exista; nomes de revistas abreviados pela LTWA; conjuntos de dados com
  «[dataset]» e software citados como referências.
- **Declarações** na submissão: interesses concorrentes; uso de IA generativa (obrigatória); disponibilidade dos
  dados; financiamento. Revisão *single anonymized*.
- **Acesso aberto**: opcional (híbrida); ver a página de opções da revista para o APC.

## Fontes

- Guia de autores: https://www.sciencedirect.com/journal/geoenergy-science-and-engineering/publish/guide-for-authors
- Âmbito: https://www.sciencedirect.com/journal/geoenergy-science-and-engineering/about/aims-and-scope
- Métricas e prazos: https://www.sciencedirect.com/journal/geoenergy-science-and-engineering/about/insights
- Modelo CAS (CTAN): https://ctan.org/pkg/els-cas-templates ; modelo clássico: https://ctan.org/pkg/elsarticle ;
  instruções Elsevier: https://www.elsevier.com/latex
- Artigos próximos (JPSE): https://www.sciencedirect.com/science/article/pii/S0920410521005842 ,
  https://www.sciencedirect.com/science/article/pii/S0920410521000590 ,
  https://www.sciencedirect.com/science/article/pii/S0920410510000458
- Gas Science and Engineering: https://www.sciencedirect.com/journal/gas-science-and-engineering/about/aims-and-scope
- International Journal of Multiphase Flow: https://www.sciencedirect.com/journal/international-journal-of-multiphase-flow/about/aims-and-scope
- SPE (recursos para autores): https://www.spe.org/en/authors/resources/ ; https://www.spe.org/en/authors/faq/
- Journal of Pipeline Science and Engineering: https://keaipublishing.com/cn/journals/journal-of-pipeline-science-and-engineering
