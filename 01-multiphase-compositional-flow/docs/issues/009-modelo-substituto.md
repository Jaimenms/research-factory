# 009 · Modelo substituto: massa de dados pelo simulador, rede e treino em JAX, grelha e encadeamento

| | |
|---|---|
| tipo | feature / estudo |
| área | capítulo 4 e §5.7 (modelo substituto), Apêndices B e C |
| estado | feito (2026-10-15); achados térmicos do Apêndice B em aberto |
| onde | `surrogate/` (`sampling`, `dataset`, `model`, `train`, `grid`, `predict`, `pipeline`), `generators/surrogate.py`, `cases/surrogate/`, `models/surrogate/`, `dissertation/extract.py` (Apêndices B e C), `dissertation/data/surrogate_tables.json`, `notebooks_src/thesis/nb_5_7_surrogate.py`, `tests/test_surrogate.py` |
| dados | Tabelas 27–33, Figuras 45–58 digitalizadas, Apêndice B (`appendix_b_{well,line,riser}.csv`) e C (`appendix_c_*.csv`) |

## Antes

A dissertação gera 300 amostras por trecho do sistema offshore (§5.5, Tabela 21) com o fluido da
Tabela 6 e o modelo de não escorregamento, variando P_in, T_in e m por hipercubo latino (SMT) nas
faixas das Tabelas 27–29, e registando P_out e T_out; divide 70/30; treina em PyTorch redes
multicamada (todas as camadas escondidas iguais) com Adam e erro quadrático médio; procura em
grelha 135 configurações (camadas 1–3 × sigmóide/tanh/ReLU × 5–25 neurónios × taxa 0,1/0,01/0,001)
e escolhe 1 camada, 20 neurónios, ReLU, 0,01 (Tabela 31); R² de teste 0,98/0,94 (poço), 0,98/0,93
(linha), 0,96/0,91 (riser) em P/T (Tabela 32); encadeia os três modelos e compara com o modelo físico
(Figuras 57–58, Tabela 33). O Apêndice B imprime as amostras (com as saídas do modelo físico do
autor) e o Apêndice C os R² das 135 redes por trecho. Nada disto existia no cmfsim; o simulador está
em revisão por outra sessão e não podia ser tocado.

## Depois

Tudo fora do simulador, por cima da API pública (`Case.from_dict`, `load_json`, `march_many`,
`simulate`, `Results`):

- **Amostragem** (`surrogate.sampling`): hipercubo latino próprio (critérios `random`, `center`,
  `maximin`), semente explícita.
- **Massa de dados** (`surrogate.dataset`): entradas nas faixas das Tabelas 27–29 (ou lidas de um
  CSV, p. ex. o Apêndice B) → `march_many` por blocos (uma compilação por trecho, `vmap` sobre as
  amostras) → NPZ com X (bar, K, kg/s), Y (bar, K), máscara de validade, perfis, caso e proveniência,
  mais CSV. 300 amostras por trecho em 67–104 s, incluindo a compilação; todas válidas.
- **Rede e treino** (`surrogate.model`, `surrogate.train`): MLP em JAX com inicialização uniforme
  ±1/√fan_in (a do PyTorch), entradas e saídas padronizadas pelo treino, Adam próprio em lote
  completo com o ciclo de épocas num `lax.scan` compilado, histórico do EQM de treino e de teste,
  métricas (R², RMSE, erro percentual médio, máximo). `MLP.save/load` em JSON; `predict` em numpy puro.
- **Grelha** (`surrogate.grid`): as 135 configurações na numeração do Apêndice C, CSV com R² P/T.
- **Encadeamento** (`surrogate.predict`): a saída de um trecho é a entrada do seguinte.
- **Pipeline** (`surrogate.pipeline`): massa de dados própria e nas entradas do Apêndice B, corrida de
  referência do sistema completo (170 bar, 80 °C, 20 kg/s como nas Figuras 57–58), grelha e modelos
  finais com a Tabela 31, num só processo.
- **Dados da dissertação**: `dissertation.extract` passa a extrair também os Apêndices B (299 amostras
  por trecho: o autor salta o caso 253 nas três tabelas) e C (135 redes por trecho); Tabelas 27–33
  transcritas em `surrogate_tables.json`.
- **Casos** (`generators/surrogate.py`): três trechos isolados (Tabela 21, fluido da Tabela 6, entrada
  no centro das faixas) e o sistema completo com esse fluido.
- **Notebook 5.7** e **testes** (rotina: amostragem, treino numa função suave, divisão, encadeamento,
  parsers, casos; noturno: geração de 4 amostras pelo simulador).

## Achados

### O modelo físico do cmfsim nas entradas do Apêndice B (sem rede)

| trecho | R² P_out | R² T_out | desvio médio P | desvio médio T | ΔP médio autor / cmfsim | ΔT médio autor / cmfsim |
|---|---|---|---|---|---|---|
| poço | 0,979 | 0,996 | +1,35 bar | −0,07 K | 83,5 / 82,2 bar | 11,4 / 11,5 K |
| linha | 0,957 | 0,314 | −1,82 bar | +10,1 K | 7,9 / 9,7 bar | 34,4 / 24,3 K |
| riser | 0,986 | 0,997 | −1,16 bar | −0,35 K | 9,3 / 10,4 bar | 2,8 / 3,2 K |

O poço (gravidade e expansão) e o riser coincidem, com o cmfsim a perder 12–19 % mais pressão por atrito.
Na linha horizontal de 6000 m o modelo físico do autor
arrefece 1,4× mais do que o cmfsim e perde 19 % menos pressão, com o mesmo U, T_amb e D da Tabela 21.
É um achado do mesmo tipo do da §5.6 (issue 004: fator 3,42 no termo de perda de calor, lá resolvido
como erro de apresentação de U), agora com o fluido da Tabela 6 e sem fator que o explique a partir
das tabelas; fica em aberto para a revisão do modelo em curso e para a pergunta ao autor.

### Grelha, modelos finais e encadeamento

Com os nossos dados (suaves, 300 amostras válidas por trecho, 70/30 com semente 0) e 2000 épocas de
Adam em lote completo, a grelha dá R² > 0,9 em P e T em 133, 131 e 131 das 135 redes (poço, linha,
riser); na dissertação foram 25, 8 e 35 (Apêndice C), com muitas redes divergentes a taxa 0,1 e sem
convergência a 0,001, o que sugere poucas épocas ou mini-lotes muito pequenos no treino do autor (as
Figuras 45, 49 e 53 param em ~200 contagens). As oito redes da Tabela 30 estão todas entre as nossas.

| trecho | R² teste P / T (Tabela 32) | R² teste P / T (cmfsim) | erro % médio P / T | erro abs. máx. P / T |
|---|---|---|---|---|
| poço | 0,983 / 0,936 | 0,9997 / 0,9972 | 0,14 / 0,06 | 0,84 bar / 5,0 K |
| linha | 0,980 / 0,926 | 0,9996 / 0,9979 | 0,36 / 0,14 | 1,0 bar / 3,5 K |
| riser | 0,963 / 0,905 | 0,9999 / 0,9994 | 0,15 / 0,05 | 0,44 bar / 2,0 K |

Modelos finais (Tabela 31: 1 camada, 20 neurónios, ReLU, 0,01; 122 parâmetros cada) em
`models/surrogate/{well,line,riser}.json`. **A escolha segue a Tabela 31 da dissertação (reprodução), não a
melhor rede da nossa grelha**: as dez melhores por trecho (R² ≥ 0,9993 em P e T, p. ex. 20 neurónios
sigmóide 1 camada a taxa 0,1) e as piores (5 neurónios ReLU com 3 camadas, que não aprendem) estão no
notebook 5.7, e as 135 linhas por trecho em `models/surrogate/grid_{well,line,riser}.csv`. Encadeados a 170 bar, 80 °C e 20 kg/s contra a corrida de
referência do cmfsim (Figuras 57–58, Tabela 33):

| ponto | P físico / RNA [bar] | erro P cmfsim / Tabela 33 | T físico / RNA [°C] | erro T cmfsim / Tabela 33 |
|---|---|---|---|---|
| cabeça do poço (1500 m) | 84,58 / 84,46 | 0,14 % / 1,0 % | 71,67 / 71,62 | 0,02 % / 0,3 % |
| base do riser (7500 m) | 68,70 / 68,53 | 0,25 % / 3,8 % | 56,20 / 55,98 | 0,07 % / 0,3 % |
| chegada à unidade (7800 m) | 56,96 / 56,83 | 0,24 % / 4,7 % | 53,62 / 53,37 | 0,08 % / 0,6 % |

O perfil físico da dissertação nessas figuras chega a 55,2 bar e 45,4 °C: a pressão coincide com a
nossa (57,0 bar) e a temperatura repete o arrefecimento 1,4× maior na linha visto no Apêndice B.

## Benefício

Massa de dados de treino reproduzível a partir de um caso JSON e de uma semente (300 corridas em
~1 min, uma compilação); treino de uma rede em segundos e grelha de 135 em minutos; modelos
empíricos treinados como ficheiros JSON autocontidos, com métricas e proveniência; comparação
direta com as amostras e os R² do autor.
