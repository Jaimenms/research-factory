# O que o código original da dissertação revela

*Observações sobre o código-fonte de Gustavo A. de Carvalho
([`../masters_code`](../masters_code), submódulo, commit `d0b91fb`), agrupadas por família. Cada afirmação
foi lida no código e, quando havia número a confirmar, medida a correr o próprio código original em
memória, sem o alterar ([`../scripts`](../scripts)). Estado em 2026-10-16.*

## As questões em aberto e as respostas

| questão (onde estava em aberto) | resposta | família |
|---|---|---|
| fator 3,42 em U nos perfis térmicos da §5.6 (issue 004) | o script passou o **diâmetro externo como espessura**; o perímetro térmico do código é π (D + 2e) = 3,395 π D. O código original reproduz a Figura 39 a 0,02–0,06 °C | [L01](01-termico.md) |
| linha do Apêndice B a arrefecer 1,4× mais (issue 009) | U referido ao **D_ext** da Tabela 21 (1,83× na linha e no riser); o cmfsim com U × 1,833 reproduz a temperatura do Apêndice B a −0,37 K em média | [L01](01-termico.md) |
| §5.5 só fechar com D_in (issue 004) | a figura foi gerada sem espessura (ou com o termo antigo em π D); com D_in o código original reproduz T (0,41 °C) e o holdup (0,007) | [L01](01-termico.md) |
| holdup no-slip da §5.5 a 0,85× do cmfsim (Figura 34, issue 004) | o ln φ do código **não é o de Peng–Robinson**: a_ij = a_i a_j e sinal trocado no termo de atração. Com ln φ padrão injetado, o flash original dá o mesmo que o cmfsim e a `thermo` (λ_L 0,394) | [L02](02-termodinamica-e-flash.md) |
| banda morta da Figura 40 (issue 004) | flash sem teste de estabilidade que converge para a **solução trivial** até 2,5 K abaixo do orvalho e declara vapor; o código original reproduz o salto em 6949 / 3394 / 4364 m | [L02](02-termodinamica-e-flash.md) |
| Cp do C7+ (issue 001, B1) | a entalpia vem dos **polinómios de gás ideal do HYSYS**; Cp do C20pseud 1,59–2,02 kJ/(kg K) a 300–400 K, a extrapolação do cmfsim fica 3–6 % acima | [L02](02-termodinamica-e-flash.md) |
| «falta de propriedades» | o H2S (e outros) não tem polinómio do HYSYS: o fluido da Tabela 6 dá entalpia NaN; a validação das Figuras 13–21 usou o fluido sem N2, CO2, H2S; os valores do HYSYS dessas figuras estão num ficheiro local do autor | [L02](02-termodinamica-e-flash.md), [L03](03-propriedades-de-transporte.md), [L06](06-dados-casos-e-ficheiros.md) |
| k_ij, Nc_eff (issue 001, B2–B3) | k_ij = 0; Nc de Allan & Teja lido do **nome** do componente: o C7+ (`C20pseud`) conta **zero** | [L02](02-termodinamica-e-flash.md), [L03](03-propriedades-de-transporte.md) |
| «dificuldade de modelar a viscosidade em função dos estados» | duas camadas: constante na equação de estado (1,26 mPa·s / 0,012 mPa·s por um limiar de 150 kg/m³) e Allan & Teja nos modelos hidrodinâmicos com Nc = 0 para o C7+, isto é, **μ_L = 0,0134 mPa·s** (valor de gás) no Apêndice A; houve também redes treinadas no HYSYS | [L03](03-propriedades-de-transporte.md) |
| Apêndice A horizontal a 0,4× da dissertação (issue 005) | atrito de **Darcy em tensões de Fanning (×4)** compensado pela μ_L de gás; com os dois corrigidos e a rugosidade do cmfsim, o código original fica a 1,1× do cmfsim. A rugosidade do autor é 2e-4 m. As «sigmoides» deslocam as transições em 1,38 unidades: horizontal sempre estratificado | [L04](04-hidrodinamica-mecanicista.md), [L06](06-dados-casos-e-ficheiros.md) |
| passos e tolerâncias numéricas (B4) | diferenças atrasadas de 0,1 bar e 0,1 K, LSODA com rtol 1e-4, Wilson em cada flash, 3 nós no Apêndice A, RK45 nas amostras do Apêndice B | [L05](05-numerica-e-solvers.md) |
| comprimento do riser | 250 m em todos os scripts (300 m na Tabela 21) | [L01](01-termico.md), [L06](06-dados-casos-e-ficheiros.md) |

## Famílias

| ficheiro | família | pontos |
|---|---|---|
| [01-termico.md](01-termico.md) | balanço térmico: perímetro da perda de calor, geometria dos casos | L01.1–L01.4 |
| [02-termodinamica-e-flash.md](02-termodinamica-e-flash.md) | Peng–Robinson, ln φ, flash, entalpia | L02.1–L02.5 |
| [03-propriedades-de-transporte.md](03-propriedades-de-transporte.md) | viscosidades, tensão superficial, validação de propriedades | L03.1–L03.5 |
| [04-hidrodinamica-mecanicista.md](04-hidrodinamica-mecanicista.md) | atrito, transições de padrão, fechamentos, ablação do Apêndice A | L04.1–L04.5 |
| [05-numerica-e-solvers.md](05-numerica-e-solvers.md) | integradores, derivadas, malhas, pontos frágeis | L05.1–L05.3 |
| [06-dados-casos-e-ficheiros.md](06-dados-casos-e-ficheiros.md) | entradas por caso, ficheiros de dados e resultados, ficheiros em falta | L06.1–L06.4 |

## O que fica em aberto

- **Pressão no-slip da §5.5:** a pressão na linha (Figura 32, «Modelo no-slip») não se reproduz com o
  código tal como está: o código perde 4,8 bar a mais até à base do riser. É provável uma versão
  anterior do atrito no-slip (L01.4).
- **§5.5 com perímetro externo:** com a espessura da Tabela 21, a integração LSODA do código original
  foi interrompida aos 22 min sem terminar (L01.4, L05.1).
- **Ficheiros do autor:** os ficheiros com os valores do HYSYS das Figuras 13–21 e o estudo de casos de
  `props_hysys.xlsx` precisam do autor (L06.3).

## Decisões para o dono (nada foi mudado no cmfsim)

1. **§5.6:** manter U_eff = 3,42 U, valor ajustado, ou passar a 3,395 U, o valor exato do código
   (L01.2).
2. **Perímetro da perda de calor:** os casos do §5.7 (`generators/surrogate.py`) referem U ao D
   interno; o autor referiu-o ao D_ext. A opção física é escolher a área de referência de U, com uma
   bandeira no caso (L01.3).
3. **Volume I:** o texto pode dizer porque é que o horizontal fica abaixo da faixa do HYSYS (L04,
   «Consequências»).
4. **Dados do autor:** `pt_values.csv` e os `*_results_rna.csv` podem substituir dados digitalizados
   ou arredondados (L06.4).

## Como reproduzir

O código original precisa de numpy 1.x. Os scripts correm-no no próprio submódulo, sem escrever lá
nada, e põem os resultados em `runs/legacy/`.

```bash
uv venv --python 3.10 .venv-legacy
VIRTUAL_ENV=.venv-legacy uv pip install -r legacy/scripts/requirements-legacy.txt
.venv-legacy/bin/python legacy/scripts/flash_checks.py          # L02.1, L02.2 (segundos)
.venv-legacy/bin/python legacy/scripts/condensate_5_6.py        # L01.2, L02.2 (~2 min)
.venv-legacy/bin/python legacy/scripts/offshore_5_5.py          # L01.4, L02.1 (~1 min)
.venv-legacy/bin/python legacy/scripts/appendix_b_line.py       # L01.3 (~2 min)
.venv-legacy/bin/python legacy/scripts/property_grid.py         # L03.4 (~1 min)
.venv-legacy/bin/python legacy/scripts/appendix_a.py --variants base,fanning,visc20,smooth,fanning_visc20,fanning_visc20_smooth   # L04.5 (~10 min, 12 processos)
```
