# Figuras digitalizadas da dissertação (Carvalho, UERJ 2022)

Digitalização das figuras com dados da dissertação *Modelagem e simulação composicional do escoamento
multifásico em tubulações com geração de modelos substitutos* (Gustavo Alves de Carvalho, UERJ/IQ, 2022),
a partir do PDF em `../tese-gustavo-carvalho.pdf`. Todas as figuras do PDF são imagens JPEG raster
(não há vetores), pelo que os dados foram recuperados por calibração de eixos e segmentação por cor.

**Aviso importante.** Para as Figuras 24–28 (comparação do modelo com os modelos da literatura, quedas de
pressão por caso) os números exatos estão no **Apêndice A (Tabelas 36–37)** da dissertação. A digitalização
dessas figuras é apenas complementar (serve para conferir a transcrição e para ter os limites mín/máx das
barras); use as tabelas como fonte.

## Conteúdo

| pasta/ficheiro | papel |
|---|---|
| `digitize.py` | script reprodutível: extrai as imagens do PDF, calibra, segmenta, grava dados e overlays |
| `img/pNNN_iK.jpeg` | imagens extraídas do PDF (página NNN, imagem K), tal como embutidas |
| `specs/fig_NN.json` | especificação de calibração por figura (moldura, ticks valor↔píxel, escala, rótulos, séries, cores, caixas a excluir) |
| `data/fig_NN.json` | dados digitalizados (esquema abaixo) |
| `data/index.json` | estado das 59 figuras |
| `overlay/fig_NN.png` | imagem original com os pontos digitalizados sobrepostos (cores contrastantes), para verificação |

Reprodução (Python 3 com numpy, scipy, matplotlib e pymupdf):

```bash
python -I digitize.py                 # tudo (extrai imagens em falta, digitaliza, escreve índice)
python -I digitize.py --only 38,39    # só algumas figuras
python -I digitize.py --no-extract    # usa as imagens já em img/
```

## Método

1. **Extração.** As imagens JPEG embutidas são extraídas página a página com PyMuPDF e guardadas em `img/`.
   A Figura 27 está cortada em duas imagens no PDF (`p124_i0` tira superior + `p124_i1`), que são empilhadas
   verticalmente antes da calibração (os ticks ficam contínuos, o que confirma o alinhamento).
2. **Calibração.** Para cada eixo, a posição em píxeis das marcas de tick (lida na imagem e conferida com a
   deteção automática) é ajustada por mínimos quadrados a `px = a·u + b`, com `u = valor` (escala linear) ou
   `u = log10(valor)` (escala log). O RMSE do ajuste fica registado (`x_fit.rmse_px`, `y_fit.rmse_px`); em
   todas as figuras é inferior a 0,8 px. Figuras com vários painéis (24–28) têm uma calibração por painel.
3. **Segmentação.** Cada série é isolada por regras de cor aplicadas ao interior da moldura (recuada 4 px),
   excluindo as caixas de legenda/texto indicadas na spec: distância RGB a uma cor de referência com
   tolerância (`rgb`), escuridão (`dark`, para linhas pretas/barras de erro/traços), tonalidade HSV com
   saturação mínima (`hue`, para séries pontilhadas cujos pontos ficam muito pálidos no JPEG) e regras de
   azul/verde/cinza para as hachuras. Nas séries pontilhadas exclui-se ainda uma vizinhança (6–16 px) das
   outras séries do painel, para não confundir o *ringing* de croma do JPEG com pontos, e aplica-se um
   filtro de densidade local.
4. **Conversão em dados**, por tipo de série:
   - `line`: por coluna de píxel, mediana das linhas da máscara (uma curva y(x)); se a tinta numa coluna
     ocupar mais do que `max_span` px (salto vertical), devolvem-se os dois extremos; subamostragem uniforme
     em x para ~300 pontos;
   - `step` (Figura 37): como `line` + patamares inteiros `steps = [[x_início, x_fim, nível], …]`;
   - `scatter`: componentes conexas → centróides (um por marcador); componentes maiores do que ~1,6× a área
     de um marcador são divididas por um filtro casado guloso com um disco do raio do marcador
     (`n_markers` é por isso uma estimativa; `n_area_estimate` = área total / área de um marcador);
   - `bar` (Figura 26): topo por coluna → classes de largura igual; altura e contagem (`count`) por classe;
   - `hist_overlay` (Figuras 46, 50, 54): histogramas sobrepostos — topo da tinta (máx(P, T)), topo da região
     cinza (sobreposição = mín(P, T)) e tonalidade da faixa entre eles para decidir qual é P e qual é T;
     produz três séries `bar` (Pressão, Temperatura, Sobreposição);
   - `errorbar` (Figuras 27–29): por caso, extremos da barra preta (`y_min`, `y_max`) e centróide do marcador
     do modelo (`y_model`, com `marker` = `circle_blue` dentro do intervalo ou `triangle_red` fora);
   - `dashed_chain` (Figuras 22–23): traços pretos → centróides ligados por proximidade e orientação; as
     junções (grau ≥ 3) separam ramos, cada ramo é uma série `… — cadeia k`;
   - `boundary_h`/`boundary_v` (Figura 30): fronteiras entre regiões coloridas (por coluna e por linha).
5. **Overlay.** Cada `overlay/fig_NN.png` mostra a imagem original com a moldura e os ticks calibrados (a
   vermelho), as caixas excluídas (tracejado vermelho) e os pontos digitalizados em cores contrastantes.
   Todos os overlays foram inspecionados visualmente.

## Esquema de `data/fig_NN.json`

```json
{"figure": 38, "caption": "Gasoduto: perfil de pressão", "page": 138, "source_image": "img/p138_i0.jpeg",
 "section": "5.6", "status": "digitized", "confidence": "alta",
 "axes": {"x": {"label": "Comprimento", "unit": "m", "scale": "linear", "ticks": [[0, 178.5], [1000, 365.0]]},
          "y": {"label": "Pressão", "unit": "bar", "scale": "linear", "ticks": [[70.0, 79.5], [69.8, 218.5]]}},
 "calibration": {"frame_px": {"top": 42, "bottom": 852, "left": 103, "right": 1746},
                 "x_fit": {"a": 0.18657, "b": 178.57, "rmse_px": 0.161, "n_ticks": 9},
                 "y_fit": {"a": -699.21, "b": 49024.14, "rmse_px": 0.396, "n_ticks": 6}},
 "series": [{"name": "1", "legend_color_rgb": [0, 0, 255], "type": "line", "axis": "y", "linestyle": "solid",
             "n": 300, "x": [420.4, 500.8], "y": [69.947, 69.937]}],
 "uncertainty": {"x_percent_of_range": 0.13, "y_percent_of_range": 0.3},
 "notes": "..."}
```

- `axes.*.ticks` são pares `[valor, píxel]`; o ajuste é `px = a·u + b` (`u = log10(valor)` em escala log).
- `series[].axis` é `"y"` ou `"y2"` (Figura 36 tem eixo direito `axes.y2` e `calibration.y2_fit`).
- Campos extra por tipo: `scatter` → `area_px`, `n_components`, `n_markers`, `n_split`, `marker_area_px`,
  `n_area_estimate`; `step` → `steps`; `bar` → `bins` (`left`, `right`, `center`, `height`, `count` ou
  `kind`), `segments`, `bin_width`; `errorbar` → `cases` (`case`, `x`, `y_min`, `y_max`, `y_model`, `marker`),
  com `x` = número do caso e `y` = `y_model`.
- Figuras com vários painéis (24–28): `calibration` é `null`, `panels` tem `axes`/`calibration`/`uncertainty`
  por painel e cada série traz `panel` (`"4 in"`, `"6 in"`, `"8 in"`, `"10 in"`; Figura 26: `"horizontal"`,
  `"vertical"`). O `source_image` da Figura 27 é uma lista (duas imagens empilhadas).
- Unidades: os eixos rotulados em múltiplos (1e6 J/kg, 1e-5 Pa.s) foram convertidos para a unidade base.

## Incerteza

`uncertainty` dá, por eixo, `(erro_base + rmse_calibração) / extensão_da_moldura × 100`, com erro base de
1,5 px (dispersão), 2 px (linhas, barras, barras de erro, fronteiras), 2,5 px (histogramas sobrepostos) e
3 px (traços). Valores típicos: 0,1–0,4 % do intervalo do eixo em x e 0,2–0,6 % em y (em escala log, % do
intervalo em log10). A isto acrescem erros sistemáticos não contabilizados: espessura da linha e
anti-aliasing (±1 px), compressão JPEG das imagens (artefactos de croma de até ~8–16 px em volta de linhas
coloridas, só relevantes para séries pontilhadas), sobreposição de marcadores (contagem e centróides
aproximados) e oclusão onde séries coincidem (pontos em falta, não pontos errados).

## Limitações e cuidados

- **Figuras 24–28:** usar o Apêndice A (Tabelas 36–37). Nos gráficos de dispersão (24–25) marcadores de
  modelos diferentes sobrepõem-se no mesmo caso, pelo que alguns ficam ocultos ou com centróide deslocado;
  o eixo x é categórico (número do caso) e `x` deve ser arredondado.
- **Gráficos de paridade (14–21, 47–48, 51–52, 55–56):** marcadores grandes e muito sobrepostos; o número de
  pontos recuperado (60–110) é inferior aos 159 pares analisados; não há reta y = x desenhada nas 14–21
  (nas 47–56 há, e foi extraída como série `Reta y = x`).
- **Séries pontilhadas (TULSA/Tulsa nas 32–34 e 37; séries 2 e 5 nas 38–40):** pontos pálidos no JPEG,
  segmentados por tonalidade com exclusão da vizinhança das outras séries — ficam lacunas onde encostam ou
  cruzam outras séries. Na Figura 37 os patamares de Tulsa coincidentes com outras séries não são
  recuperáveis da imagem (ver `notes`).
- **Saltos verticais (34, 35, 40):** nas colunas do salto a série devolve os dois extremos; a posição em x
  do salto tem a resolução de 1 coluna (≈ 4–5 m).
- **Figura 36:** intervalo do eixo direito muito estreito (2,5e-5 mol/mol); a série preta pontilhada começa
  dentro da caixa de legenda "Vapor" (perdem-se os primeiros ~80 m).
- **Histogramas sobrepostos (46, 50, 54):** 36 classes (18 na 50) entre 0 e o máximo dos dados, com a
  largura estimada a partir da aresta direita da última barra; a atribuição P/T em classes muito baixas
  depende da tonalidade da hachura e pode falhar (`kind` regista o caso). Cada série soma ≈ 50 %.
- **Figuras 22–23:** as curvas de transição tracejadas foram divididas em ramos nas junções; a atribuição
  de um traço junto de uma junção pode ser ambígua.
- **Figuras 41–43:** dispersões 3D em projeção oblíqua — não digitalizadas (`skipped`); as faixas das
  variáveis amostradas estão nas Tabelas 27–29.
- As legendas (`caption`) seguem a lista de figuras da dissertação; as secções indicadas são as do capítulo 5.

## Figuras por estado

**Esquemas sem dados (`schematic`, 14):** 1–12, 31, 44.

**Não digitalizadas (`skipped`, 3):** 41, 42, 43 (gráficos 3D).

**Digitalizadas (`digitized`, 42):**

| confiança | figuras |
|---|---|
| alta (17) | 13, 22, 23, 26, 29, 30, 32, 33, 34, 35, 37, 38, 39, 40, 57, 58, 59 |
| média (25) | 14–21 (paridade), 24, 25 (dispersão por caso), 27, 28 (barras de erro; fonte exata no Apêndice A), 36 (eixo duplo, série preta), 45, 49, 53 (erro de treino, marcadores sobrepostos), 46, 50, 54 (histogramas hachurados), 47, 48, 51, 52, 55, 56 (paridade RNA) |
| baixa (0) | — |

A lista completa com legenda, página, ficheiro de dados, confiança e nota está em `data/index.json`.
