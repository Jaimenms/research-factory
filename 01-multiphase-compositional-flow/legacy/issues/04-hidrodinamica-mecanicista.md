# L04 · Hidrodinâmica mecanicista (modelo com escorregamento)

| | |
|---|---|
| família | fechamentos do modelo mecanicista: atrito, padrões, holdup |
| código | [`app/hydrodynamic/hydrodynamic_slip.py`](../masters_code/app/hydrodynamic/hydrodynamic_slip.py), [`app/hydrodynamic/friction_factor.py`](../masters_code/app/hydrodynamic/friction_factor.py), [`app/simulator/simulator_slip.py`](../masters_code/app/simulator/simulator_slip.py) |
| verificado | corrida do código original (2026-10-16), Apêndice A horizontal, 100 casos × 6 variantes ([`scripts/appendix_a.py`](../scripts/appendix_a.py)) |
| responde a | Apêndice A horizontal a 0,4× da dissertação (issue 005); cmfsim abaixo da faixa do HYSYS no horizontal (volume I) |

## Resumo

O código original reproduz a coluna «Dissertação» do Apêndice A horizontal (mediana 1,000, 92 % dos
casos a menos de 1 %). Essa concordância assenta em dois erros que se compensam em parte:

- **L04.1:** um fator de atrito de Darcy numa tensão de parede escrita para Fanning, o que dá 4× o atrito.
- **L03.2:** uma viscosidade do líquido de 0,013 mPa·s, uma ordem de grandeza típica de um gás.

Corrigidos os dois e a rugosidade, o código original fica a 1,1× do cmfsim (mediana). O défice
horizontal do cmfsim face à dissertação deixa de ser uma discrepância do nosso modelo.

## L04.1 Fator de atrito de Darcy em tensões de parede de Fanning (×4)

- `churchill_friction` devolve o fator de **Darcy**: `f = 8*((8/Re)**12 + 1/(a+b)**1.5)**(1/12)`
  ([friction_factor.py:9](../masters_code/app/hydrodynamic/friction_factor.py#L9)).
- As tensões de parede usam a forma de **Fanning**, τ = f ρ |v| v / 2, em todo o modelo
  mecanicista:
  - estratificado ([hydrodynamic_slip.py:472-475](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L472-L475));
  - golfada e filme ([:697](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L697), [:736](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L736));
  - núcleo anular.
- As bolhas horizontal e vertical usam 2 f ρ v²/D com f de Darcy
  ([:626](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L626), [:896](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L896)),
  também 4×.
- Só o padrão 9 (bolhas forçado) usa a forma certa de Darcy, 0,5 f ρ v²/D
  ([:1058](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L1058)).
- O no-slip ([`hydrodynamic.py`](../masters_code/app/hydrodynamic/hydrodynamic.py)) divide τ por D_h e não por D_h/4.
  O fator 4 do Darcy compensa aí e o no-slip fica consistente.

**Efeito medido:** com f/4 no modelo com escorregamento, o Apêndice A horizontal cai exatamente
para 0,250× da dissertação. No estratificado o ΔP horizontal é só atrito.

**Relação com o cmfsim:** a issue 005 testou «atrito de Darcy como impresso» e rejeitou-o. Dava +81 %
no horizontal e +471 % no caso 5 vertical, porque o cmfsim aplicava o fator 4 também a padrões onde o
original não o aplica (padrão 9). O original tem o fator 4 em todos os padrões menos o 9.

## L04.2 Critérios de transição por sigmoides com limiar deslocado

`sigmoid(x) = 1/(1 + exp(-10 x))`
([:30-32](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L30-L32)). As decisões exigem
`sigmoid(...) >= 1 - 1e-6` ou `<= 1e-6`. Isso desloca cada fronteira de ln(1e6)/10 = 1,38 unidades
do argumento:

| transição | critério publicado | o que o código faz |
|---|---|---|
| estratificado → não estratificado (Kelvin–Helmholtz, [:480](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L480)) | v_G > v_KH | v_G > v_KH + 1,38 m/s |
| anular ↔ intermitente ([:489-497](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L489-L497)) | X de Lockhart–Martinelli com C = 16, limiar 1,6 | anular só com X ≤ 0,22 |
| intermitente ↔ bolhas dispersas ([:628-635](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L628-L635)) | Taitel & Dukler | intermitente se o argumento ≤ −1,38, **ou** se v_SL − 0,09 v_SG − 4,2 ≤ −1,38 (critério extra, não documentado) |
| anular vertical ([:755](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L755)) | número de Kutateladze ≥ 3,1 | ≥ 4,5 |

No Apêndice A horizontal o código original dá 92 casos estratificados e 8 em bolhas forçadas
(padrão 9), sem nenhum intermitente nem anular. O cmfsim dá 67 intermitentes e 33 estratificados. As
«sigmoides de ganho não documentado» que a issue 005 suspeitava existem e são estas.

## L04.3 Estratificado: holdup e dissipação com fechamentos diferentes

- **Holdup** ([:34-94](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L34-L94)): resolve o balanço de Taitel & Dukler com
  f_G = 0,001375 [1 + (2e4 ε/D + 1e6/Re_G)^(1/3)] e
  f_L = 1,6291 (v_SG/v_SL)^0,0926 / Re_L^0,5161. As tensões levam velocidades superficiais.
- **Dissipação**
  ([:448-475](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L448-L475)):
  - usa outro fecho, Churchill com Re = D_h v_S ρ/μ;
  - τ_L = f ρ_L |v_SL| v_L / 2 mistura a velocidade superficial com a real;
  - τ_G = f ρ_G v_SG² / 2;
  - D_G exclui o perímetro da interface.
- **Atrito interfacial:** a forma de Andritsos fica no cálculo do holdup e não entra na dissipação.

## L04.4 Intermitente: perímetro da golfada e outros detalhes

- **Perímetro da golfada:** no horizontal, a parcela de atrito da golfada multiplica por
  `slug_interface_perimeter = π D² (1 − H_f) / d_b`
  ([:710](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L710), [:742](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L742)),
  e não pelo perímetro molhado π D. O vertical usa π D
  ([:1007](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L1007)).
  Com d_b ≈ 7 mm e D = 0,1 m, a parcela fica ~13 (1 − H_f) vezes maior.
- **Perímetro do vapor no filme:** é `(2π − θ) D`, sem o fator 1/2
  ([:717](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L717)).
- **Fricção interfacial:** constante, 0,0142 ([:733](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L733)).
- **`friction_slug`:** é um Colebrook em forma de Fanning com logaritmo natural no lugar de log10
  ([friction_factor.py:14](../masters_code/app/hydrodynamic/friction_factor.py#L14)).
  Usa-se no holdup intermitente e na golfada vertical.
- **Comprimento da golfada:** no horizontal é o de Scott, em pés. No ramo vertical a fórmula está
  sem a conversão de unidades, mas esse ramo é inalcançável, porque com θ ≥ 1° vale sempre 30 D.
- **Filme anular:** a fricção do filme usa o fecho do líquido estratificado, 1,6291 (…)
  ([:572](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L572), [:830](../masters_code/app/hydrodynamic/hydrodynamic_slip.py#L830)).

Nos casos do Apêndice A estes ramos não chegam a ser usados, porque o horizontal nunca é
intermitente (L04.2). Pesam no caso offshore (§5.5) e em corridas com outras condições.

## L04.5 Apêndice A horizontal: ablação no código original

Variantes aplicadas em memória ao código original, sem editar o submódulo, nos 100 casos
horizontais. O cmfsim usa o lote `batches/dp_comparison.json`, com rugosidade 4,5e-5 m e Nc_eff = 20.

| variante | original / dissertação, mediana [q10–q90] | original / cmfsim, mediana [q10–q90] | padrões no original |
|---|---|---|---|
| código tal como está | 1,000 [1,00–1,00]; 92 % a < 1 % | 2,56 [1,61–2,90] | 92 estratificado, 8 bolhas forçado |
| atrito de Fanning (f/4) | 0,250 [0,25–0,25] | 0,64 [0,40–0,72] | idem |
| μ_L de Allan & Teja com Nc_eff = 20 (L03.2) | 1,88 [1,30–3,01] | 4,97 [2,94–7,47] | 100 estratificado |
| rugosidade 4,5e-5 m em vez de 2e-4 m | 0,750 [0,72–0,80] | 1,95 [1,57–2,18] | 97 / 3 |
| Fanning + Nc_eff = 20 | 0,470 [0,32–0,75] | 1,24 [0,74–1,87] | 100 estratificado |
| Fanning + Nc_eff = 20 + 4,5e-5 m | 0,432 [0,29–0,68] | 1,12 [0,67–1,67] | 98 estratificado, 1 anular, 1 intermitente |

O cmfsim está a 0,388× da dissertação (mediana). As propriedades na entrada no código original
diferem das do cmfsim:

| propriedade | código original | cmfsim |
|---|---|---|
| μ_L | 0,0134 mPa·s em todos os casos | 0,69–2,95 mPa·s |
| μ_G | 12–17 µPa·s | — |
| σ | 20–30 mN/m | 15–29 mN/m |
| H_L | 0,17–0,81 | 0,37–0,94 |

O resto da diferença, de 0,67 a 1,67 caso a caso, vem dos padrões (estratificado sempre, L04.2) e
dos fechamentos do estratificado (L04.3).

## Consequências

- O défice horizontal do cmfsim face à dissertação (issue 005) fica explicado. A dissertação
  ficava dentro da faixa das correlações do HYSYS em 63 % dos casos porque o atrito estava 4× acima.
- O volume I compara o cmfsim com o HYSYS e não com a dissertação. Nele o horizontal continua abaixo
  da faixa do HYSYS, e isso é uma propriedade do modelo de Taitel & Dukler tal como publicado, não um
  erro de implementação. O texto pode passar a dizer isto.
- Nada a mudar no cmfsim: o atrito de Fanning e a viscosidade do líquido pelo estado já são os
  corretos. Fica como opção possível uma bandeira «padrões como no código original», com as
  sigmoides e o critério extra, só para reprodução.
