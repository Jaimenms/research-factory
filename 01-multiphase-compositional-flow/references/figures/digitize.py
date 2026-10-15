#!/usr/bin/env python
"""Digitalização reprodutível das figuras com dados da dissertação de Gustavo Carvalho (UERJ, 2022).

Fluxo:
  1. Extrai as imagens JPEG embutidas no PDF para `img/` (nomes pNNN_iK.jpeg = página NNN, imagem K).
  2. Para cada especificação `specs/fig_NN.json` (moldura, ticks valor<->píxel, cores das séries, caixas a excluir):
     - calibra cada eixo por ajuste linear px = a*u + b (u = valor, ou log10(valor) em escala log) e regista o RMSE;
     - segmenta cada série por cor (regras RGB com tolerância, ou regras de escuridão/tonalidade) dentro do
       interior da moldura, excluindo as caixas de legenda/texto indicadas;
     - converte píxeis em dados segundo o tipo da série (line, step, scatter, bar, hist_overlay, errorbar,
       dashed_chain, boundary_h, boundary_v);
     - grava `data/fig_NN.json` e um overlay de verificação `overlay/fig_NN.png`.
  3. Escreve `data/index.json` com o estado das 59 figuras.

Uso:
  python -I digitize.py [--pdf CAMINHO] [--only 38,39] [--no-extract] [--no-overlay]

Dependências: numpy, scipy, matplotlib, pymupdf.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import pymupdf
except ImportError:  # versões antigas
    import fitz as pymupdf

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "img")
SPEC_DIR = os.path.join(BASE, "specs")
DATA_DIR = os.path.join(BASE, "data")
OVERLAY_DIR = os.path.join(BASE, "overlay")
DEFAULT_PDF = os.path.normpath(os.path.join(BASE, "..", "tese-gustavo-carvalho.pdf"))

# Figuras sem dados (esquemas/diagramas): número -> (legenda, página, secção)
SCHEMATIC = {
    1: ("Formulação da equação geral de conservação", 40, "2"),
    2: ("Regimes de escoamento horizontais. Nome dos regimes de escoamento de cima para baixo: estratificado, "
        "anular, intermitente, disperso com bolhas", 51, "3"),
    3: ("Regimes de escoamento verticais. Nome dos regimes de escoamento da esquerda para a direita: anular, "
        "intermitente, disperso com bolhas", 52, "3"),
    4: ("Identificação do padrão de escoamento", 57, "3"),
    5: ("Padrão de escoamento estratificado", 59, "3"),
    6: ("Padrão de escoamento anular", 63, "3"),
    7: ("Padrão de escoamento de bolhas", 67, "3"),
    8: ("Padrão de escoamento intermitente", 69, "3"),
    9: ("Estrutura geral simplificada dos dados requeridos pelo programa", 86, "4"),
    10: ("Resolução do problema através do método integrador", 93, "4"),
    11: ("Resolução do problema através do método de diferenças finitas", 101, "4"),
    12: ("Etapas para o treinamento do modelo substituto", 104, "4"),
    31: ("Esquemático de sistema de produção de petróleo", 130, "5.5"),
    44: ("Estrutura do modelo substituto", 144, "5.7"),
}

# Erro base (px) assumido por tipo de extração, somado ao RMSE da calibração
BASE_ERR_PX = dict(line=2.0, step=2.0, scatter=1.5, bar=2.0, hist_overlay=2.5, errorbar=2.0,
                   dashed_chain=3.0, boundary_h=2.0, boundary_v=2.0)
OVERLAY_COLORS = ["#ff00ff", "#00ff00", "#00e5ff", "#ff8c00", "#ffff00", "#ff1493", "#7fff00", "#1e90ff",
                  "#ff4500", "#40e0d0", "#ee82ee", "#adff2f"]


# ----------------------------------------------------------------------------- utilitários de imagem
def load_image(path):
    """Lê um JPEG/PNG como matriz RGB uint8 (H, W, 3)."""
    pix = pymupdf.Pixmap(path)
    if pix.n == 4:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    return a.copy()


def load_images(names):
    """Carrega uma ou mais imagens; várias imagens são empilhadas verticalmente (figura cortada no PDF)."""
    arrays = [load_image(os.path.join(IMG_DIR, n)) for n in names]
    if len(arrays) == 1:
        return arrays[0]
    w = min(a.shape[1] for a in arrays)
    return np.vstack([a[:, :w] for a in arrays])


def extract_images(pdf_path, names):
    """Extrai do PDF as imagens pNNN_iK necessárias (só as que ainda não existem em img/)."""
    missing = [n for n in names if not os.path.exists(os.path.join(IMG_DIR, n))]
    if not missing:
        return 0
    os.makedirs(IMG_DIR, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    done = 0
    for n in missing:
        m = re.match(r"p(\d+)_i(\d+)\.(\w+)$", n)
        page, idx = int(m.group(1)), int(m.group(2))
        images = doc[page - 1].get_images(full=True)
        info = doc.extract_image(images[idx][0])
        with open(os.path.join(IMG_DIR, n), "wb") as f:
            f.write(info["image"])
        done += 1
    return done


# ----------------------------------------------------------------------------- calibração
def fit_axis(ax):
    """Ajuste px = a*u + b (u = valor ou log10(valor)). Devolve a, b e RMSE em px."""
    t = np.array(ax["ticks"], dtype=float)
    v, p = t[:, 0], t[:, 1]
    u = np.log10(v) if ax.get("scale") == "log" else v
    A = np.vstack([u, np.ones_like(u)]).T
    (a, b), *_ = np.linalg.lstsq(A, p, rcond=None)
    rmse = float(np.sqrt(np.mean((a * u + b - p) ** 2)))
    return dict(a=float(a), b=float(b), rmse_px=round(rmse, 3), n_ticks=int(len(v)))


def px_to_val(px, fit, scale):
    u = (np.asarray(px, dtype=float) - fit["b"]) / fit["a"]
    return 10.0 ** u if scale == "log" else u


def val_to_px(v, fit, scale):
    u = np.log10(np.asarray(v, dtype=float)) if scale == "log" else np.asarray(v, dtype=float)
    return fit["a"] * u + fit["b"]


# ----------------------------------------------------------------------------- máscaras
def rule_mask(img, rule):
    R, G, B = (img[..., i].astype(np.int32) for i in range(3))
    kind = rule.get("kind", "rgb")
    if kind == "rgb":
        r, g, b = rule["rgb"]
        tol = rule.get("tol", 60)
        return (R - r) ** 2 + (G - g) ** 2 + (B - b) ** 2 <= tol * tol
    V = np.maximum(np.maximum(R, G), B)
    S = V - np.minimum(np.minimum(R, G), B)
    if kind == "dark":
        return (V <= rule.get("vmax", 90)) & (S <= rule.get("satmax", 70))
    if kind == "gray":
        return (V >= rule.get("vmin", 70)) & (V <= rule.get("vmax", 170)) & (S <= rule.get("satmax", 25))
    if kind == "blueness":
        return (B - (R + G) / 2.0) >= rule["min"]
    if kind == "greenness":
        return (G - (R + B) / 2.0) >= rule["min"]
    if kind == "redness":
        return (R - (G + B) / 2.0) >= rule["min"]
    if kind == "hue":
        # Tonalidade HSV (graus) com saturação mínima: apanha pontos pálidos de séries pontilhadas (JPEG)
        Vf = V.astype(float)
        mn = np.minimum(np.minimum(R, G), B).astype(float)
        sat = np.where(Vf > 0, (Vf - mn) / np.maximum(Vf, 1.0), 0.0)
        d = np.maximum(Vf - mn, 1e-9)
        h = np.where(V == R, (G - B) / d, np.where(V == G, 2.0 + (B - R) / d, 4.0 + (R - G) / d)) * 60.0
        h = np.mod(h, 360.0)
        diff = np.abs(np.mod(h - rule["h"] + 180.0, 360.0) - 180.0)
        return (diff <= rule.get("tol", 20)) & (sat >= rule.get("smin", 0.08)) & (V >= rule.get("vmin", 60))
    raise ValueError(f"regra desconhecida: {kind}")


def rules_mask(img, rules):
    m = np.zeros(img.shape[:2], dtype=bool)
    for r in rules:
        m |= rule_mask(img, r)
    return m


def interior_mask(shape, fr, inset, excludes):
    m = np.zeros(shape[:2], dtype=bool)
    m[fr["top"] + inset:fr["bottom"] - inset + 1, fr["left"] + inset:fr["right"] - inset + 1] = True
    for x0, y0, x1, y1 in excludes or []:
        m[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = False
    return m


def contiguous_groups(idx):
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    return np.split(idx, breaks + 1)


def drop_components(mask, min_area=None, max_area=None):
    """Remove componentes conexas fora de [min_area, max_area]."""
    if min_area is None and max_area is None:
        return mask
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    areas = ndimage.sum(mask, lab, np.arange(1, n + 1))
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False
    if min_area is not None:
        keep[1:] &= areas >= min_area
    if max_area is not None:
        keep[1:] &= areas <= max_area
    return keep[lab]


# ----------------------------------------------------------------------------- extração: linhas
def columns_profile(mask, fr, max_span):
    """Por coluna: (x, y_mediana, None) ou, se a tinta ocupar mais de max_span px, (x, y_topo, y_base)."""
    cols = []
    for x in range(fr["left"], fr["right"] + 1):
        rows = np.flatnonzero(mask[:, x])
        if rows.size == 0:
            continue
        g = max(contiguous_groups(rows), key=len)
        if g[-1] - g[0] > max_span:
            cols.append((x, float(g[0]), float(g[-1])))
        else:
            cols.append((x, float(np.median(g)), None))
    return cols


def subsample(cols, n_target, force_idx=()):
    if len(cols) <= n_target:
        return cols
    idx = set(np.round(np.linspace(0, len(cols) - 1, n_target)).astype(int).tolist()) | set(force_idx)
    return [cols[i] for i in sorted(idx)]


def expand_jumps(cols):
    """Transforma (x, y0, y1) em dois pontos ordenados pela proximidade ao ponto anterior."""
    pts, prev = [], None
    for x, y0, y1 in cols:
        if y1 is None:
            pts.append((float(x), y0))
            prev = y0
        else:
            ends = [y0, y1]
            if prev is not None and abs(y1 - prev) < abs(y0 - prev):
                ends = [y1, y0]
            pts.extend([(float(x), ends[0]), (float(x), ends[1])])
            prev = ends[1]
    return pts


def density_filter(cols, window=30, min_neighbors=2, tol_px=4.0):
    """Mantém só as colunas com pelo menos `min_neighbors` outras colunas a <= window px em x e <= tol_px em y
    (um dos extremos, nas colunas com salto). Remove pontos isolados (ruído/ringing JPEG) em séries pontilhadas."""
    if not cols:
        return cols
    xs = np.array([c[0] for c in cols], dtype=float)
    lo = np.array([c[1] for c in cols], dtype=float)
    hi = np.array([c[1] if c[2] is None else c[2] for c in cols], dtype=float)
    keep = []
    for i, (x, y0, y1) in enumerate(cols):
        near = np.flatnonzero((np.abs(xs - x) <= window) & (np.arange(len(cols)) != i))
        ok = False
        for y in ([y0] if y1 is None else [y0, y1]):
            n = int(np.sum((np.abs(lo[near] - y) <= tol_px) | (np.abs(hi[near] - y) <= tol_px)))
            if n >= min_neighbors:
                ok = True
        if ok:
            keep.append(cols[i])
    return keep


def extract_line(mask, fr, max_span=12, n_target=300, density=None):
    cols = columns_profile(mask, fr, max_span)
    if density:
        cols = density_filter(cols, **density)
    jumps = [i for i, c in enumerate(cols) if c[2] is not None]
    return expand_jumps(subsample(cols, n_target, jumps)), len(cols)


def extract_step(mask, fr, yfit, yscale, max_span=12, n_target=300, tol_level=0.15, min_len=40, max_gap=45, density=None):
    """Curva completa + patamares inteiros [x_ini_px, x_fim_px, nível]. Colunas no mesmo nível separadas por
    intervalos (traços/pontos) até max_gap px pertencem ao mesmo patamar; patamares com menos de min_len px
    (rampas que cruzam um nível inteiro) são descartados."""
    cols = columns_profile(mask, fr, max_span)
    if density:
        cols = density_filter(cols, **density)
    levels = []
    for x, y0, y1 in cols:
        if y1 is not None:          # salto vertical: não é patamar
            levels.append((x, None))
            continue
        yv = float(px_to_val(y0, yfit, yscale))
        lv = round(yv)
        levels.append((x, lv if abs(yv - lv) <= tol_level else None))
    # patamares por nível: colunas do mesmo nível fundem-se se o intervalo entre elas for <= max_gap px,
    # ignorando colunas intercaladas de outros níveis (ruído isolado não parte um patamar)
    runs = []
    for lv in sorted({lv for _, lv in levels if lv is not None}):
        xs_lv = [x for x, l in levels if l == lv]
        cur = [xs_lv[0], xs_lv[0], lv]
        for x in xs_lv[1:]:
            if x - cur[1] <= max_gap:
                cur[1] = x
            else:
                runs.append(cur)
                cur = [x, x, lv]
        runs.append(cur)
    runs.sort()
    steps = [s for s in runs if s[1] - s[0] >= min_len]
    change = [i for i in range(1, len(levels)) if levels[i][1] != levels[i - 1][1]]
    jumps = [i for i, c in enumerate(cols) if c[2] is not None]
    pts = expand_jumps(subsample(cols, n_target, set(change) | set(jumps)))
    return pts, steps, len(cols)


# ----------------------------------------------------------------------------- extração: dispersão
def extract_scatter(mask, min_area=20, max_area=None, split=False, marker_area=None):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    info = dict(n_components=0, n_markers=0, n_split=0, marker_area_px=None)
    if n == 0:
        return [], info
    areas = ndimage.sum(mask, lab, np.arange(1, n + 1))
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_area and (max_area is None or a <= max_area)]
    if not keep:
        return [], info
    kept = np.array([areas[i - 1] for i in keep])
    single = float(marker_area) if marker_area else float(np.percentile(kept, 30))
    info.update(n_components=len(keep), marker_area_px=round(single, 1))
    objs = ndimage.find_objects(lab)
    pts = []
    r = math.sqrt(single / math.pi)
    for i in keep:
        a = float(areas[i - 1])
        if split and a > 1.6 * single:
            sl = objs[i - 1]
            sub = lab[sl] == i
            for cx, cy in split_blob(sub, r):
                pts.append((float(cx + sl[1].start), float(cy + sl[0].start), a))
            info["n_split"] += 1
        else:
            cy, cx = ndimage.center_of_mass(mask, lab, i)
            pts.append((float(cx), float(cy), a))
    info["n_markers"] = len(pts)
    info["n_area_estimate"] = int(round(float(kept.sum()) / single))
    return pts, info


def split_blob(sub, r, thr=0.45):
    """Deteção gulosa de marcadores circulares sobrepostos numa componente: filtro casado com um disco de
    raio r; em cada iteração escolhe-se a posição com maior fração de tinta coberta e remove-se esse disco."""
    from scipy.signal import fftconvolve
    rr = int(math.ceil(r))
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    disc = (xx ** 2 + yy ** 2 <= r * r).astype(float)
    area = disc.sum()
    rem = np.pad(sub.astype(float), rr)
    centers = []
    for _ in range(int(2 * rem.sum() / area) + 2):
        resp = fftconvolve(rem, disc, mode="same") / area
        k = int(np.argmax(resp))
        cy, cx = divmod(k, resp.shape[1])
        if resp[cy, cx] < thr:
            break
        centers.append((cx - rr, cy - rr))
        rem[cy - rr:cy + rr + 1, cx - rr:cx + rr + 1] *= (1.0 - disc)
    if not centers:
        cy, cx = ndimage.center_of_mass(sub)
        centers = [(cx, cy)]
    return centers


# ----------------------------------------------------------------------------- extração: barras
def extract_bars(mask, fr, bin_edges_px=None, n_bins=None, baseline_px=None):
    """Topo por coluna -> segmentos de altura constante; opcionalmente altura por classe (bin)."""
    tops = {}
    for x in range(fr["left"], fr["right"] + 1):
        rows = np.flatnonzero(mask[:, x])
        if rows.size:
            tops[x] = float(rows.min())
    xs = sorted(tops)
    segments = []
    if xs:
        cur = [xs[0], xs[0], [tops[xs[0]]]]
        for x in xs[1:]:
            if x - cur[1] <= 1 and abs(tops[x] - np.median(cur[2])) <= 3:
                cur[1] = x
                cur[2].append(tops[x])
            else:
                segments.append((cur[0], cur[1], float(np.median(cur[2]))))
                cur = [x, x, [tops[x]]]
        segments.append((cur[0], cur[1], float(np.median(cur[2]))))
    if bin_edges_px is None and n_bins and xs:
        bin_edges_px = np.linspace(xs[0], xs[-1] + 1, n_bins + 1)
    bins = []
    if bin_edges_px is not None:
        for e0, e1 in zip(bin_edges_px[:-1], bin_edges_px[1:]):
            c0, c1 = e0 + 0.2 * (e1 - e0), e1 - 0.2 * (e1 - e0)
            vals = [tops[x] for x in xs if c0 <= x <= c1]
            top = float(np.median(vals)) if vals else (baseline_px if baseline_px is not None else None)
            bins.append((float(e0), float(e1), top))
    return segments, bins


def extract_hist_overlay(img, interior, fr, bin_edges_px, hue_rules, baseline_px, v_ink=230, v_gray=175):
    """Histogramas sobrepostos: 'Pressão' (hachura azul sobre fundo branco), 'Temperatura' (hachura verde) e
    'Sobreposição' (barra cinza escura = mín(P, T), por baixo das hachuras). Por classe (colunas centrais):
    topo da tinta (qualquer cor) e topo da região cinza (valor HSV < v_gray em >= 50 % das colunas); a faixa
    entre os dois topos é só hachura e a sua tonalidade (azulada/esverdeada) diz qual das séries é a maior."""
    V = img.max(axis=2).astype(int)
    S = V - img.min(axis=2).astype(int)
    ink = ((V < v_ink) | (S > 25)) & interior
    gray = (V < v_gray) & (S < 40) & interior          # cinza da sobreposição (com ou sem hachura por cima)
    blue = rule_mask(img, hue_rules["blue"]) & interior
    green = rule_mask(img, hue_rules["green"]) & interior
    r0, r1 = fr["top"] + 2, fr["bottom"] - 3
    out = []
    for e0, e1 in zip(bin_edges_px[:-1], bin_edges_px[1:]):
        c0, c1 = int(round(e0 + 0.3 * (e1 - e0))), int(round(e1 - 0.3 * (e1 - e0)))
        if c1 <= c0:
            c1 = c0 + 1
        rows_ink = np.flatnonzero(ink[r0:r1, c0:c1 + 1].mean(axis=1) >= 0.5) + r0
        if rows_ink.size == 0 or rows_ink.min() > baseline_px - 3:
            out.append(dict(left=e0, right=e1, P=None, T=None, O=None, kind="empty"))
            continue
        top_ink = float(rows_ink.min())
        prof = ndimage.uniform_filter1d(gray[r0:r1, c0:c1 + 1].mean(axis=1), size=5, mode="nearest")
        rows_gray = np.flatnonzero(prof >= 0.4) + r0
        top_gray = float(rows_gray.min()) if rows_gray.size else None
        lo = int(top_gray) if top_gray is not None else int(baseline_px)
        region_b = blue[int(top_ink):lo, c0:c1 + 1]
        region_g = green[int(top_ink):lo, c0:c1 + 1]
        fb = float(region_b.mean()) if region_b.size else 0.0
        fg = float(region_g.mean()) if region_g.size else 0.0
        if top_gray is None:
            if fb >= fg and fb > 0.02:
                out.append(dict(left=e0, right=e1, P=top_ink, T=None, O=None, kind="P only"))
            elif fg > 0.02:
                out.append(dict(left=e0, right=e1, P=None, T=top_ink, O=None, kind="T only"))
            else:
                out.append(dict(left=e0, right=e1, P=top_ink, T=top_ink, O=top_ink, kind="P = T"))
        elif top_gray - top_ink <= 3:
            out.append(dict(left=e0, right=e1, P=top_gray, T=top_gray, O=top_gray, kind="P = T"))
        elif fb >= fg:
            out.append(dict(left=e0, right=e1, P=top_ink, T=top_gray, O=top_gray, kind="P > T"))
        else:
            out.append(dict(left=e0, right=e1, P=top_gray, T=top_ink, O=top_gray, kind="T > P"))
    return out


# ----------------------------------------------------------------------------- extração: barras de erro
def extract_errorbars(img, dark_mask, interior, fr, xticks_px, half_window, marker_rules):
    out = []
    for case, xpx in xticks_px:
        c0, c1 = int(round(xpx - 3)), int(round(xpx + 3))
        sub = (dark_mask & interior)[fr["top"] + 2:fr["bottom"] - 2, c0:c1 + 1]
        rows = np.flatnonzero(sub.any(axis=1)) + fr["top"] + 2
        rec = dict(case=case, x_px=float(xpx), ymin_px=None, ymax_px=None, marker=None, ymodel_px=None)
        if rows.size:
            rec["ymin_px"], rec["ymax_px"] = float(rows.min()), float(rows.max())
        w0, w1 = int(round(xpx - half_window)), int(round(xpx + half_window))
        for name, rules in marker_rules.items():
            m = rules_mask(img, rules) & interior
            m[:, :w0] = False
            m[:, w1 + 1:] = False
            m = drop_components(m, min_area=25)
            if m.any():
                cy, cx = ndimage.center_of_mass(m)
                rec["marker"], rec["ymodel_px"], rec["xmodel_px"] = name, float(cy), float(cx)
                break
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- extração: traços tracejados
def extract_dashed_chains(mask, min_area=12, max_gap=40, angle_tol=40):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return []
    areas = ndimage.sum(mask, lab, np.arange(1, n + 1))
    objs = ndimage.find_objects(lab)
    dashes = []
    for i in range(1, n + 1):
        if areas[i - 1] < min_area:
            continue
        sl = objs[i - 1]
        ys, xs = np.nonzero(lab[sl] == i)
        xs = xs + sl[1].start
        ys = ys + sl[0].start
        cx, cy = xs.mean(), ys.mean()
        cov = np.cov(np.vstack([xs - cx, ys - cy]))
        w, v = np.linalg.eigh(cov)
        d = v[:, np.argmax(w)]
        proj = (xs - cx) * d[0] + (ys - cy) * d[1]
        length = float(proj.max() - proj.min())
        theta = math.degrees(math.atan2(d[1], d[0])) % 180.0
        dashes.append(dict(cx=cx, cy=cy, theta=theta, length=length, area=float(areas[i - 1])))

    def angdiff(a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    m = len(dashes)
    # candidatos compatíveis (distância <= max_gap e direção coerente com a orientação dos traços)
    cand = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            di, dj = dashes[i], dashes[j]
            dx, dy = dj["cx"] - di["cx"], dj["cy"] - di["cy"]
            dist = math.hypot(dx, dy)
            if dist > max_gap:
                continue
            phi = math.degrees(math.atan2(dy, dx)) % 180.0
            if di["length"] >= 8 and angdiff(phi, di["theta"]) > angle_tol:
                continue
            if dj["length"] >= 8 and angdiff(phi, dj["theta"]) > angle_tol:
                continue
            cand[i].append((dist, j))
            cand[j].append((dist, i))
    # grafo: cada traço liga-se aos seus 2 vizinhos compatíveis mais próximos (união)
    edges = set()
    for i in range(m):
        for d, j in sorted(cand[i])[:2]:
            edges.add((min(i, j), max(i, j)))
    adj = [set() for _ in range(m)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    # junções (grau >= 3) separam ramos; cada ramo = componente conexa dos nós restantes
    junction = {i for i in range(m) if len(adj[i]) >= 3}
    seen = set()
    chains = []
    for s0 in range(m):
        if s0 in seen or s0 in junction:
            continue
        comp, stack = [], [s0]
        seen.add(s0)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen and v not in junction:
                    seen.add(v)
                    stack.append(v)
        if len(comp) < 3:
            continue
        compset = set(comp)
        deg = {u: len([v for v in adj[u] if v in compset]) for u in comp}
        start = min(comp, key=lambda k: (deg[k], dashes[k]["cx"]))
        order, vis, cur = [start], {start}, start
        while True:
            nxt = [v for v in adj[cur] if v in compset and v not in vis]
            if not nxt:
                rest = [k for k in comp if k not in vis]
                if not rest:
                    break
                nxt = [min(rest, key=lambda k: math.hypot(dashes[k]["cx"] - dashes[cur]["cx"], dashes[k]["cy"] - dashes[cur]["cy"]))]
            cur = min(nxt, key=lambda k: math.hypot(dashes[k]["cx"] - dashes[cur]["cx"], dashes[k]["cy"] - dashes[cur]["cy"]))
            order.append(cur)
            vis.add(cur)
        # anexar as junções adjacentes aos extremos, para que o ramo chegue ao ponto de encontro
        for end, pos in ((order[0], 0), (order[-1], len(order))):
            js = [v for v in adj[end] if v in junction]
            if js:
                order.insert(pos, js[0])
        chains.append([(dashes[k]["cx"], dashes[k]["cy"]) for k in order])
    chains.sort(key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
    return chains


# ----------------------------------------------------------------------------- extração: fronteiras de regiões
def extract_boundary_h(above, below, fr, n_target=300):
    pts = []
    for x in range(fr["left"], fr["right"] + 1):
        a = np.flatnonzero(above[:, x])
        b = np.flatnonzero(below[:, x])
        if a.size and b.size and 0 < b.min() - a.max() <= 8:
            pts.append((float(x), (a.max() + b.min()) / 2.0, None))
    return expand_jumps(subsample(pts, n_target))


def extract_boundary_v(left, right, fr, n_target=300):
    pts = []
    for y in range(fr["top"], fr["bottom"] + 1):
        l_ = np.flatnonzero(left[y, :])
        r_ = np.flatnonzero(right[y, :])
        if l_.size and r_.size:
            lm = l_[l_ < r_.min()]
            if lm.size and 0 < r_.min() - lm.max() <= 8:
                pts.append(((lm.max() + r_.min()) / 2.0, float(y)))
    if len(pts) > n_target:
        idx = np.round(np.linspace(0, len(pts) - 1, n_target)).astype(int)
        pts = [pts[i] for i in idx]
    return pts


# ----------------------------------------------------------------------------- processamento de um painel
def axis_out(ax):
    return dict(label=ax.get("label"), unit=ax.get("unit"), scale=ax.get("scale", "linear"), ticks=ax["ticks"])


def process_panel(img, pnl, fig_no, overlay_items):
    fr = pnl["frame_px"]
    xs, ys = pnl["x"].get("scale", "linear"), pnl["y"].get("scale", "linear")
    xfit, yfit = fit_axis(pnl["x"]), fit_axis(pnl["y"])
    cal = dict(frame_px=fr, x_fit=xfit, y_fit=yfit)
    y2fit = y2s = None
    if "y2" in pnl:
        y2fit, y2s = fit_axis(pnl["y2"]), pnl["y2"].get("scale", "linear")
        cal["y2_fit"] = y2fit
    interior = interior_mask(img.shape, fr, pnl.get("inset", 4), pnl.get("exclude_px"))
    rx, ry = fr["right"] - fr["left"], fr["bottom"] - fr["top"]
    series_out = []
    err_px = dict(x=0.0, y=0.0)

    def yconv(px, axis_name):
        if axis_name == "y2":
            return px_to_val(px, y2fit, y2s)
        return px_to_val(px, yfit, ys)

    for s in pnl["series"]:
        typ = s["type"]
        axis_name = s.get("axis", "y")
        base = dict(name=s["name"], legend_color_rgb=s.get("legend_color_rgb"), type=typ, axis=axis_name)
        if s.get("marker"):
            base["marker"] = s["marker"]
        if s.get("linestyle"):
            base["linestyle"] = s["linestyle"]
        e = BASE_ERR_PX.get(typ, 2.0)
        err_px["x"] = max(err_px["x"], e)
        err_px["y"] = max(err_px["y"], e)

        if typ in ("line", "step"):
            mask = rules_mask(img, s["rules"]) & interior
            if s.get("exclude_others"):
                # retira a vizinhança (dilatada) das outras séries do painel: evita que o 'ringing' JPEG
                # (cor complementar) em torno de uma linha vizinha seja lido como pontos da série pontilhada
                others = np.zeros_like(mask)
                for o in pnl["series"]:
                    if o is not s and o.get("rules"):
                        others |= rules_mask(img, o["rules"])
                mask &= ~ndimage.binary_dilation(others, iterations=int(s.get("exclude_radius", 6)))
            if s.get("search_px"):
                x0, y0, x1, y1 = s["search_px"]
                box = np.zeros_like(mask)
                box[y0:y1 + 1, x0:x1 + 1] = True
                mask &= box
            mask = drop_components(mask, s.get("min_area"), s.get("max_area"))
            dens = s.get("density_filter")
            if typ == "line":
                pts, ncols = extract_line(mask, fr, s.get("max_span", 12), s.get("n_target", 300), dens)
                extra = {}
            else:
                pts, steps_px, ncols = extract_step(mask, fr, yfit, ys, s.get("max_span", 12), s.get("n_target", 300), density=dens)
                extra = dict(steps=[[round(float(px_to_val(a, xfit, xs)), 3), round(float(px_to_val(b, xfit, xs)), 3), int(lv)]
                                    for a, b, lv in steps_px])
            xv = px_to_val([p[0] for p in pts], xfit, xs)
            yv = yconv([p[1] for p in pts], axis_name)
            out = dict(base, n=len(pts), n_columns_with_ink=ncols, x=[float(v) for v in xv], y=[float(v) for v in yv], **extra)
            overlay_items.append((s["name"], [p[0] for p in pts], [p[1] for p in pts], "line"))

        elif typ == "scatter":
            mask = rules_mask(img, s["rules"]) & interior
            if s.get("search_px"):
                x0, y0, x1, y1 = s["search_px"]
                box = np.zeros_like(mask)
                box[y0:y1 + 1, x0:x1 + 1] = True
                mask &= box
            pts, info = extract_scatter(mask, s.get("min_area", 20), s.get("max_area"), s.get("split_overlaps", False), s.get("marker_area"))
            pts.sort(key=lambda p: (p[0], p[1]))
            xv = px_to_val([p[0] for p in pts], xfit, xs)
            yv = yconv([p[1] for p in pts], axis_name)
            out = dict(base, n=len(pts), x=[float(v) for v in xv], y=[float(v) for v in yv], area_px=[round(p[2], 1) for p in pts], **info)
            overlay_items.append((s["name"], [p[0] for p in pts], [p[1] for p in pts], "scatter"))

        elif typ == "bar":
            mask = rules_mask(img, s["rules"]) & interior
            b = s.get("bins") or {}
            edges = None
            if "start" in b:
                edges = val_to_px(np.arange(b["start"], b["start"] + b["width"] * (b["n"] + 0.5), b["width"]), xfit, xs)
            baseline_px = float(val_to_px(0.0, yfit, ys)) if ys == "linear" else fr["bottom"]
            segments, bins = extract_bars(mask, fr, edges, b.get("n"), baseline_px)
            seg_out = [dict(x_left=float(px_to_val(a, xfit, xs)), x_right=float(px_to_val(c + 1, xfit, xs)),
                            height=float(px_to_val(t, yfit, ys))) for a, c, t in segments]
            bin_out = []
            for e0, e1, top in bins:
                h = float(px_to_val(top, yfit, ys)) if top is not None else 0.0
                bin_out.append(dict(left=float(px_to_val(e0, xfit, xs)), right=float(px_to_val(e1, xfit, xs)),
                                    center=float(px_to_val((e0 + e1) / 2, xfit, xs)), height=h, count=int(round(h))))
            out = dict(base, n=len(bin_out), x=[d["center"] for d in bin_out], y=[d["height"] for d in bin_out],
                       bins=bin_out, segments=seg_out)
            overlay_items.append((s["name"], [(e0 + e1) / 2 for e0, e1, t in bins if t is not None],
                                  [t for e0, e1, t in bins if t is not None], "bar"))
            for a, c, t in segments:
                overlay_items.append((None, list(range(a, c + 1)), [t] * (c - a + 1), "segment"))

        elif typ == "hist_overlay":
            b = s["bins"]
            start_px = float(val_to_px(b["start"], xfit, xs))
            if b.get("end") == "auto":
                # classes de largura igual entre `start` e a aresta direita da última barra (coluna de tinta
                # mais à direita dentro da moldura): histogramas com n classes sobre [0, máx(dados)]
                Vv = img.max(axis=2).astype(int)
                Sv = Vv - img.min(axis=2).astype(int)
                ink_cols = np.flatnonzero((((Vv < s.get("v_ink", 230)) | (Sv > 25)) & interior).any(axis=0))
                end_px = float(ink_cols.max() + 1)
            else:
                end_px = float(val_to_px(b["start"] + b["width"] * b["n"], xfit, xs))
            edges = np.linspace(start_px, end_px, b["n"] + 1)
            bin_width = float(px_to_val(edges[1], xfit, xs) - px_to_val(edges[0], xfit, xs))
            baseline_px = float(val_to_px(0.0, yfit, ys))
            recs = extract_hist_overlay(img, interior, fr, edges, s["hue_rules"], baseline_px,
                                        s.get("v_ink", 230), s.get("v_gray", 175))
            names = s.get("out_names", dict(blue="Pressão", green="Temperatura", gray="Sobreposição"))
            colors = dict(blue=[0, 0, 255], green=[0, 200, 120], gray=[128, 128, 128])
            for key, field in (("blue", "P"), ("green", "T"), ("gray", "O")):
                bins_out, xs_px, ys_px = [], [], []
                for r in recs:
                    top = r[field]
                    h = float(px_to_val(top, yfit, ys)) if top is not None else 0.0
                    if r["kind"] == "empty":
                        h = 0.0
                    bins_out.append(dict(left=float(px_to_val(r["left"], xfit, xs)), right=float(px_to_val(r["right"], xfit, xs)),
                                         center=float(px_to_val((r["left"] + r["right"]) / 2, xfit, xs)), height=h, kind=r["kind"]))
                    if top is not None:
                        xs_px.append((r["left"] + r["right"]) / 2)
                        ys_px.append(top)
                series_out.append(dict(name=names[key], legend_color_rgb=colors[key], type="bar", axis="y", n=len(bins_out),
                                       x=[d["center"] for d in bins_out], y=[d["height"] for d in bins_out],
                                       bin_width=round(bin_width, 4), bins=bins_out))
                overlay_items.append((names[key], xs_px, ys_px, "bar"))
            continue

        elif typ == "errorbar":
            dark = rules_mask(img, s["rules"])
            xt = [(int(round(v)), p) for v, p in pnl["x"]["ticks"]]
            recs = extract_errorbars(img, dark, interior, fr, xt, s.get("half_window", 10), s["marker_rules"])
            cases = []
            for r in recs:
                cases.append(dict(case=r["case"], x=float(px_to_val(r["x_px"], xfit, xs)),
                                  y_min=float(px_to_val(r["ymax_px"], yfit, ys)) if r["ymax_px"] is not None else None,
                                  y_max=float(px_to_val(r["ymin_px"], yfit, ys)) if r["ymin_px"] is not None else None,
                                  y_model=float(px_to_val(r["ymodel_px"], yfit, ys)) if r["ymodel_px"] is not None else None,
                                  marker=r["marker"]))
            out = dict(base, n=len(cases), x=[c["case"] for c in cases], y=[c["y_model"] for c in cases], cases=cases)
            overlay_items.append((s["name"] + " (mín/máx)", [r["x_px"] for r in recs for _ in range(2) if r["ymin_px"] is not None],
                                  [v for r in recs if r["ymin_px"] is not None for v in (r["ymin_px"], r["ymax_px"])], "scatter"))
            overlay_items.append(("marcador do modelo", [r["xmodel_px"] for r in recs if r["ymodel_px"] is not None],
                                  [r["ymodel_px"] for r in recs if r["ymodel_px"] is not None], "scatter"))

        elif typ == "dashed_chain":
            mask = rules_mask(img, s["rules"]) & interior
            chains = extract_dashed_chains(mask, s.get("min_area", 12), s.get("max_gap", 40), s.get("angle_tol", 40))
            for k, ch in enumerate(chains, 1):
                xv = px_to_val([p[0] for p in ch], xfit, xs)
                yv = yconv([p[1] for p in ch], axis_name)
                series_out.append(dict(base, name=f"{s['name']} — cadeia {k}", n=len(ch), x=[float(v) for v in xv], y=[float(v) for v in yv]))
                overlay_items.append((f"cadeia {k}", [p[0] for p in ch], [p[1] for p in ch], "chain"))
            continue

        elif typ == "boundary_h":
            above = rules_mask(img, s["rules_above"]) & interior
            below = rules_mask(img, s["rules_below"]) & interior
            pts = extract_boundary_h(above, below, fr, s.get("n_target", 300))
            xv = px_to_val([p[0] for p in pts], xfit, xs)
            yv = yconv([p[1] for p in pts], axis_name)
            out = dict(base, n=len(pts), x=[float(v) for v in xv], y=[float(v) for v in yv])
            overlay_items.append((s["name"], [p[0] for p in pts], [p[1] for p in pts], "line"))

        elif typ == "boundary_v":
            left = rules_mask(img, s["rules_left"]) & interior
            right = rules_mask(img, s["rules_right"]) & interior
            pts = extract_boundary_v(left, right, fr, s.get("n_target", 300))
            xv = px_to_val([p[0] for p in pts], xfit, xs)
            yv = yconv([p[1] for p in pts], axis_name)
            out = dict(base, n=len(pts), x=[float(v) for v in xv], y=[float(v) for v in yv])
            overlay_items.append((s["name"], [p[0] for p in pts], [p[1] for p in pts], "line"))
        else:
            raise ValueError(f"tipo de série desconhecido: {typ}")
        series_out.append(out)

    unc = dict(x_percent_of_range=round(100.0 * (err_px["x"] + xfit["rmse_px"]) / rx, 2),
               y_percent_of_range=round(100.0 * (err_px["y"] + yfit["rmse_px"]) / ry, 2))
    if y2fit:
        unc["y2_percent_of_range"] = round(100.0 * (err_px["y"] + y2fit["rmse_px"]) / ry, 2)
    axes = dict(x=axis_out(pnl["x"]), y=axis_out(pnl["y"]))
    if "y2" in pnl:
        axes["y2"] = axis_out(pnl["y2"])
    return dict(axes=axes, calibration=cal, uncertainty=unc), series_out


# ----------------------------------------------------------------------------- overlay
def draw_overlay(img, items, specd, path):
    H, W = img.shape[:2]
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    for pnl in specd.get("panels", []):
        fr = pnl["frame_px"]
        ax.plot([fr["left"], fr["right"], fr["right"], fr["left"], fr["left"]],
                [fr["top"], fr["top"], fr["bottom"], fr["bottom"], fr["top"]], color="#ff0000", lw=0.6, alpha=0.6)
        for v, p in pnl["x"]["ticks"]:
            ax.plot([p, p], [fr["bottom"] - 6, fr["bottom"] + 6], color="#ff0000", lw=0.8)
        for v, p in pnl["y"]["ticks"]:
            ax.plot([fr["left"] - 6, fr["left"] + 6], [p, p], color="#ff0000", lw=0.8)
        for v, p in pnl.get("y2", {}).get("ticks", []):
            ax.plot([fr["right"] - 6, fr["right"] + 6], [p, p], color="#ff0000", lw=0.8)
        for x0, y0, x1, y1 in pnl.get("exclude_px", []) or []:
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#ff0000", lw=0.6, ls="--", alpha=0.7)
    k = 0
    handles = []
    for name, xs, ys, kind in items:
        if kind == "segment":
            ax.plot(xs, ys, color="#ff0000", lw=1.0, alpha=0.8)
            continue
        c = OVERLAY_COLORS[k % len(OVERLAY_COLORS)]
        k += 1
        if kind == "line":
            h = ax.plot(xs, ys, ".", color=c, ms=2.5, mew=0, alpha=0.9, label=name)[0]
        elif kind == "chain":
            h = ax.plot(xs, ys, "-o", color=c, ms=3, lw=1.0, mew=0, alpha=0.9, label=name)[0]
        elif kind == "bar":
            h = ax.plot(xs, ys, "_", color=c, ms=9, mew=2, alpha=0.95, label=name)[0]
        else:
            h = ax.plot(xs, ys, "+", color=c, ms=7, mew=1.2, alpha=0.95, label=name)[0]
        handles.append(h)
    if handles:
        ax.legend(handles=handles, loc="upper center", fontsize=7, ncol=min(4, len(handles)), framealpha=0.8)
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------------- figura
def process_figure(specd, make_overlay=True):
    fig_no = specd["figure"]
    img = load_images(specd["source_images"])
    out = dict(figure=fig_no, caption=specd["caption"], page=specd["page"],
               source_image=("img/" + specd["source_images"][0]) if len(specd["source_images"]) == 1
               else ["img/" + n for n in specd["source_images"]],
               section=specd["section"], status=specd.get("status", "digitized"), confidence=specd.get("confidence"))
    items = []
    panels = specd.get("panels", [])
    all_series, panel_info = [], {}
    for pnl in panels:
        info, series = process_panel(img, pnl, fig_no, items)
        if len(panels) > 1:
            for s in series:
                s = dict(s)
                s_panel = {"panel": pnl["name"]}
                s_panel.update(s)
                all_series.append(s_panel)
        else:
            all_series.extend(series)
        panel_info[pnl["name"]] = info
    if len(panels) == 1:
        info = panel_info[panels[0]["name"]]
        out["axes"], out["calibration"], out["uncertainty"] = info["axes"], info["calibration"], info["uncertainty"]
    else:
        first = panel_info[panels[0]["name"]]
        out["axes"] = {k: dict(label=v["label"], unit=v["unit"], scale=v["scale"]) for k, v in first["axes"].items()}
        out["calibration"] = None
        out["panels"] = panel_info
        out["uncertainty"] = dict(
            x_percent_of_range=max(p["uncertainty"]["x_percent_of_range"] for p in panel_info.values()),
            y_percent_of_range=max(p["uncertainty"]["y_percent_of_range"] for p in panel_info.values()))
    out["series"] = all_series
    out["notes"] = specd.get("notes", "")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, f"fig_{fig_no:02d}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=None, separators=(",", ":"))
    if make_overlay:
        os.makedirs(OVERLAY_DIR, exist_ok=True)
        draw_overlay(img, items, specd, os.path.join(OVERLAY_DIR, f"fig_{fig_no:02d}.png"))
    return out


def summary_line(out):
    parts = []
    for s in out["series"]:
        parts.append(f"{s['name'][:28]}:{s.get('n', 0)}")
    cal = out.get("calibration")
    if cal:
        rm = f"rmse x={cal['x_fit']['rmse_px']} y={cal['y_fit']['rmse_px']}"
    else:
        rm = "; ".join(f"{k}: x={v['calibration']['x_fit']['rmse_px']} y={v['calibration']['y_fit']['rmse_px']}" for k, v in out["panels"].items())
    return f"Fig {out['figure']:02d} [{rm}] " + ", ".join(parts)


# ----------------------------------------------------------------------------- índice
def write_index(specs):
    entries = {}
    for n, (cap, page, sec) in SCHEMATIC.items():
        entries[n] = dict(figure=n, caption=cap, page=page, section=sec, status="schematic", data_file=None,
                          confidence=None, note="Esquema/diagrama sem dados numéricos.")
    for sp in specs:
        n = sp["figure"]
        st = sp.get("status", "digitized")
        entries[n] = dict(figure=n, caption=sp["caption"], page=sp["page"], section=sp["section"], status=st,
                          data_file=f"data/fig_{n:02d}.json" if st == "digitized" else None,
                          confidence=sp.get("confidence") if st == "digitized" else None,
                          note=sp.get("notes", ""))
    idx = [entries[k] for k in sorted(entries)]
    with open(os.path.join(DATA_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=1)
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--only", default=None, help="lista de figuras, ex.: 38,39,40")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    specs = []
    for fn in sorted(os.listdir(SPEC_DIR)):
        if re.match(r"fig_\d+\.json$", fn):
            with open(os.path.join(SPEC_DIR, fn), encoding="utf-8") as f:
                specs.append(json.load(f))
    only = set(int(v) for v in args.only.split(",")) if args.only else None

    names = sorted({n for sp in specs for n in sp.get("source_images", [])})
    if not args.no_extract:
        k = extract_images(args.pdf, names)
        print(f"imagens extraídas do PDF: {k} (total necessárias: {len(names)})")

    for sp in specs:
        if only and sp["figure"] not in only:
            continue
        if sp.get("status", "digitized") != "digitized":
            print(f"Fig {sp['figure']:02d} ignorada ({sp.get('status')})")
            continue
        out = process_figure(sp, make_overlay=not args.no_overlay)
        print(summary_line(out))
    if not only:
        idx = write_index(specs)
        from collections import Counter
        print("índice:", dict(Counter(e["status"] for e in idx)))


if __name__ == "__main__":
    main()
