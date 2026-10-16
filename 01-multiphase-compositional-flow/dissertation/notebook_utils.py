"""Apoio aos notebooks de reprodução (`notebooks_src/thesis/nb_5_*.py`): caminhos, leitura das
figuras digitalizadas e dos dados da dissertação, fluidos e utilidades de desenho. Fora do simulador."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIG = ROOT / "references" / "figures" / "data"
DATA = ROOT / "dissertation" / "data"
RUNS = Path(os.environ.get("CMFSIM_RUNS", ROOT / "runs"))
OUT = ROOT / "notebooks" / "figs"
OUT.mkdir(parents=True, exist_ok=True)


def figure(n):
    """Figura digitalizada n: dict com `caption`, `axes`, `series` (name -> (x, y) ordenados por x) e `raw`."""
    d = json.load(open(FIG / f"fig_{n:02d}.json"))
    series = {}
    for s in d["series"]:
        x, y = np.array(s["x"], float), np.array(s["y"], float)
        o = np.argsort(x)
        key = s["name"] if s.get("panel") is None else f"{s['panel']}|{s['name']}"
        series[key] = (x[o], y[o])
    return dict(caption=d["caption"], axes=d["axes"], series=series, raw=d)


def appendix_a(orientation):
    with open(DATA / f"appendix_a_{orientation}.csv") as f:
        return list(csv.DictReader(f))


def components_db():
    return json.load(open(DATA / "components.json"))["components"]


def fluids_db():
    return json.load(open(DATA / "fluids.json"))["fluids"]


def tables():
    return json.load(open(DATA / "cases_tables.json"))


def make_fluid(name):
    """(ComponentSet, z) de um fluido de `fluids.json`."""
    from cmfsim.thermo import ComponentSet

    comps, fl = components_db(), fluids_db()[name]
    names = list(fl)
    cs = ComponentSet.from_records([dict(id=k, **comps[k]) for k in names])
    z = np.array([fl[k] for k in names])
    return cs, z / z.sum()


def load_runs(pattern):
    """Corridas `RUNS/<pattern>` (glob) como dict nome -> Results."""
    from cmfsim import load_run

    return {p.stem: load_run(p) for p in sorted(RUNS.glob(pattern))}


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print("figura:", path.relative_to(ROOT))
    return path


def table(rows, headers, fmt=None):
    """Imprime uma tabela simples em Markdown."""
    fmt = fmt or {}
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for r in rows:
        cells = []
        for h, v in zip(headers, r):
            f = fmt.get(h)
            cells.append(f.format(v) if f and v is not None and not (isinstance(v, float) and np.isnan(v)) else str(v))
        print("| " + " | ".join(cells) + " |")


def merge_like_figure(values, fig_data, diam_px=None):
    """Funde os nossos valores como a digitalização funde os marcadores sobrepostos de uma paridade:
    dois marcadores a menos de um diâmetro (em píxeis da figura, ao longo da diagonal) viram um só.
    Devolve os valores fundidos (ordenados). `diam_px` por omissão vem da área do marcador na spec."""
    import numpy as np

    raw = fig_data["raw"]
    s0 = raw["series"][0]
    if diam_px is None:
        area = s0.get("marker_area_px")
        diam_px = 2.0 * np.sqrt(area / np.pi) if area else 12.0
    xt = np.array(raw["axes"]["x"]["ticks"], float)
    px_per_val = abs(xt[-1, 1] - xt[0, 1]) / abs(xt[-1, 0] - xt[0, 0])
    tol = diam_px / (np.sqrt(2.0) * px_per_val)  # distância ao longo da diagonal x = y
    v = np.sort(np.asarray(values)[np.isfinite(values)])
    out = [v[0]]
    for a in v[1:]:
        if a - out[-1] > tol:
            out.append(a)
    return np.array(out)


def qq_parity(ax, fig_data, mine, n_points, unit_scale=1.0, q=50):
    """Q–Q dos nossos valores contra os dois eixos de uma paridade digitalizada «Implementado vs
    HYSYS» (Figuras 14–21): y = HYSYS (referência), x = implementação da dissertação. As
    distribuições não são emparelhadas (não se sabe a que (P, T) corresponde cada marcador) e os
    marcadores sobrepostos nas zonas densas fundem-se na digitalização, pelo que a comparação
    honesta é com os nossos valores **fundidos da mesma maneira** (`merge_like_figure`), a cheio;
    os 159 valores tal e qual ficam a tracejado. Devolve (mediana fundida/HYSYS, mediana crua/HYSYS,
    n marcadores da figura, n marcadores fundidos)."""
    import numpy as np

    name, (xd, yd) = next(iter(fig_data["series"].items()))
    raw = fig_data["raw"]
    n_mark = raw["series"][0].get("n_markers", xd.size)
    m = np.sort(np.asarray(mine)[np.isfinite(mine)] * unit_scale)
    mm = merge_like_figure(m, fig_data)
    qs = np.linspace(0, 1, q)
    ax.plot(np.quantile(np.sort(yd), qs), np.quantile(mm, qs), "o-", ms=3, label=f"HYSYS × cmfsim fundido ({mm.size})")
    ax.plot(np.quantile(np.sort(yd), qs), np.quantile(m, qs), "--", lw=0.8, color="C0", label="HYSYS × cmfsim (159)")
    ax.plot(
        np.quantile(np.sort(xd), qs),
        np.quantile(mm, qs),
        "-",
        lw=0.8,
        color="gray",
        label="dissertação × cmfsim fundido",
    )
    lo = min(xd.min(), yd.min(), m.min())
    hi = max(xd.max(), yd.max(), m.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_title(
        f"Figura {raw['figure']} — {fig_data['caption']} ({n_mark} marcadores de {n_points} pontos)", fontsize=9
    )
    ax.set_xlabel("quantis da figura digitalizada")
    ax.set_ylabel("quantis do cmfsim")
    ax.grid(alpha=0.3)
    return float(np.median(mm) / np.median(yd)), float(np.median(m) / np.median(yd)), int(n_mark), int(mm.size)
