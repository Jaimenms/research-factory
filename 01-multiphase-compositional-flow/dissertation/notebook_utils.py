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
