"""Apoio às figuras dos artigos (importado por `papers/<artigo>/figures/make_figures.py`).

Dá os caminhos do projeto (corridas em `runs/`, figuras digitalizadas, dados da dissertação), o estilo
matplotlib de artigo, as larguras de coluna da Elsevier, uma paleta categórica em ordem fixa por
entidade e a gravação em PDF (vetorial, lido pelo LaTeX) mais PNG (pré-visualização). Os guiões de
figuras só leem corridas gravadas; não simulam.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COMMON = Path(__file__).resolve().parent
PROJECT = COMMON.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

RUNS = Path(os.environ.get("CMFSIM_RUNS", PROJECT / "runs"))
DIGITIZED = PROJECT / "references" / "figures" / "data"
DATA = PROJECT / "dissertation" / "data"

MM = 1 / 25.4
SINGLE, MID, DOUBLE = 90 * MM, 140 * MM, 190 * MM  # larguras de figura da Elsevier, em polegadas

# Paleta categórica (Okabe–Ito, segura para daltonismo) atribuída por entidade, nunca por posição:
# a mesma entidade tem a mesma cor em todas as figuras de todos os artigos.
COLORS = {
    "noslip": "#0072B2",  # este trabalho, modelo de não escorregamento
    "mechanistic": "#D55E00",  # este trabalho, modelo mecanicista com escorregamento
    "reference": "#000000",  # dissertação (curvas digitalizadas)
    "hysys_tulsa": "#009E73",
    "hysys_bb": "#E69F00",
    "hysys_other": "#CC79A7",
}


def use_style():
    plt.style.use(str(COMMON / "cmfsim-paper.mplstyle"))


def figsize(width=SINGLE, ratio=0.72):
    return (width, width * ratio)


def load_run(path):
    from cmfsim import load_run as _load_run

    return _load_run(path)


def load_runs(pattern):
    """Corridas `RUNS/<pattern>` (glob) como dict nome -> Results, por ordem de nome."""
    return {p.stem: load_run(p) for p in sorted(RUNS.glob(pattern))}


def digitized(n):
    """Séries da figura digitalizada n: dict nome -> (x, y) ordenados por x (painel|nome quando há painéis)."""
    d = json.load(open(DIGITIZED / f"fig_{n:02d}.json"))
    out = {}
    for s in d["series"]:
        x, y = np.array(s["x"], float), np.array(s["y"], float)
        o = np.argsort(x)
        key = s["name"] if s.get("panel") is None else f"{s['panel']}|{s['name']}"
        out[key] = (x[o], y[o])
    return out


def appendix_a(orientation):
    """Linhas do Apêndice A (`dissertation/data/appendix_a_<orientation>.csv`) como dicts de strings."""
    with open(DATA / f"appendix_a_{orientation}.csv") as f:
        return list(csv.DictReader(f))


def panel_label(ax, text):
    """Letra do painel, (a), (b)..., no canto superior esquerdo, fora da área de dados."""
    ax.text(-0.02, 1.02, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")


def placeholder(ax, text):
    """Aviso no lugar dos dados (corrida em falta), para o PDF compilar na mesma."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=7, wrap=True, color="0.35")
    ax.set_xticks([])
    ax.set_yticks([])


def save(fig, out_dir, name, png=True):
    """Grava `out_dir/name.pdf` (e `.png` a 300 dpi) e fecha a figura; devolve o caminho do PDF."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf)
    if png:
        fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    print("figura:", pdf)
    return pdf


def run_cli(figures, description):
    """Linha de comando comum: `--out DIR` (obrigatório), `--only NOME` (repetível), `--no-png`.

    `figures` é um dict nome -> função(out_dir, png) que grava a figura com esse nome.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", required=True, type=Path, help="pasta das figuras geradas")
    parser.add_argument("--only", action="append", choices=sorted(figures), help="gerar só esta figura (repetível)")
    parser.add_argument("--no-png", action="store_true", help="não gravar a pré-visualização PNG")
    args = parser.parse_args()
    use_style()
    for name in args.only or sorted(figures):
        figures[name](args.out, not args.no_png)
    return args
