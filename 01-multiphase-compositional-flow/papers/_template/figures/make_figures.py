"""Figuras do artigo @@SLUG@@ a partir das corridas gravadas em `runs/` (não simula).

uso: uv run python figures/make_figures.py --out figures/generated [--only NOME] [--no-png]
     (a partir da pasta do artigo; `make figures` faz o mesmo)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_common"))

import matplotlib.pyplot as plt  # noqa: E402

import figlib as fl  # noqa: E402


def example(out, png):
    """Exemplo: substituir por figuras que leem corridas com fl.load_run / fl.load_runs."""
    fig, ax = plt.subplots(figsize=fl.figsize(fl.SINGLE))
    fl.placeholder(ax, "figura por fazer (figures/make_figures.py)")
    fl.save(fig, out, "example", png)


FIGURES = {"example": example}

if __name__ == "__main__":
    fl.run_cli(FIGURES, __doc__.splitlines()[0])
