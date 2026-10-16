"""Predição com os modelos empíricos treinados, isolados ou encadeados poço → linha → riser (a saída de
um trecho é a entrada do seguinte, com a mesma vazão), como nas Figuras 57–58.

Uso: uv run python -m surrogate.predict models/surrogate/well.json models/surrogate/line.json \\
         models/surrogate/riser.json --P 170 --T 353.15 --m 10
"""

from __future__ import annotations

import argparse

import numpy as np

from .model import MLP


def chain_predict(models, P_in_bar, T_in_K, m_kg_s) -> np.ndarray:
    """Devolve (n_trechos, n, 2): (P_out bar, T_out K) à saída de cada trecho para entradas (n,)."""
    P, T, m = (np.atleast_1d(np.asarray(v, float)) for v in (P_in_bar, T_in_K, m_kg_s))
    out = []
    for model in models:
        Y = model.predict(np.stack([P, T, m], axis=1))
        P, T = Y[:, 0], Y[:, 1]
        out.append(Y)
    return np.stack(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="modelos JSON na ordem do escoamento")
    ap.add_argument("--P", type=float, required=True, help="pressão na entrada [bar]")
    ap.add_argument("--T", type=float, required=True, help="temperatura na entrada [K]")
    ap.add_argument("--m", type=float, required=True, help="vazão mássica [kg/s]")
    a = ap.parse_args(argv)
    models = [MLP.load(p) for p in a.models]
    Y = chain_predict(models, a.P, a.T, a.m)
    for path, model, y in zip(a.models, models, Y):
        print(
            f"{model.meta.get('segment') or path}: P_out = {y[0, 0]:.2f} bar, T_out = {y[0, 1]:.2f} K ({y[0, 1] - 273.15:.2f} °C)"
        )


if __name__ == "__main__":
    main()
