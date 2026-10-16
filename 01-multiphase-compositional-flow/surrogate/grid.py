"""Busca em grelha dos hiperparâmetros (§5.7): camadas {1, 2, 3} × ativação {relu, sigmoid, tanh} ×
neurónios {5, 10, 15, 20, 25} × taxa {0,1, 0,01, 0,001} = 135 redes, treinadas com os mesmos dados e
avaliadas pelo R² de teste em P e T, na ordem de numeração do Apêndice C. Grava CSV.

Uso: uv run python -m surrogate.grid runs/surrogate/well.npz --out runs/surrogate/grid_well.csv [--epochs 2000]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .dataset import TABLES
from .model import Hyperparameters
from .train import train_dataset

COLUMNS = ("case", "neurons", "learning_rate", "activation", "layers", "r2_P", "r2_T", "r2_sum", "both_above_0_9")


def thesis_grid() -> list[Hyperparameters]:
    g = json.load(open(TABLES))["grid"]
    out = []
    for n in g["neurons"]:
        for lr in g["learning_rate"]:
            for act in g["activation"]:
                for layers in g["layers"]:
                    out.append(Hyperparameters(neurons=n, layers=layers, activation=act, learning_rate=lr))
    return out


def grid_search(dataset_path, grid=None, epochs=2000, seed=0, split_seed=0, progress=True):
    rows = []
    t0 = time.time()
    for k, hp0 in enumerate(grid or thesis_grid()):
        hp = Hyperparameters(**{**hp0.__dict__, "epochs": epochs})
        model, _, _ = train_dataset(dataset_path, hp, seed=seed, split_seed=split_seed)
        r2 = model.meta["test"]["r2"]
        rows.append(
            dict(
                case=k,
                neurons=hp.neurons,
                learning_rate=hp.learning_rate,
                activation=hp.activation,
                layers=hp.layers,
                r2_P=float(r2[0]),
                r2_T=float(r2[1]),
                r2_sum=float(r2[0] + r2[1]),
                both_above_0_9=bool(r2[0] > 0.9 and r2[1] > 0.9),
            )
        )
        if progress and (k + 1) % 15 == 0:
            print(f"  {k + 1} redes em {time.time() - t0:.0f} s")
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "r2_P": f"{r['r2_P']:.4f}", "r2_T": f"{r['r2_T']:.4f}", "r2_sum": f"{r['r2_sum']:.4f}"})
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args(argv)
    rows = grid_search(a.dataset, epochs=a.epochs, seed=a.seed, split_seed=a.split_seed)
    write_csv(rows, a.out)
    good = [r for r in rows if r["both_above_0_9"]]
    print(f"gravado {a.out}: {len(good)} de {len(rows)} redes com R² > 0,9 em P e T")


if __name__ == "__main__":
    main()
