"""Encadeia, num só processo (uma compilação por trecho), o que o §5.7 precisa: massa de dados por
trecho nas faixas das Tabelas 27–29 (hipercubo latino) e nas entradas do Apêndice B (comparação com o
modelo físico do autor), corrida de referência do sistema completo (Figuras 57–58), busca em grelha
(Apêndice C) e modelos finais com os hiperparâmetros da Tabela 31. Passos selecionáveis.

Uso: uv run python -m surrogate.pipeline [--steps datasets,reference,grid,models] [--n 300] [--seed 0]
         [--epochs 2000] [--runs runs/surrogate] [--models models/surrogate]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .dataset import TABLES, generate
from .grid import grid_search, write_csv
from .model import Hyperparameters
from .train import train_dataset

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ("well", "line", "riser")


def step_datasets(cases_dir, runs_dir, n, seed, criterion):
    for seg in SEGMENTS:
        case = cases_dir / f"{seg}.json"
        out, meta = generate(case, runs_dir / f"{seg}.npz", n=n, seed=seed, segment=seg, criterion=criterion)
        print(f"[{seg}] amostras próprias: {meta['n_ok']}/{n} válidas, {meta['cpu']:.1f} s")
        app_b = ROOT / "dissertation" / "data" / f"appendix_b_{seg}.csv"
        if app_b.exists():
            out, meta = generate(case, runs_dir / f"{seg}_appendix_b.npz", segment=seg, inputs_csv=app_b)
            print(f"[{seg}] entradas do Apêndice B: {meta['n_ok']} válidas, {meta['cpu']:.1f} s")


def step_reference(cases_dir, runs_dir):
    from cmfsim.io.spec import Case, load_json
    from cmfsim.simulation import simulate

    case = Case.from_dict(load_json(cases_dir / "offshore_table6_noslip.json"))
    res = simulate(case, backend="jax")
    res.save(runs_dir / "offshore_table6_noslip.npz")


def step_grid(runs_dir, epochs, seed, split_seed):
    for seg in SEGMENTS:
        t0 = time.time()
        rows = grid_search(runs_dir / f"{seg}.npz", epochs=epochs, seed=seed, split_seed=split_seed, progress=False)
        write_csv(rows, runs_dir / f"grid_{seg}.csv")
        good = sum(r["both_above_0_9"] for r in rows)
        print(f"[{seg}] grelha: {good}/135 redes com R² > 0,9 em P e T ({time.time() - t0:.0f} s)")


def step_models(runs_dir, models_dir, epochs, seed, split_seed):
    hp31 = json.load(open(TABLES))["table31_hyperparameters"]
    hp = Hyperparameters(
        neurons=hp31["neurons"],
        layers=hp31["layers"],
        activation=hp31["activation"],
        learning_rate=hp31["learning_rate"],
        epochs=epochs,
    )
    for seg in SEGMENTS:
        model, hist, (tr, te) = train_dataset(runs_dir / f"{seg}.npz", hp, seed=seed, split_seed=split_seed)
        model.meta["hyperparameters_source"] = "Tabela 31"
        model.save(models_dir / f"{seg}.json")
        np.savez(runs_dir / f"history_{seg}.npz", history=hist, train_idx=tr, test_idx=te)
        t = model.meta["test"]
        print(
            f"[{seg}] modelo final: R² teste P = {t['r2'][0]:.3f}, T = {t['r2'][1]:.3f}; erro percentual médio "
            f"P = {t['mean_percent_error'][0]:.2f} %, T = {t['mean_percent_error'][1]:.2f} %"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="datasets,reference,grid,models")
    ap.add_argument("--cases", type=Path, default=ROOT / "cases" / "surrogate")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs" / "surrogate")
    ap.add_argument("--models", type=Path, default=ROOT / "models" / "surrogate")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--criterion", default="random", choices=("random", "center", "maximin"))
    ap.add_argument("--epochs", type=int, default=2000)
    a = ap.parse_args(argv)
    steps = [s.strip() for s in a.steps.split(",")]
    a.runs.mkdir(parents=True, exist_ok=True)
    if "datasets" in steps:
        step_datasets(a.cases, a.runs, a.n, a.seed, a.criterion)
    if "reference" in steps:
        step_reference(a.cases, a.runs)
    if "grid" in steps:
        step_grid(a.runs, a.epochs, a.seed, a.split_seed)
    if "models" in steps:
        step_models(a.runs, a.models, a.epochs, a.seed, a.split_seed)


if __name__ == "__main__":
    main()
