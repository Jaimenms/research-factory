"""Encadeia, num só processo (uma compilação por trecho), o que o §5.7 precisa e as extensões (perfil, PINN): massa de dados por
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


PROFILE_DX = {"well": 30.0, "line": 120.0, "riser": 6.0}  # 51 pontos de saída por trecho
PROFILE_HP = Hyperparameters(neurons=32, layers=2, activation="tanh", learning_rate=0.01, epochs=3000)


def step_profiles(cases_dir, runs_dir, n, seed, criterion):
    """Massa de dados com perfis finos (51 pontos) para o substituto de perfil e o PINN."""
    for seg in SEGMENTS:
        out, meta = generate(
            cases_dir / f"{seg}.json",
            runs_dir / f"{seg}_profile.npz",
            n=n,
            seed=seed,
            segment=seg,
            criterion=criterion,
            overrides=[f"outputs.dx_out={PROFILE_DX[seg]}"],
        )
        print(f"[{seg}] perfis: {meta['n_ok']}/{n} válidos, {meta['cpu']:.1f} s")


def step_profile_models(runs_dir, models_dir, seed, split_seed, n_sparse=20):
    """Substituto de perfil só com dados: todos os perfis de treino (modelo final) e `n_sparse` perfis
    (ponto de partida do estudo PINN)."""
    from .profile import train_profile

    for seg in SEGMENTS:
        for n_train, out in (
            (None, models_dir / f"{seg}_profile.json"),
            (n_sparse, runs_dir / f"{seg}_profile_n{n_sparse}.json"),
        ):
            model, hist, (tr, te) = train_profile(
                runs_dir / f"{seg}_profile.npz", PROFILE_HP, n_train=n_train, seed=seed, split_seed=split_seed
            )
            model.save(out)
            np.savez(runs_dir / f"history_{out.stem}.npz", history=hist, train_idx=tr, test_idx=te)
            t = model.mlp.meta["test_profile"]
            print(
                f"[{seg}] perfil ({len(tr)} perfis de treino): RMSE teste P {t['rmse'][0]:.3f} bar, T {t['rmse'][1]:.3f} K"
            )


def step_pinn(
    cases_dir,
    runs_dir,
    models_dir,
    epochs,
    seed,
    split_seed,
    n_sparse=20,
    n_collocation=150,
    lambda_phys=1.0,
    segments=SEGMENTS,
    reuse=False,
):
    """Estudo PINN com dados escassos: o modelo só de dados com `n_sparse` perfis, o mesmo com o resíduo
    físico na perda (a partir dos pesos do primeiro) e o PINN treinado do zero; resíduo físico e erro
    nos perfis de teste de cada um, em `pinn_study.json` (fundido com o que lá estiver). Com `reuse`,
    as redes PINN já gravadas não são treinadas de novo (só avaliadas)."""
    from .pinn import residual_rms, train_pinn
    from .profile import ProfileMLP

    study_path = runs_dir / "pinn_study.json"
    results = json.loads(study_path.read_text()) if study_path.exists() else {}
    for seg in segments:
        ds, case = runs_dir / f"{seg}_profile.npz", cases_dir / f"{seg}.json"
        data_only = ProfileMLP.load(runs_dir / f"{seg}_profile_n{n_sparse}.json")
        hp = Hyperparameters(**{**PROFILE_HP.__dict__, "epochs": epochs})
        t0 = time.time()
        p_init, p_zero = models_dir / f"{seg}_pinn.json", runs_dir / f"{seg}_pinn_from_scratch.json"
        if reuse and p_init.exists() and p_zero.exists():
            pinn_init, pinn_zero = ProfileMLP.load(p_init), ProfileMLP.load(p_zero)
            feats_c = np.load(runs_dir / f"history_{seg}_pinn.npz")["collocation"]
        else:
            pinn_init, hist_i, _, feats_c = train_pinn(
                ds,
                case,
                hp,
                n_train=n_sparse,
                n_collocation=n_collocation,
                lambda_phys=lambda_phys,
                seed=seed,
                split_seed=split_seed,
                init_from=data_only,
                progress=False,
            )
            pinn_init.save(p_init)
            np.savez(runs_dir / f"history_{seg}_pinn.npz", history=hist_i, collocation=np.asarray(feats_c))
            pinn_zero, hist_z, _, _ = train_pinn(
                ds,
                case,
                hp,
                n_train=n_sparse,
                n_collocation=n_collocation,
                lambda_phys=lambda_phys,
                seed=seed,
                split_seed=split_seed,
                progress=False,
            )
            pinn_zero.save(p_zero)
            np.savez(runs_dir / f"history_{seg}_pinn_from_scratch.npz", history=hist_z)
        full = ProfileMLP.load(models_dir / f"{seg}_profile.json")
        rows = {}
        for name, model in (
            ("data_full", full),
            ("data_sparse", data_only),
            ("pinn_from_sparse", pinn_init),
            ("pinn_from_scratch", pinn_zero),
        ):
            rows[name] = {
                "n_train_profiles": model.mlp.meta.get("n_train_profiles") or model.mlp.meta.get("n_train"),
                "test_profile": model.mlp.meta["test_profile"],
                "residual_rms": residual_rms(model, case, np.asarray(feats_c), dataset=ds),
            }
        results[seg] = {
            "n_collocation": int(len(feats_c)),
            "lambda_phys": lambda_phys,
            "epochs": epochs,
            "cpu": time.time() - t0,
            "models": rows,
        }
        print(
            f"[{seg}] PINN ({time.time() - t0:.0f} s): "
            + "; ".join(
                f"{k}: RMSE P {v['test_profile']['rmse'][0]:.3f} T {v['test_profile']['rmse'][1]:.3f}, resíduo {v['residual_rms']:.3f}"
                for k, v in rows.items()
            ),
            flush=True,
        )
        study_path.write_text(json.dumps(results, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="datasets,reference,grid,models", help="também: profiles, profile_models, pinn")
    ap.add_argument("--cases", type=Path, default=ROOT / "cases" / "surrogate")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs" / "surrogate")
    ap.add_argument("--models", type=Path, default=ROOT / "models" / "surrogate")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--criterion", default="random", choices=("random", "center", "maximin"))
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--pinn-epochs", type=int, default=600)
    ap.add_argument("--n-collocation", type=int, default=150)
    ap.add_argument("--lambda-phys", type=float, default=1.0)
    ap.add_argument("--segments", help="trechos do passo pinn, separados por vírgula (omissão: todos)")
    ap.add_argument("--reuse", action="store_true", help="passo pinn: não treina de novo as redes já gravadas")
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
    if "profiles" in steps:
        step_profiles(a.cases, a.runs, a.n, a.seed, a.criterion)
    if "profile_models" in steps:
        step_profile_models(a.runs, a.models, a.seed, a.split_seed)
    if "pinn" in steps:
        segs = tuple(x.strip() for x in a.segments.split(",")) if a.segments else SEGMENTS
        step_pinn(
            a.cases,
            a.runs,
            a.models,
            a.pinn_epochs,
            a.seed,
            a.split_seed,
            n_collocation=a.n_collocation,
            lambda_phys=a.lambda_phys,
            segments=segs,
            reuse=a.reuse,
        )


if __name__ == "__main__":
    main()
