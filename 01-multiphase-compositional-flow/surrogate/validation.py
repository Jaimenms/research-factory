"""Validação dos substitutos: sobreajuste, viés e a escolha dos conjuntos de treino e de teste.

Com dados sintéticos baratos a tentação é treinar com tudo; aqui medem-se, para o substituto de saída e
para o de perfil: (i) validação cruzada k-fold (média e desvio do erro de teste entre dobras); (ii) curvas
de aprendizagem, erro de teste e de treino contra o número de perfis de treino, com repetições por semente
(a diferença treino–teste é a medida de sobreajuste; o desvio entre repetições, a variância); (iii) o viés,
erro médio com sinal; (iv) três estratégias de divisão para os perfis: por perfil (generalização a
condições de entrada novas), por posição (todas as condições, mas 30 % dos pontos ao longo de x nunca
vistos: interpolação em x) e por bloco (teste = perfis com P_in no topo da faixa: extrapolação); (v) o
resíduo físico num conjunto de colocação nunca usado (independente dos dados).

Uso: uv run python -m surrogate.validation --out runs/surrogate/validation.json [--k 5 --repeats 3]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .dataset import load_dataset
from .model import Hyperparameters, MLP
from .profile import PROFILE_INPUTS, PROFILE_OUTPUTS, ProfileMLP, profile_rows
from .sampling import latin_hypercube
from .train import _standardize, r2_score, split_indices, train

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ("well", "line", "riser")
OUTLET_HP = Hyperparameters(neurons=20, layers=1, activation="relu", learning_rate=0.01, epochs=2000)
PROFILE_HP = Hyperparameters(neurons=32, layers=2, activation="tanh", learning_rate=0.01, epochs=1500)
SPLIT_MODES = ("profile", "position", "block")


def kfold_indices(n: int, k: int, seed=0):
    """k dobras de uma permutação aleatória: lista de (treino, teste)."""
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, k)
    return [(np.sort(np.concatenate([f for j, f in enumerate(folds) if j != i])), np.sort(folds[i])) for i in range(k)]


def metrics(Y, Yp) -> dict:
    """RMSE, viés (erro médio com sinal), R² e máximo por saída, em unidades físicas."""
    Y, Yp = np.asarray(Y, float), np.asarray(Yp, float)
    err = Yp - Y
    return {
        "rmse": np.sqrt((err**2).mean(axis=0)).tolist(),
        "bias": err.mean(axis=0).tolist(),
        "r2": r2_score(Y, Yp).tolist(),
        "max_abs_error": np.abs(err).max(axis=0).tolist(),
        "n": int(len(Y)),
    }


# ----------------------------------------------------------------------------------------- saída


def cross_validate_outlet(dataset_path, hp=OUTLET_HP, k=5, seed=0) -> dict:
    d = load_dataset(dataset_path)
    ok = np.where(d["ok"])[0]
    X, Y = d["X"][ok], d["Y"][ok]
    folds = []
    for i, (tr, te) in enumerate(kfold_indices(len(ok), k, seed)):
        model, _ = train(X[tr], Y[tr], hp, seed=seed + i)
        folds.append({"train": metrics(Y[tr], model.predict(X[tr])), "test": metrics(Y[te], model.predict(X[te]))})
    return {"k": k, "folds": folds, "summary": _summary(folds)}


def _summary(folds):
    """Média e desvio padrão entre dobras/repetições do RMSE e do viés, e a diferença treino–teste."""
    te = np.array([f["test"]["rmse"] for f in folds])
    tr = np.array([f["train"]["rmse"] for f in folds])
    bias = np.array([f["test"]["bias"] for f in folds])
    return {
        "test_rmse_mean": te.mean(axis=0).tolist(),
        "test_rmse_std": te.std(axis=0).tolist(),
        "train_rmse_mean": tr.mean(axis=0).tolist(),
        "gap_mean": (te - tr).mean(axis=0).tolist(),
        "bias_mean": bias.mean(axis=0).tolist(),
        "bias_std": bias.std(axis=0).tolist(),
    }


def learning_curve_outlet(dataset_path, hp=OUTLET_HP, sizes=(10, 20, 50, 100, 150, 210), n_repeats=3, seed=0) -> dict:
    d = load_dataset(dataset_path)
    ok = np.where(d["ok"])[0]
    X, Y = d["X"][ok], d["Y"][ok]
    curve = []
    for n in sizes:
        reps = []
        for r in range(n_repeats):
            tr, te = split_indices(len(ok), 0.7, seed=seed + r)
            tr = tr[:n]
            model, _ = train(X[tr], Y[tr], hp, seed=seed + r)
            reps.append({"train": metrics(Y[tr], model.predict(X[tr])), "test": metrics(Y[te], model.predict(X[te]))})
        curve.append({"n_train": int(n), **_summary(reps)})
    return {"sizes": list(sizes), "n_repeats": n_repeats, "curve": curve}


# ---------------------------------------------------------------------------------------- perfis


def profile_split(d: dict, mode: str, seed=0, test_fraction=0.3, n_train=None):
    """Linhas (features, alvos, y, y_in) de treino e de teste segundo a estratégia `mode`."""
    ok = np.where(d["ok"])[0]
    rng = np.random.default_rng(seed)
    if mode == "profile":
        tr, te = split_indices(len(ok), 1 - test_fraction, seed=seed)
        tr, te = ok[tr], ok[te]
        if n_train is not None:
            tr = tr[:n_train]
        return profile_rows(d, tr), profile_rows(d, te), {"train_profiles": int(len(tr)), "test_profiles": int(len(te))}
    if mode == "block":
        P_in = d["X"][ok, 0]
        cut = np.quantile(P_in, 1 - test_fraction)
        te, tr = ok[P_in >= cut], ok[P_in < cut]
        tr = tr[rng.permutation(len(tr))]
        if n_train is not None:
            tr = tr[:n_train]
        return (
            profile_rows(d, tr),
            profile_rows(d, te),
            {"train_profiles": int(len(tr)), "test_profiles": int(len(te)), "P_in_cut_bar": float(cut)},
        )
    if mode == "position":
        # todos os perfis em treino e teste, mas 30 % das posições s > 0 ficam só no teste
        n_out = d["x_out"].size
        pos = np.arange(1, n_out)
        test_pos = np.sort(rng.choice(pos, size=int(round(test_fraction * len(pos))), replace=False))
        train_pos = np.setdiff1d(pos, test_pos)
        prof = ok[rng.permutation(len(ok))]
        if n_train is not None:
            prof = prof[:n_train]
        rows = profile_rows(d, prof)  # linhas ordenadas por perfil e depois por posição (s > 0)
        s_grid = d["x_out"][1:] / float(d["x_out"][-1])
        s_rows = rows[0][:, 3]
        is_test = np.isin(np.searchsorted(s_grid, s_rows), test_pos - 1)
        pick = lambda m: tuple(a[m] for a in rows)  # noqa: E731
        return pick(~is_test), pick(is_test), {"train_profiles": int(len(prof)), "test_positions": test_pos.tolist()}
    raise ValueError(f"modo desconhecido: {mode}")


def _fit_profile(rows_tr, rows_te, hp, seed, L):
    feats_tr, tgt_tr, y_tr, _ = rows_tr
    feats_te, tgt_te, y_te, _ = rows_te
    mlp, _ = train(feats_tr, tgt_tr, hp, seed=seed)
    mlp.input_names, mlp.output_names = PROFILE_INPUTS, PROFILE_OUTPUTS
    model = ProfileMLP(mlp=mlp, L=L)
    return model, {
        "train": metrics(y_tr, model.predict_rows(feats_tr)),
        "test": metrics(y_te, model.predict_rows(feats_te)),
    }


def learning_curve_profile(
    dataset_path, hp=PROFILE_HP, sizes=(10, 20, 50, 100, 210), n_repeats=3, seed=0, mode="profile"
) -> dict:
    d = load_dataset(dataset_path)
    L = float(d["x_out"][-1])
    curve = []
    for n in sizes:
        reps = []
        for r in range(n_repeats):
            rows_tr, rows_te, info = profile_split(d, mode, seed=seed + r, n_train=n)
            _, m = _fit_profile(rows_tr, rows_te, hp, seed + r, L)
            reps.append(m)
        curve.append({"n_train": int(n), **_summary(reps)})
    return {"mode": mode, "sizes": list(sizes), "n_repeats": n_repeats, "curve": curve}


def heldout_residual(seg, dataset_path, case_path, models: dict, n=60, seed=7) -> dict:
    """RMS do resíduo físico num conjunto de colocação novo (semente diferente da do treino do PINN)."""
    from .pinn import residual_rms

    d = load_dataset(dataset_path)
    r = d["meta"]["ranges"]
    feats_c = latin_hypercube(n, [r["P_in_bar"], r["T_in_K"], r["m_kg_s"], [1.0 / n, 1.0]], seed=seed)
    return {name: residual_rms(model, case_path, feats_c, dataset=dataset_path) for name, model in models.items()}


def run_all(
    runs_dir, models_dir, cases_dir, k=5, repeats=3, seed=0, outlet_epochs=2000, profile_epochs=1500, progress=True
) -> dict:
    out = {}
    hp_o = Hyperparameters(**{**OUTLET_HP.__dict__, "epochs": outlet_epochs})
    hp_p = Hyperparameters(**{**PROFILE_HP.__dict__, "epochs": profile_epochs})
    for seg in SEGMENTS:
        t0 = time.time()
        res = {"outlet": {}, "profile": {}}
        ds_o, ds_p = runs_dir / f"{seg}.npz", runs_dir / f"{seg}_profile.npz"
        if ds_o.exists():
            res["outlet"]["kfold"] = cross_validate_outlet(ds_o, hp_o, k=k, seed=seed)
            res["outlet"]["learning_curve"] = learning_curve_outlet(ds_o, hp_o, n_repeats=repeats, seed=seed)
        if ds_p.exists():
            for mode in SPLIT_MODES:
                res["profile"][mode] = learning_curve_profile(ds_p, hp_p, n_repeats=repeats, seed=seed, mode=mode)
            models = {}
            for name, path in (
                ("data_full", models_dir / f"{seg}_profile.json"),
                ("data_sparse", runs_dir / f"{seg}_profile_n20.json"),
                ("pinn_from_sparse", models_dir / f"{seg}_pinn.json"),
                ("pinn_from_scratch", runs_dir / f"{seg}_pinn_from_scratch.json"),
            ):
                if path.exists():
                    models[name] = ProfileMLP.load(path)
            if models:
                res["profile"]["heldout_residual"] = heldout_residual(
                    seg, ds_p, cases_dir / f"{seg}.json", models, seed=seed + 7
                )
        res["cpu"] = time.time() - t0
        out[seg] = res
        if progress:
            lc = res["outlet"].get("learning_curve", {}).get("curve", [])
            print(
                f"[{seg}] validação em {res['cpu']:.0f} s; saída: "
                + "; ".join(
                    f"n={c['n_train']}: teste {c['test_rmse_mean'][0]:.3f}/{c['test_rmse_mean'][1]:.3f}, treino {c['train_rmse_mean'][0]:.3f}/{c['train_rmse_mean'][1]:.3f}"
                    for c in lc
                ),
                flush=True,
            )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs" / "surrogate")
    ap.add_argument("--models", type=Path, default=ROOT / "models" / "surrogate")
    ap.add_argument("--cases", type=Path, default=ROOT / "cases" / "surrogate")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "surrogate" / "validation.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = run_all(a.runs, a.models, a.cases, k=a.k, repeats=a.repeats, seed=a.seed)
    a.out.write_text(json.dumps(out, indent=1))
    print("gravado", a.out)


if __name__ == "__main__":
    main()
