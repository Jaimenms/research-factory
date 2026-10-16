"""Treino da rede a partir de dados previamente simulados: divisão 70/30 (§4.3), erro quadrático médio
nas saídas padronizadas, Adam (Kingma & Ba 2015) em lote completo com o ciclo de épocas dentro de um
`lax.scan` compilado, histórico do erro de treino e de teste por época, métricas de teste (R² por
saída, erro percentual médio, RMSE).

Uso: uv run python -m surrogate.train runs/surrogate/well.npz --out models/surrogate/well.json \\
         [--neurons 20 --layers 1 --activation relu --lr 0.01 --epochs 2000 --seed 0 --split-seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .model import MLP, Hyperparameters, forward, init_params


def split_indices(n: int, train_fraction=0.7, seed=0):
    """Permutação aleatória dividida em (treino, teste)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def r2_score(y_true, y_pred) -> np.ndarray:
    """R² = 1 − SQ_res/SQ_tot por coluna."""
    y_true, y_pred = np.atleast_2d(np.asarray(y_true, float)), np.atleast_2d(np.asarray(y_pred, float))
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot


def evaluate(model: MLP, X, Y) -> dict:
    """Métricas em unidades físicas, por saída."""
    Yp = model.predict(X)
    Y = np.asarray(Y, float)
    err = Yp - Y
    return {
        "r2": r2_score(Y, Yp).tolist(),
        "rmse": np.sqrt((err**2).mean(axis=0)).tolist(),
        "mean_abs_error": np.abs(err).mean(axis=0).tolist(),
        "mean_percent_error": (100.0 * np.abs(err) / np.abs(Y)).mean(axis=0).tolist(),
        "max_abs_error": np.abs(err).max(axis=0).tolist(),
        "n": int(len(Y)),
    }


def _standardize(X_train, Y_train):
    xm, xs = X_train.mean(axis=0), X_train.std(axis=0)
    ym, ys = Y_train.mean(axis=0), Y_train.std(axis=0)
    xs = np.where(xs > 0, xs, 1.0)
    ys = np.where(ys > 0, ys, 1.0)
    return xm, xs, ym, ys


def _fit(hp: Hyperparameters, sizes, seed, Xn, Yn, Xt, Yt):
    """Adam em lote completo, `hp.epochs` épocas num `lax.scan` compilado. Devolve (params, histórico)."""
    import jax
    import jax.numpy as jnp

    params0 = init_params(jax.random.PRNGKey(seed), sizes)

    def loss(params, x, y):
        return jnp.mean((forward(params, x, hp.activation) - y) ** 2)

    grad = jax.grad(loss)
    lr, b1, b2, eps = hp.learning_rate, hp.beta1, hp.beta2, hp.eps

    def step(carry, t):
        params, m, v = carry
        g = grad(params, Xn, Yn)
        m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_**2, v, g)
        mh = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1**t), m)
        vh = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2**t), v)
        params = jax.tree_util.tree_map(lambda p, mh_, vh_: p - lr * mh_ / (jnp.sqrt(vh_) + eps), params, mh, vh)
        hist = jnp.stack([loss(params, Xn, Yn), loss(params, Xt, Yt) if Xt is not None else jnp.nan])
        return (params, m, v), hist

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params0)

    @jax.jit
    def run(params0):
        (params, _, _), hist = jax.lax.scan(step, (params0, zeros, zeros), jnp.arange(1, hp.epochs + 1, dtype=float))
        return params, hist

    params, hist = run(params0)
    return [(np.asarray(W), np.asarray(b)) for W, b in params], np.asarray(hist)


def train(X, Y, hp: Hyperparameters = Hyperparameters(), seed=0, test=None, meta=None):
    """Treina em (X, Y) físicos; `test` = (X_test, Y_test) opcional para o histórico do erro de teste.
    Devolve (MLP, histórico (épocas, 2) com o EQM padronizado de treino e de teste)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if hp.activation not in ("relu", "sigmoid", "tanh", "identity"):
        raise ValueError(f"ativação desconhecida: {hp.activation}")
    xm, xs, ym, ys = _standardize(X, Y)
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys
    Xt = Yt = None
    if test is not None:
        Xt, Yt = (np.asarray(test[0], float) - xm) / xs, (np.asarray(test[1], float) - ym) / ys
    params, hist = _fit(hp, hp.sizes(X.shape[1], Y.shape[1]), seed, Xn, Yn, Xt, Yt)
    model = MLP(hp=hp, params=params, x_mean=xm, x_std=xs, y_mean=ym, y_std=ys, meta=dict(meta or {}))
    model.meta.update({"seed": seed, "n_train": int(len(X)), "final_train_mse_std": float(hist[-1, 0])})
    return model, hist


def train_dataset(dataset_path, hp: Hyperparameters, seed=0, split_seed=0, train_fraction=0.7):
    """Carrega a massa de dados, divide, treina e avalia; devolve (modelo, histórico, índices)."""
    from .dataset import load_dataset

    d = load_dataset(dataset_path)
    ok = np.where(d["ok"])[0]
    tr, te = split_indices(len(ok), train_fraction, seed=split_seed)
    tr, te = ok[tr], ok[te]
    X, Y = d["X"], d["Y"]
    meta = {
        "dataset": str(dataset_path),
        "segment": d["meta"].get("segment"),
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "n_test": int(len(te)),
    }
    model, hist = train(X[tr], Y[tr], hp, seed=seed, test=(X[te], Y[te]), meta=meta)
    model.meta["test"] = evaluate(model, X[te], Y[te])
    model.meta["train"] = evaluate(model, X[tr], Y[tr])
    model.input_names = tuple(d["meta"]["input_columns"])
    model.output_names = tuple(d["meta"]["output_columns"])
    return model, hist, (tr, te)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True, help="modelo treinado (JSON)")
    ap.add_argument("--neurons", type=int, default=20)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--activation", default="relu", choices=("relu", "sigmoid", "tanh"))
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--history", help="NPZ com o histórico do erro e os índices de treino/teste")
    a = ap.parse_args(argv)
    hp = Hyperparameters(
        neurons=a.neurons, layers=a.layers, activation=a.activation, learning_rate=a.lr, epochs=a.epochs
    )
    model, hist, (tr, te) = train_dataset(a.dataset, hp, seed=a.seed, split_seed=a.split_seed)
    model.save(a.out)
    if a.history:
        Path(a.history).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.history, history=hist, train_idx=tr, test_idx=te)
    t = model.meta["test"]
    print(
        f"gravado {a.out}: {model.n_parameters} parâmetros; teste R² = {', '.join(f'{v:.3f}' for v in t['r2'])}; "
        f"erro percentual médio = {', '.join(f'{v:.2f} %' for v in t['mean_percent_error'])}"
    )


if __name__ == "__main__":
    main()
