"""Substituto de perfil: a posição axial entra na rede, que devolve P(x) e T(x) em vez de só a saída.
Entradas (P_in, T_in, m, s = x/L); a rede aprende a «queda por unidade de s», g = (y_in − y(x))/s, e o
perfil é y(x) = y_in − s·g(P_in, T_in, m, s): a condição de entrada (s = 0) fica satisfeita por
construção. Treino com os mesmos Adam e MLP de `surrogate.train`, sobre todos os pontos de saída dos
perfis gravados na massa de dados (`profiles`, Pa e K).

Uso: uv run python -m surrogate.profile runs/surrogate/well_profile.npz --out models/surrogate/well_profile.json \\
         [--neurons 32 --layers 2 --activation tanh --lr 0.01 --epochs 3000 --n-train 210]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import load_dataset
from .model import MLP, Hyperparameters
from .train import evaluate, split_indices, train

PROFILE_INPUTS = ("P_in_bar", "T_in_K", "m_kg_s", "s")
PROFILE_OUTPUTS = ("dP_per_s_bar", "dT_per_s_K")


def profile_rows(d: dict, idx):
    """Linhas (features, alvos) de todos os pontos s > 0 dos perfis `idx`: features (n, 4) e alvos (n, 2)
    = queda por unidade de s em bar e K. Devolve também (P, T) físicos e s para avaliação."""
    L = float(d["x_out"][-1])
    s = d["x_out"] / L
    X, prof = d["X"][idx], d["profiles"][idx]
    n, n_out = prof.shape[0], prof.shape[1]
    S = np.broadcast_to(s[None, :], (n, n_out))
    P, T = prof[:, :, 0] / 1e5, prof[:, :, 1]
    feats = np.stack(
        [np.repeat(X[:, 0], n_out), np.repeat(X[:, 1], n_out), np.repeat(X[:, 2], n_out), S.ravel()], axis=1
    )
    y = np.stack([P.ravel(), T.ravel()], axis=1)
    y_in = np.stack([np.repeat(X[:, 0], n_out), np.repeat(X[:, 1], n_out)], axis=1)
    keep = feats[:, 3] > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        target = (y_in - y) / feats[:, 3:4]
    return feats[keep], target[keep], y[keep], y_in[keep]


@dataclass
class ProfileMLP:
    """Modelo empírico de perfil: MLP da queda por unidade de s mais a forma y = y_in − s·g."""

    mlp: MLP
    L: float

    def predict(self, P_in_bar, T_in_K, m_kg_s, s) -> np.ndarray:
        """Perfil em (n,) pontos s = x/L para uma condição de entrada: (n, 2) em bar e K."""
        s = np.atleast_1d(np.asarray(s, float))
        X = np.stack([np.full_like(s, P_in_bar), np.full_like(s, T_in_K), np.full_like(s, m_kg_s), s], axis=1)
        g = self.mlp.predict(X)
        return np.stack([P_in_bar - s * g[:, 0], T_in_K - s * g[:, 1]], axis=1)

    def predict_rows(self, feats) -> np.ndarray:
        """Para linhas (n, 4) = (P_in, T_in, m, s): (n, 2) em bar e K."""
        feats = np.asarray(feats, float)
        g = self.mlp.predict(feats)
        return np.stack([feats[:, 0] - feats[:, 3] * g[:, 0], feats[:, 1] - feats[:, 3] * g[:, 1]], axis=1)

    def save(self, path):
        d = self.mlp.to_dict()
        d["format"] = "cmfsim-surrogate-profile-mlp/1"
        d["L"] = self.L
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
        return path

    @classmethod
    def load(cls, path) -> "ProfileMLP":
        d = json.loads(Path(path).read_text())
        if d.get("format") != "cmfsim-surrogate-profile-mlp/1":
            raise ValueError(f"formato desconhecido: {d.get('format')}")
        d["format"] = "cmfsim-surrogate-mlp/1"
        return cls(mlp=MLP.from_dict(d), L=float(d["L"]))


def evaluate_profile(model: ProfileMLP, d: dict, idx) -> dict:
    """Erros em unidades físicas sobre todos os pontos dos perfis `idx` e só na saída."""
    feats, _, y, _ = profile_rows(d, idx)
    yp = model.predict_rows(feats)
    err = yp - y
    out_mask = np.isclose(feats[:, 3], 1.0)
    return {
        "rmse": np.sqrt((err**2).mean(axis=0)).tolist(),
        "max_abs_error": np.abs(err).max(axis=0).tolist(),
        "mean_percent_error": (100.0 * np.abs(err) / np.abs(y)).mean(axis=0).tolist(),
        "outlet_rmse": np.sqrt((err[out_mask] ** 2).mean(axis=0)).tolist(),
        "outlet_mean_percent_error": (100.0 * np.abs(err[out_mask]) / np.abs(y[out_mask])).mean(axis=0).tolist(),
        "n_profiles": int(len(idx)),
        "n_points": int(len(y)),
    }


def train_profile(dataset_path, hp: Hyperparameters, n_train=None, seed=0, split_seed=0, train_fraction=0.7):
    """Divide os perfis (não os pontos) em treino/teste, treina a rede da queda por unidade de s e avalia.
    `n_train` limita o número de perfis de treino (estudo de dados escassos)."""
    d = load_dataset(dataset_path)
    ok = np.where(d["ok"])[0]
    tr, te = split_indices(len(ok), train_fraction, seed=split_seed)
    tr, te = ok[tr], ok[te]
    if n_train is not None:
        tr = tr[:n_train]
    feats_tr, tgt_tr, _, _ = profile_rows(d, tr)
    feats_te, tgt_te, _, _ = profile_rows(d, te)
    meta = {
        "dataset": str(dataset_path),
        "segment": d["meta"].get("segment"),
        "split_seed": split_seed,
        "n_train_profiles": int(len(tr)),
        "n_test_profiles": int(len(te)),
    }
    mlp, hist = train(feats_tr, tgt_tr, hp, seed=seed, test=(feats_te, tgt_te), meta=meta)
    mlp.input_names, mlp.output_names = PROFILE_INPUTS, PROFILE_OUTPUTS
    model = ProfileMLP(mlp=mlp, L=float(d["x_out"][-1]))
    mlp.meta["test_profile"] = evaluate_profile(model, d, te)
    mlp.meta["train_profile"] = evaluate_profile(model, d, tr)
    mlp.meta["test_target"] = evaluate(mlp, feats_te, tgt_te)
    return model, hist, (tr, te)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neurons", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--activation", default="tanh", choices=("relu", "sigmoid", "tanh"))
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--n-train", type=int, help="limita os perfis de treino")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--history")
    a = ap.parse_args(argv)
    hp = Hyperparameters(
        neurons=a.neurons, layers=a.layers, activation=a.activation, learning_rate=a.lr, epochs=a.epochs
    )
    model, hist, (tr, te) = train_profile(a.dataset, hp, n_train=a.n_train, seed=a.seed, split_seed=a.split_seed)
    model.save(a.out)
    if a.history:
        np.savez(a.history, history=hist, train_idx=tr, test_idx=te)
    t = model.mlp.meta["test_profile"]
    print(
        f"gravado {a.out}: perfis de teste {t['n_profiles']} ({t['n_points']} pontos): RMSE P {t['rmse'][0]:.3f} bar, "
        f"T {t['rmse'][1]:.3f} K; máx {t['max_abs_error'][0]:.2f} bar, {t['max_abs_error'][1]:.2f} K; "
        f"saída RMSE {t['outlet_rmse'][0]:.3f} bar, {t['outlet_rmse'][1]:.3f} K"
    )


if __name__ == "__main__":
    main()
