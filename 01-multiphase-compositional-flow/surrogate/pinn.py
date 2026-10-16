"""Rede com informação da física (PINN) para o substituto de perfil: além do erro nos dados, a perda inclui
o resíduo das equações do modelo de não escorregamento, r(s) = du/dx − f(x, u), avaliado em pontos de
colocação (P_in, T_in, m, s) que não precisam de ser simulados. `f` é o lado direito do simulador
(`NoSlipModel.rhs`: fluxos conservados, J = ∂Φ/∂u por AD, flash dentro), chamado sem o alterar. Como o
flash corre num `while_loop` (sem modo inverso; a retropropagação por ele nem sequer compila), f e
∂f/∂u entram como valores (f pelo simulador, ∂f/∂u por diferenças avançadas) e o gradiente em
relação aos pesos vem de uma perda auxiliar 2·mean(r₀/escala² · (du/dx − J₀ u)), cujo gradiente é
exatamente o de mean((r/escala)²). A rede é a de `surrogate.profile` (y = y_in − s·g).

Uso: uv run python -m surrogate.pinn runs/surrogate/well_profile.npz cases/surrogate/well.json \\
         --out models/surrogate/well_pinn.json [--n-train 20 --n-collocation 150 --lambda-phys 1 --epochs 1000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .dataset import load_dataset
from .model import MLP, Hyperparameters, forward, init_params
from .profile import PROFILE_INPUTS, PROFILE_OUTPUTS, ProfileMLP, evaluate_profile, profile_rows
from .sampling import latin_hypercube
from .train import _standardize, split_indices


def make_physics(case_path, fd_steps=(1.0e3, 1.0e-2)):
    """(f_and_J, L, case): `f_and_J(u, x, m)` = (du/dx (2,) em Pa/m e K/m, ∂f/∂u (2, 2)) do modelo de não
    escorregamento num ponto: f pelo simulador (modo direto) e ∂f/∂u por diferenças avançadas com passos
    `fd_steps` = (dP [Pa], dT [K]); L = comprimento."""
    import jax
    import jax.numpy as jnp

    from cmfsim.backend import get_backend
    from cmfsim.io.spec import Case, load_json
    from cmfsim.simulation import build_noslip

    case = Case.from_dict(load_json(case_path))
    model, params0 = build_noslip(case, get_backend("jax"))

    def f_point(u, x, m):
        p = dict(params0)
        p["m"] = m
        return model.rhs(x, u, 0, p)

    dP, dT = fd_steps

    def f_and_J(u, x, m):
        # ∂f/∂u por diferenças avançadas (como as eqs. 169–170 da dissertação): f já contém uma
        # linearização do flash e um segundo nível de AD por cima não compila (memória)
        f = f_point(u, x, m)
        fP = f_point(u + jnp.array([dP, 0.0]), x, m)
        fT = f_point(u + jnp.array([0.0, dT]), x, m)
        J = jnp.stack([(fP - f) / dP, (fT - f) / dT], axis=1)
        return f, J

    return f_and_J, float(case.pipe.L), case


def _profile_fn(activation, norm, L):
    """y(θ, feat) em (Pa, K) e dy/dx para feat = (P_in, T_in, m, s); `norm` = (xm, xs, ym, ys)."""
    import jax
    import jax.numpy as jnp

    xm, xs, ym, ys = (jnp.asarray(v) for v in norm)

    def y_of(params, feat):
        g = forward(params, ((feat - xm) / xs)[None, :], activation)[0] * ys + ym  # queda por unidade de s (bar, K)
        s = feat[3]
        return jnp.array([(feat[0] - s * g[0]) * 1e5, feat[1] - s * g[1]])

    def y_and_dydx(params, feat):
        def along_s(s):
            return y_of(params, feat.at[3].set(s))

        y, dy_ds = jax.jvp(along_s, (feat[3],), (jnp.ones(()),))
        return y, dy_ds / L

    return y_of, y_and_dydx


def network_values(params, feats_c, activation, norm, L):
    """u (n, 2) em (Pa, K) e du/dx (n, 2) da rede nos pontos de colocação (n, 4)."""
    import jax

    _, y_and_dydx = _profile_fn(activation, norm, L)
    return jax.vmap(lambda feat: y_and_dydx(params, feat))(feats_c)


def physics_values(f_and_J, u, feats_c, L):
    """f (n, 2) e ∂f/∂u (n, 2, 2) do simulador nos estados `u` dos pontos de colocação."""
    import jax

    return jax.vmap(lambda uu, feat: f_and_J(uu, feat[3] * L, feat[2]))(u, feats_c)


def physics_residual(params, feats_c, f_and_J, activation, norm, L, scale):
    """Resíduo escalado (n, 2) = (du/dx − f)/escala nos pontos de colocação; só valores."""
    import jax.numpy as jnp

    u, dudx = network_values(params, feats_c, activation, norm, L)
    f, _ = physics_values(f_and_J, u, feats_c, L)
    return (dudx - f) / jnp.asarray(scale)


def residual_scale(d: dict) -> np.ndarray:
    """Escala típica de |du/dx| (Pa/m, K/m) nos perfis gravados: mediana das diferenças finitas."""
    L = float(d["x_out"][-1])
    dx = np.diff(d["x_out"])
    prof = d["profiles"][d["ok"]]
    dudx = np.diff(prof, axis=1) / dx[None, :, None]
    sc = np.median(np.abs(dudx).reshape(-1, 2), axis=0)
    return np.where(sc > 0, sc, 1.0)


def train_pinn(
    dataset_path,
    case_path,
    hp: Hyperparameters = Hyperparameters(neurons=32, layers=2, activation="tanh", learning_rate=0.01, epochs=1000),
    n_train=None,
    n_collocation=150,
    lambda_phys=1.0,
    seed=0,
    split_seed=0,
    collocation_seed=1,
    train_fraction=0.7,
    init_from: ProfileMLP | None = None,
    progress=True,
):
    """Treina o substituto de perfil com perda = EQM dos dados (perfis de treino, alvos padronizados) +
    λ · EQM do resíduo escalado nos pontos de colocação. `n_train` limita os perfis de treino;
    `init_from` parte dos pesos de um modelo só de dados (mesma arquitetura e padronização)."""
    import jax
    import jax.numpy as jnp

    d = load_dataset(dataset_path)
    f_and_J, L, case = make_physics(case_path)
    ok = np.where(d["ok"])[0]
    tr, te = split_indices(len(ok), train_fraction, seed=split_seed)
    tr, te = ok[tr], ok[te]
    if n_train is not None:
        tr = tr[:n_train]
    feats_tr, tgt_tr, _, _ = profile_rows(d, tr)
    if init_from is not None:
        norm = (init_from.mlp.x_mean, init_from.mlp.x_std, init_from.mlp.y_mean, init_from.mlp.y_std)
        params0 = [(jnp.asarray(W), jnp.asarray(b)) for W, b in init_from.mlp.params]
        hp = Hyperparameters(**{**init_from.mlp.hp.__dict__, "epochs": hp.epochs, "learning_rate": hp.learning_rate})
    else:
        norm = _standardize(feats_tr, tgt_tr)
        params0 = init_params(jax.random.PRNGKey(seed), hp.sizes(4, 2))
    xm, xs, ym, ys = norm
    Xn, Yn = jnp.asarray((feats_tr - xm) / xs), jnp.asarray((tgt_tr - ym) / ys)
    r = d["meta"]["ranges"]
    bounds = [r["P_in_bar"], r["T_in_K"], r["m_kg_s"], [1.0 / n_collocation, 1.0]]
    feats_c = jnp.asarray(latin_hypercube(n_collocation, bounds, seed=collocation_seed))
    scale = residual_scale(d)
    act = hp.activation

    sc2 = jnp.asarray(scale) ** 2
    lr, b1, b2, eps = hp.learning_rate, hp.beta1, hp.beta2, hp.eps

    def data_loss(params):
        return jnp.mean((forward(params, Xn, act) - Yn) ** 2)

    def phys_surrogate(params, r0, J0):
        """Escalar cujo gradiente em θ é o de mean((r/escala)²) com r = du/dx − f(u): f e ∂f/∂u entram
        como valores (r0, J0), e só a rede é diferenciada (modo inverso pelo flash não compila)."""
        u, dudx = network_values(params, feats_c, act, norm, L)
        term = dudx - jnp.einsum("nij,nj->ni", J0, u)
        return 2.0 * jnp.mean(jnp.sum(r0 / sc2 * term, axis=1))

    @jax.jit
    def step(carry, t):
        params, m, v = carry
        u, dudx = network_values(params, feats_c, act, norm, L)
        f, J = physics_values(f_and_J, u, feats_c, L)
        r0 = jax.lax.stop_gradient(dudx - f)
        J0 = jax.lax.stop_gradient(J)
        phys = jnp.mean(r0**2 / sc2)
        data, g_data = jax.value_and_grad(data_loss)(params)
        g_phys = jax.grad(phys_surrogate)(params, r0, J0)
        g = jax.tree_util.tree_map(lambda a, b: a + lambda_phys * b, g_data, g_phys)
        m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_**2, v, g)
        mh = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1**t), m)
        vh = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2**t), v)
        params = jax.tree_util.tree_map(lambda p, mh_, vh_: p - lr * mh_ / (jnp.sqrt(vh_) + eps), params, mh, vh)
        return (params, m, v), jnp.array([data, phys])

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params0)
    carry = (params0, zeros, zeros)
    hist = np.zeros((hp.epochs, 2))
    t0 = time.time()
    for t in range(1, hp.epochs + 1):
        carry, h = step(carry, float(t))
        hist[t - 1] = np.asarray(h)
        if progress and (t == 1 or t % 200 == 0):
            print(f"  época {t}: EQM dados {hist[t-1, 0]:.3e}, resíduo {hist[t-1, 1]:.3e} ({time.time() - t0:.0f} s)")
    params = [(np.asarray(W), np.asarray(b)) for W, b in carry[0]]
    mlp = MLP(
        hp=hp, params=params, x_mean=np.asarray(xm), x_std=np.asarray(xs), y_mean=np.asarray(ym), y_std=np.asarray(ys)
    )
    mlp.input_names, mlp.output_names = PROFILE_INPUTS, PROFILE_OUTPUTS
    mlp.meta = {
        "dataset": str(dataset_path),
        "case": str(case_path),
        "segment": d["meta"].get("segment"),
        "pinn": {
            "n_collocation": n_collocation,
            "lambda_phys": lambda_phys,
            "scale": scale.tolist(),
            "collocation_seed": collocation_seed,
            "init_from": init_from is not None,
        },
        "seed": seed,
        "n_train_profiles": int(len(tr)),
        "n_test_profiles": int(len(te)),
        "cpu": time.time() - t0,
    }
    model = ProfileMLP(mlp=mlp, L=L)
    mlp.meta["test_profile"] = evaluate_profile(model, d, te)
    mlp.meta["train_profile"] = evaluate_profile(model, d, tr)
    return model, hist, (tr, te), feats_c


def residual_rms(model: ProfileMLP, case_path, feats_c, scale=None, dataset=None) -> float:
    """RMS do resíduo físico escalado de um modelo de perfil (PINN ou só de dados) nos pontos `feats_c`."""
    import jax.numpy as jnp

    f_and_J, L, _ = make_physics(case_path)
    sc = scale if scale is not None else residual_scale(load_dataset(dataset))
    params = [(jnp.asarray(W), jnp.asarray(b)) for W, b in model.mlp.params]
    norm = (model.mlp.x_mean, model.mlp.x_std, model.mlp.y_mean, model.mlp.y_std)
    r = physics_residual(params, jnp.asarray(feats_c), f_and_J, model.mlp.hp.activation, norm, L, sc)
    return float(jnp.sqrt(jnp.mean(r**2)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("case")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neurons", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--activation", default="tanh", choices=("relu", "sigmoid", "tanh"))
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--n-collocation", type=int, default=150)
    ap.add_argument("--lambda-phys", type=float, default=1.0)
    ap.add_argument("--init-from", help="modelo de perfil só de dados (JSON) para arranque")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--history")
    a = ap.parse_args(argv)
    hp = Hyperparameters(
        neurons=a.neurons, layers=a.layers, activation=a.activation, learning_rate=a.lr, epochs=a.epochs
    )
    init = ProfileMLP.load(a.init_from) if a.init_from else None
    model, hist, (tr, te), feats_c = train_pinn(
        a.dataset,
        a.case,
        hp,
        n_train=a.n_train,
        n_collocation=a.n_collocation,
        lambda_phys=a.lambda_phys,
        seed=a.seed,
        split_seed=a.split_seed,
        init_from=init,
    )
    model.save(a.out)
    if a.history:
        np.savez(a.history, history=hist, train_idx=tr, test_idx=te, collocation=np.asarray(feats_c))
    t = model.mlp.meta["test_profile"]
    print(
        f"gravado {a.out}: perfis de teste {t['n_profiles']}: RMSE P {t['rmse'][0]:.3f} bar, T {t['rmse'][1]:.3f} K; "
        f"máx {t['max_abs_error'][0]:.2f} bar, {t['max_abs_error'][1]:.2f} K; resíduo final {hist[-1, 1]:.3e}"
    )


if __name__ == "__main__":
    main()
