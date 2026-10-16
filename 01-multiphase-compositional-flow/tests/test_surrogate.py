"""Modelo substituto (`surrogate/`): amostragem, rede e treino, divisão, predição encadeada, extração
dos Apêndices B e C e casos gerados. A geração da massa de dados pelo simulador é noturna."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cmfsim.io import validate_case
from dissertation.extract import parse_grid, parse_samples
from generators.surrogate import segment_case, system_case
from surrogate import MLP, Hyperparameters, evaluate, r2_score, split_indices, train
from surrogate.dataset import INPUT_COLUMNS, thesis_ranges
from surrogate.grid import thesis_grid
from surrogate.predict import chain_predict
from surrogate.sampling import latin_hypercube

ROOT = Path(__file__).resolve().parents[1]


def test_latin_hypercube_stratified():
    b = [[145.0, 185.0], [333.15, 373.15], [2.0, 25.0]]
    for crit in ("random", "center", "maximin"):
        x = latin_hypercube(50, b, seed=1, criterion=crit, n_candidates=5)
        assert x.shape == (50, 3)
        for j in range(3):
            strata = np.floor((x[:, j] - b[j][0]) / (b[j][1] - b[j][0]) * 50).astype(int)
            assert sorted(np.clip(strata, 0, 49)) == list(range(50))  # cada estrato visitado uma vez
    assert np.allclose(latin_hypercube(10, b, seed=3), latin_hypercube(10, b, seed=3))


def test_thesis_ranges_and_grid():
    r = thesis_ranges("well")
    assert r["P_in_bar"] == [145.0, 185.0] and r["T_in_K"] == [333.15, 373.15] and r["m_kg_s"] == [2.0, 25.0]
    g = thesis_grid()
    assert len(g) == 135
    # ordem do Apêndice C: caso 42 = 10 neurónios, 0,01, tanh, 1 camada; caso 90 = 20, 0,01, relu, 1
    assert (g[42].neurons, g[42].learning_rate, g[42].activation, g[42].layers) == (10, 0.01, "tanh", 1)
    assert (g[90].neurons, g[90].learning_rate, g[90].activation, g[90].layers) == (20, 0.01, "relu", 1)


def test_split_indices():
    tr, te = split_indices(299, 0.7, seed=0)
    assert len(tr) == 209 and len(te) == 90 and not set(tr) & set(te)


def test_train_fits_smooth_function_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform([145.0, 333.15, 2.0], [185.0, 373.15, 25.0], (300, 3))
    Y = np.stack([0.5 * X[:, 0] - 0.4 * X[:, 2] ** 1.5, X[:, 1] - 0.002 * (X[:, 1] - 300) * X[:, 2]], axis=1)
    tr, te = split_indices(300, 0.7, seed=0)
    hp = Hyperparameters(neurons=20, layers=1, activation="relu", learning_rate=0.01, epochs=1500)
    model, hist = train(X[tr], Y[tr], hp, seed=0, test=(X[te], Y[te]))
    assert hist.shape == (1500, 2) and hist[-1, 0] < hist[0, 0]
    m = evaluate(model, X[te], Y[te])
    assert min(m["r2"]) > 0.99, m
    assert model.n_parameters == 3 * 20 + 20 + 20 * 2 + 2
    path = model.save(tmp_path / "m.json")
    again = MLP.load(path)
    assert np.allclose(again.predict(X[te]), model.predict(X[te]))
    assert again.hp == hp and again.input_names == model.input_names


def test_r2_score():
    y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    assert np.allclose(r2_score(y, y), [1.0, 1.0])
    assert np.allclose(r2_score(y, np.full_like(y, y.mean(axis=0))), [0.0, 0.0])


def test_chain_predict_uses_previous_outlet():
    hp = Hyperparameters(neurons=1, layers=1, activation="identity", epochs=1)
    # rede «identidade» com pesos escolhidos: P_out = P_in − 10, T_out = T_in − 5 (sem padronização)
    W1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    W2 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    one = np.ones(3)
    m = MLP(hp, [(W1, np.zeros(3)), (W2, np.array([-10.0, -5.0]))], np.zeros(3), one, np.zeros(2), np.ones(2))
    Y = chain_predict([m, m, m], [100.0, 90.0], [350.0, 340.0], [10.0, 12.0])
    assert Y.shape == (3, 2, 2)
    assert np.allclose(Y[-1], [[70.0, 335.0], [60.0, 325.0]])


def test_parse_appendix_text():
    b = "Caso\n0 \n156,4 \n335,7 \n15,8 \n71,9 \n328,2 \n1 \n158,3 \n338,8 \n4,8 \n75,6 \n325,9 \n3 \n1,0 \n2,0 \n3,0 \n4,0 \n5,0 \n"
    rows = parse_samples(b, n_expected=4, min_rows=3)
    assert [r[0] for r in rows] == [0, 1, 3] and rows[0][1:] == (156.4, 335.7, 15.8, 71.9, 328.2)
    c = "0 \n5 \n0,1 \n relu \n1 \n0,96 \n0,84 \n1,81 \nFALSE \n1 \n5 \n0,1 \nrelu \n2 \n-0,29 \n0,00 \n-0,30 \nFALSE \n"
    rows = parse_grid(c, n_expected=2, min_rows=2)
    assert rows[1] == (1, 5, 0.1, "relu", 2, -0.29, 0.0, -0.3, False)


def test_extracted_appendices_present():
    for seg in ("well", "line", "riser"):
        b = (ROOT / "dissertation" / "data" / f"appendix_b_{seg}.csv").read_text().splitlines()
        c = (ROOT / "dissertation" / "data" / f"appendix_c_{seg}.csv").read_text().splitlines()
        assert b[0].startswith("case,P_in_bar,T_in_K,m_kg_s,P_out_bar,T_out_K") and len(b) - 1 >= 295
        assert c[0].startswith("case,neurons,learning_rate,activation,layers,r2_P,r2_T") and len(c) - 1 == 135


def test_legacy_data_match_pdf():
    """Apêndice B em precisão completa e os 159 estados, do código original (legacy/issues L06.2): mesmos casos
    e mesmos valores do PDF a menos do arredondamento a 0,1; estados em kelvin, 10 K acima da Figura 13."""
    data = ROOT / "dissertation" / "data"
    for seg in ("well", "line", "riser"):
        pdf = np.loadtxt(data / f"appendix_b_{seg}.csv", delimiter=",", skiprows=1)
        full = np.loadtxt(data / f"appendix_b_{seg}_full.csv", delimiter=",", skiprows=1)
        assert np.array_equal(pdf[:, 0], full[:, 0])
        assert np.abs(pdf[:, 1:] - full[:, 1:]).max() <= 0.0501
    st = np.loadtxt(data / "hysys_states.csv", delimiter=",", skiprows=1)
    assert st.shape == (159, 3) and 147.0 < st[:, 2].min() < 148.0 and 526.0 < st[:, 2].max() < 527.0


def test_generated_surrogate_cases_validate():
    for seg in ("well", "line", "riser"):
        d = segment_case(seg)
        assert validate_case(d) == []
        assert [c["id"] for c in d["fluid"]["components"]][:3] == ["CO2", "N2", "H2S"]  # Tabela 6
        assert d["meta"]["segment"] == seg and d["pipe"]["profile"][0]["U"] in (7.0, 5.0)
    assert validate_case(system_case()) == []


@pytest.mark.nightly
def test_dataset_generation_small(tmp_path):
    from surrogate.dataset import generate, load_dataset

    case = tmp_path / "riser.json"
    case.write_text(json.dumps(segment_case("riser")))
    out, meta = generate(case, tmp_path / "riser.npz", n=4, seed=0, segment="riser")
    d = load_dataset(out)
    assert d["X"].shape == (4, 3) and d["Y"].shape == (4, 2) and d["ok"].all()
    assert meta["input_columns"] == INPUT_COLUMNS
    assert np.all(d["Y"][:, 0] < d["X"][:, 0])  # o riser perde pressão
    assert (tmp_path / "riser.csv").exists()


def _synthetic_profile_dataset(n=12, n_out=11, L=1500.0):
    """Massa de dados sintética no formato de `surrogate.dataset`: perfis lineares em s."""
    rng = np.random.default_rng(0)
    X = rng.uniform([145.0, 333.15, 2.0], [185.0, 373.15, 25.0], (n, 3))
    s = np.linspace(0.0, 1.0, n_out)
    P = (X[:, 0:1] - s[None, :] * (0.4 * X[:, 0:1] + X[:, 2:3])) * 1e5
    T = X[:, 1:2] - s[None, :] * 0.1 * (X[:, 1:2] - 300.0)
    return {
        "X": X,
        "x_out": s * L,
        "profiles": np.stack([P, T], axis=2),
        "ok": np.ones(n, bool),
        "meta": {"segment": "well"},
    }


def test_profile_rows_and_inlet_constraint():
    from surrogate.profile import PROFILE_INPUTS, PROFILE_OUTPUTS, ProfileMLP, profile_rows

    d = _synthetic_profile_dataset()
    feats, target, y, y_in = profile_rows(d, np.arange(12))
    assert feats.shape == (12 * 10, 4) and np.all(feats[:, 3] > 0)  # s = 0 fica de fora
    # alvo = queda por unidade de s: para o perfil linear é constante em s
    assert np.allclose(target[:, 0], 0.4 * feats[:, 0] + feats[:, 2]) and np.allclose(
        target[:, 1], 0.1 * (feats[:, 1] - 300.0)
    )
    hp = Hyperparameters(neurons=8, layers=1, activation="tanh", epochs=1)
    import jax

    from surrogate.model import init_params

    params = [(np.asarray(W), np.asarray(b)) for W, b in init_params(jax.random.PRNGKey(0), hp.sizes(4, 2))]
    mlp = MLP(hp, params, np.zeros(4), np.ones(4), np.zeros(2), np.ones(2), PROFILE_INPUTS, PROFILE_OUTPUTS)
    model = ProfileMLP(mlp=mlp, L=1500.0)
    prof = model.predict(160.0, 350.0, 10.0, [0.0, 0.5, 1.0])
    assert np.allclose(prof[0], [160.0, 350.0])  # condição de entrada exata por construção


def test_profile_train_and_roundtrip(tmp_path):
    from surrogate.profile import ProfileMLP, train_profile

    d = _synthetic_profile_dataset(n=40)
    path = tmp_path / "synthetic.npz"
    np.savez(
        path,
        format="cmfsim-surrogate-dataset/1",
        X=d["X"],
        Y=np.stack([d["profiles"][:, -1, 0] / 1e5, d["profiles"][:, -1, 1]], axis=1),
        ok=d["ok"],
        x_out=d["x_out"],
        profiles=d["profiles"],
        n_accepted=np.ones(40),
        n_rejected=np.zeros(40),
        meta_json=json.dumps(
            {
                "segment": "well",
                "ranges": {},
                "input_columns": list(INPUT_COLUMNS),
                "output_columns": ["P_out_bar", "T_out_K"],
            }
        ),
        case_json="{}",
        provenance_json="{}",
    )
    hp = Hyperparameters(neurons=16, layers=1, activation="tanh", learning_rate=0.01, epochs=1500)
    model, hist, (tr, te) = train_profile(path, hp, seed=0)
    t = model.mlp.meta["test_profile"]
    assert t["rmse"][0] < 0.3 and t["rmse"][1] < 0.3, t  # perfil linear aprendido a < 0,3 bar / K
    again = ProfileMLP.load(model.save(tmp_path / "p.json"))
    assert np.allclose(again.predict(160.0, 350.0, 10.0, [0.2, 1.0]), model.predict(160.0, 350.0, 10.0, [0.2, 1.0]))


@pytest.mark.nightly
def test_pinn_residual_matches_recorded_profile():
    """O resíduo físico do simulador nos perfis gravados é ~0: f(x, u) coincide com du/dx do perfil."""
    import jax.numpy as jnp

    from surrogate.dataset import load_dataset
    from surrogate.pinn import make_physics, residual_scale

    path = ROOT / "runs" / "surrogate" / "well_profile.npz"
    if not path.exists():
        pytest.skip("massa de dados de perfis não gerada")
    d = load_dataset(path)
    f_and_J, L, _ = make_physics(ROOT / "cases" / "surrogate" / "well.json")
    sc = residual_scale(d)
    prof, x = d["profiles"][0], d["x_out"]
    for j in (0, 10, 25):
        u = jnp.asarray(prof[j])
        f = np.asarray(f_and_J(u, float(x[j]), float(d["X"][0, 2]))[0])
        fd = (prof[j + 1] - prof[j]) / (x[j + 1] - x[j])
        assert np.all(np.abs(f - fd) / sc < 0.05), (j, f, fd)


def test_validation_splits_and_metrics():
    from surrogate.validation import kfold_indices, metrics, profile_split

    folds = kfold_indices(100, 5, seed=1)
    assert [len(te) for _, te in folds] == [20] * 5
    assert all(not set(tr) & set(te) and len(tr) + len(te) == 100 for tr, te in folds)
    d = _synthetic_profile_dataset(n=30, n_out=11)
    for mode in ("profile", "position", "block"):
        (ftr, ttr, ytr, _), (fte, tte, yte, _), info = profile_split(d, mode, seed=0, n_train=10)
        assert len(ftr) == len(ytr) and len(fte) == len(yte) and len(ftr) > 0 and len(fte) > 0
        if mode == "position":
            # as posições de teste nunca aparecem no treino
            assert not set(np.round(ftr[:, 3], 6)) & set(np.round(fte[:, 3], 6))
            assert info["train_profiles"] == 10
        if mode == "block":
            assert fte[:, 0].min() >= info["P_in_cut_bar"] and ftr[:, 0].max() < info["P_in_cut_bar"]
    m = metrics(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.5, 2.0], [3.5, 4.0]]))
    assert np.allclose(m["bias"], [0.5, 0.0]) and np.allclose(m["rmse"], [0.5, 0.0])
