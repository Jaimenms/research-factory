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
