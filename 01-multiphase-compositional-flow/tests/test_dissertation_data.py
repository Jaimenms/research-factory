"""Os dados transcritos e extraídos da dissertação são coerentes."""

import csv
import json

import numpy as np

from tests.conftest import DATA


def _rows(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def test_appendix_a_grid():
    for orientation, last in (("horizontal", "dp_bar_gregory"), ("vertical", "dp_bar_aziz")):
        rows = _rows(f"appendix_a_{orientation}.csv")
        assert len(rows) == 100
        grid = [(D, P, m) for D in (4, 6, 8, 10) for P in (20, 55, 90, 125, 160) for m in (2, 4, 6, 8, 10)]
        assert [(int(r["D_in"]), int(r["P_in_bar"]), int(r["m_kg_s"])) for r in rows] == grid
        assert [int(r["case"]) for r in rows] == list(range(1, 101))
        dp = np.array([float(r["dp_bar_dissertation"]) for r in rows])
        assert np.all(dp > 0)
        assert last in rows[0]


def test_appendix_a_spot_values():
    h = _rows("appendix_a_horizontal.csv")
    assert float(h[0]["dp_bar_dissertation"]) == 0.0537  # caso 1: 5,37E-02
    assert float(h[99]["dp_bar_dissertation"]) == 0.0065  # caso 100: 6,50E-03
    v = _rows("appendix_a_vertical.csv")
    assert float(v[0]["dp_bar_dissertation"]) == 2.68
    assert float(v[99]["dp_bar_dissertation"]) == 6.10


def test_compositions_sum_to_one(fluids_db):
    for name, fl in fluids_db.items():
        assert abs(sum(fl.values()) - 1.0) < 1e-9, name


def test_components_complete(components_db, fluids_db):
    for fl in fluids_db.values():
        for k in fl:
            c = components_db[k]
            assert c["Pc"] > 1e5 and c["Tc"] > 50 and c["MW"] > 1 and len(c["cp"]) == 4
