"""Casos em JSON: schema, herança `$extends`, overrides por caminho, construção do `Case`."""

import json
from pathlib import Path

import numpy as np
import pytest

from cmfsim.io import Case, case_from_json, load_json, validate_case
from generators.condensate import base_case, variant


def test_generated_cases_validate(tmp_path):
    d = base_case()
    assert validate_case(d) == []
    for n, T, U in [(1, 60.0, 1.0), (6, 80.0, 6.0)]:
        v = variant(n, T, U)
        (tmp_path / "condensate_5_6_base.json").write_text(json.dumps(d))
        (tmp_path / f"c{n}.json").write_text(json.dumps(v))
        full = load_json(tmp_path / f"c{n}.json")
        assert validate_case(full) == []
        assert full["boundary"]["inlet"]["T"] == 273.15 + T and full["pipe"]["profile"][0]["U"] == U
        assert full["fluid"] == d["fluid"] and full["meta"]["extends"]


def test_invalid_cases_are_rejected():
    d = base_case()
    bad = json.loads(json.dumps(d))
    bad["model"]["type"] = "magic"
    assert validate_case(bad)
    bad = json.loads(json.dumps(d))
    bad["pipe"]["profile"][0]["foo"] = 1
    assert validate_case(bad)
    bad = json.loads(json.dumps(d))
    del bad["boundary"]
    assert validate_case(bad)


def test_overrides_and_case_object(tmp_path):
    (tmp_path / "base.json").write_text(json.dumps(base_case()))
    d = load_json(tmp_path / "base.json", ["solver.rtol=1e-5", "pipe.profile.0.U=2.5", "boundary.inlet.m=10"])
    case = Case.from_dict(d)
    assert case.rk45.rtol == 1e-5 and case.pipe.U[0] == 2.5 and case.inlet["m"] == 10
    assert case.cs.n == 5 and abs(case.z.sum() - 1.0) < 1e-12
    assert case.pipe.L == 8000.0 and case.pipe.n_segments == 1
    assert case.model_type == "noslip" and case.solver_type == "march"


def test_pipe_profile_helpers():
    from cmfsim.pipe import PipeProfile

    pipe = PipeProfile.from_json(
        {
            "profile": [
                {"x": 0.0, "D": 0.127, "roughness": 2e-4, "angle_deg": 90.0, "U": 7.0, "T_amb": 313.15},
                {"x": 1500.0, "D": 0.1524, "roughness": 6e-4, "angle_deg": 0.0, "U": 5.0, "T_amb": 285.15},
                {"x": 7500.0, "angle_deg": 90.0},
                {"x": 7800.0},
            ]
        }
    )
    assert pipe.n_segments == 3 and pipe.L == 7800.0
    assert pipe.D.tolist() == [0.127, 0.1524, 0.1524]  # herda do troço anterior
    assert np.allclose(pipe.elevation([0.0, 1500.0, 4500.0, 7800.0]), [0.0, 1500.0, 1500.0, 1800.0])
    g = pipe.output_grid(1000.0)
    assert 1500.0 in g and 7500.0 in g and g[-1] == 7800.0 and g[0] == 0.0
    assert pipe.interval_segments(g).tolist() == pipe.segment_of(0.5 * (g[:-1] + g[1:])).tolist()
    assert pipe.segment_of([0.0, 1500.0, 7800.0]).tolist() == [0, 1, 2]
    round_trip = PipeProfile.from_json(pipe.to_json())
    assert np.allclose(round_trip.theta, pipe.theta) and np.allclose(round_trip.x_break, pipe.x_break)
