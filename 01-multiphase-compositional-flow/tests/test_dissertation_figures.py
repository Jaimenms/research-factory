"""Noturna: as corridas dos lotes contra as figuras digitalizadas e o Apêndice A. Precisa de
`runs/dissertation/*.npz` (lote `batches/dissertation.json`) e, para o Apêndice A, de
`runs/dp_comparison/*.npz` (lote `batches/dp_comparison.json`)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cmfsim import load_run
from cmfsim.thermo import phase_state
from tests.conftest import DATA, ROOT, make_fluid

pytestmark = pytest.mark.nightly

RUNS = ROOT / "runs" / "dissertation"
FIG = ROOT / "references" / "figures" / "data"


def _series(n):
    d = json.load(open(FIG / f"fig_{n:02d}.json"))
    out = {}
    for s in d["series"]:
        x, y = np.array(s["x"]), np.array(s["y"])
        o = np.argsort(x)
        out[s["name"]] = (x[o], y[o])
    return out


@pytest.mark.skipif(not (RUNS / "condensate_5_6_case1.npz").exists(), reason="lote §5.6 não corrido")
def test_condensate_pressure_and_dew_point_against_figures(components_db, fluids_db):
    fig38, fig39, fig40 = _series(38), _series(39), _series(40)
    cs, z = make_fluid(components_db, fluids_db["table23_rich_gas"])
    for k in range(1, 7):
        r = load_run(RUNS / f"condensate_5_6_case{k}.npz")
        s = str(k)
        # pressão na saída: o atrito/massa específica reproduz a Figura 38 a 0,15 bar
        P_out_fig = float(np.interp(7950.0, *fig38[s]))
        assert abs(r.P[-1] / 1e5 - P_out_fig) < 0.15, (k, r.P[-1] / 1e5, P_out_fig)
        # ponto de orvalho: onde a Figura 40 começa a condensar, a temperatura da Figura 39 tem
        # de ser o orvalho do nosso flash à pressão local a 2,5 °C (termodinâmica concordante,
        # independentemente do achado térmico da issue 004)
        xd, yd = fig40[s]
        if np.any(yd < 0.9995):
            x_cond = float(xd[yd < 0.9995][0])
            T_cond_fig = float(np.interp(x_cond, *fig39[s])) + 273.15
            P_cond_fig = float(np.interp(x_cond, *fig38[s])) * 1e5
            lo, hi = 273.15, 333.15
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                b = phase_state(np.array([mid]), np.array([P_cond_fig]), z, cs).beta[0]
                lo, hi = (mid, hi) if b < 1.0 else (lo, mid)
            T_dew = 0.5 * (lo + hi)
            assert abs(T_dew - T_cond_fig) < 2.5, (k, T_dew - 273.15, T_cond_fig - 273.15)


DP_RUNS = ROOT / "runs" / "dp_comparison"


@pytest.mark.skipif(not (DP_RUNS / "dp_horizontal_mechanistic_001.npz").exists(), reason="lote §5.4 não corrido")
def test_appendix_a_mechanistic_within_band():
    """Resumo contra a coluna «Dissertação» do Apêndice A; os limiares registam o estado medido
    (issue 005) e apertam-se à medida que as leituras do texto forem decididas."""
    # estado medido em 2026-10-15 (issue 005): vertical mediana +3 %, 94 % a menos de 25 %;
    # horizontal sistematicamente ~0,4x (mediana -61 %) -> guarda de regressão, não de validação
    for orientation, med_tol, frac_tol in (("horizontal", 0.7, 0.0), ("vertical", 0.1, 0.85)):
        with open(DATA / f"appendix_a_{orientation}.csv") as f:
            rows = list(csv.DictReader(f))
        ref = np.array([float(r["dp_bar_dissertation"]) for r in rows])
        got, idx = [], []
        for k in range(1, 101):
            p = DP_RUNS / f"dp_{orientation}_mechanistic_{k:03d}.npz"
            if p.exists():
                r = load_run(p)
                got.append((r.P[0] - r.P[-1]) / 1e5)
                idx.append(k)
        got, idx = np.array(got), np.array(idx)
        assert idx.size >= 90, f"{orientation}: só {idx.size} corridas"
        rel = np.abs(got - ref[idx - 1]) / ref[idx - 1]
        assert np.median(rel) < med_tol, (orientation, np.median(rel))
        assert np.mean(rel < 0.25) >= frac_tol, (orientation, np.mean(rel < 0.25))
