"""Matriz de métodos (issue 011): 2 modelos × 3 solvers × 2 backends resolvem as mesmas equações.
Com bolhas a C_0 = 1 no horizontal o modelo mecanicista é, por construção, o de não escorregamento,
pelo que os seis caminhos têm de coincidir a menos do erro de discretização; os volumes finitos
fecham os balanços globais a precisão de máquina; numpy e JAX coincidem."""

import json

import numpy as np
import pytest

from cmfsim import Case, simulate
from cmfsim.simulation import balances
from generators.condensate import base_case

N_NODES = 16


def _case(model, solver, backend, n=N_NODES, dx_out=500.0, **model_over):
    d = base_case(dx_out=dx_out)
    d["pipe"]["profile"][0].update(U=6.0, T_amb=288.15)
    d["solver"].update(rtol=1e-8, atol=1e-8, first_step=10.0, backend=backend)
    if model == "mechanistic":
        d["model"] = {
            "type": "mechanistic",
            "mass_split": "algebraic",
            "regime": {"regime": "bubble", "bubble_C0": 1.0},
        }
    d["model"].update(model_over)
    if solver != "march":
        d["solver"] = {
            "type": solver,
            "n": n,
            "initial_guess": "march_noslip",
            "rtol": 1e-8,
            "atol": 1e-8,
            "first_step": 10.0,
            "backend": backend,
            "newton": {"max_iter": 40, "tol": 1e-10},
        }
    return Case.from_dict(json.loads(json.dumps(d)))


@pytest.fixture(scope="module")
def reference():
    # saída densa (50 m) para que a interpolação linear nos nós dos outros métodos não pese
    return simulate(_case("noslip", "march", "numpy", dx_out=50.0), backend="numpy", progress=False)


@pytest.mark.parametrize("model", ["noslip", "mechanistic"])
@pytest.mark.parametrize("solver", ["march", "fd", "fv"])
def test_matrix_numpy_agrees_with_march(reference, model, solver):
    r = simulate(_case(model, solver, "numpy"), backend="numpy", progress=False)
    if solver != "march":
        assert r.info["converged"] and r.info["iterations"] <= 6
    Pi = np.interp(r.x, reference.x, reference.P)
    Ti = np.interp(r.x, reference.x, reference.T)
    assert np.max(np.abs(r.P - Pi)) < 15.0, (model, solver)  # Pa, erro de discretização com 17 nós
    assert np.max(np.abs(r.T - Ti)) < 0.02, (model, solver)
    b = r.info["balances"]
    assert b["mass_rel"] < 1e-12
    if solver == "fv":
        # conservativo por construção: os balanços fecham a precisão de máquina na própria malha
        assert abs(b["momentum_rel"]) < 1e-9 and abs(b["energy_rel"]) < 1e-9, b
    else:
        assert abs(b["momentum_rel"]) < 1e-4 and abs(b["energy_rel"]) < 1e-4, b


@pytest.mark.parametrize("model,solver", [("noslip", "fv"), ("mechanistic", "fv")])
def test_matrix_jax_parity(model, solver):
    # os volumes finitos são o caminho nodal de produção; a paridade da marcha mecanicista
    # (numpy com derivadas por diferenças × JAX por AD) está no lote solver_matrix / notebook 5.5
    pytest.importorskip("jax")
    a = simulate(_case(model, solver, "numpy"), backend="numpy", progress=False)
    b = simulate(_case(model, solver, "jax"), backend="jax", progress=False)
    assert np.max(np.abs(a.P - b.P)) < 1.0 and np.max(np.abs(a.T - b.T)) < 5e-3  # numpy: Jacobiana por diferenças
    assert b.info["backend"] == "jax"


def test_fv_refinement_is_second_order(reference):
    errs = []
    for n in (8, 16, 32):
        r = simulate(_case("noslip", "fv", "numpy", n=n), backend="numpy", progress=False)
        Ti = np.interp(r.x, reference.x, reference.T)
        errs.append(np.max(np.abs(r.T - Ti)))
    assert errs[0] > errs[1] > errs[2]
    assert errs[0] / errs[2] > 6.0  # ~h^2 daria 16; a marcha de referência tem o seu próprio erro


def test_balances_function_is_consistent(reference):
    b = balances(Case.from_dict(reference.case_dict), reference.x, reference.P, reference.T)
    assert set(b) == {"momentum_Pa", "momentum_rel", "energy_J_kg", "energy_rel", "mass_rel"}
    assert abs(b["momentum_rel"]) < 1e-4
