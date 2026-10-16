"""Massa de dados de treino e teste: entradas (P_in, T_in, m) amostradas nas faixas das Tabelas 27–29
(ou lidas de um CSV, p. ex. o Apêndice B) e saídas (P_out, T_out) do modelo de não escorregamento, em
lotes vetorizados (`march_many`: uma compilação, `vmap` sobre as amostras). Unidades da tabela como
na dissertação: bar, K, kg/s. Grava NPZ (com perfis, caso e proveniência) e CSV ao lado.

Uso: uv run python -m surrogate.dataset cases/surrogate/well.json --segment well --n 300 --seed 0 \\
         --out runs/surrogate/well.npz
     uv run python -m surrogate.dataset cases/surrogate/well.json --inputs dissertation/data/appendix_b_well_full.csv \\
         --out runs/surrogate/well_appendix_b.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .sampling import latin_hypercube

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "dissertation" / "data" / "surrogate_tables.json"
INPUT_COLUMNS = ("P_in_bar", "T_in_K", "m_kg_s")
OUTPUT_COLUMNS = ("P_out_bar", "T_out_K")
DATASET_FORMAT = "cmfsim-surrogate-dataset/1"


def thesis_ranges(segment: str) -> dict:
    """Faixas da Tabela 27/28/29 (`well`, `line`, `riser`) nas unidades da tabela de dados."""
    r = json.load(open(TABLES))["ranges"][segment]
    return {
        "P_in_bar": list(r["P_in_bar"]),
        "T_in_K": [r["T_in_C"][0] + 273.15, r["T_in_C"][1] + 273.15],
        "m_kg_s": list(r["m_kg_s"]),
    }


def sample_inputs(ranges: dict, n: int, seed=0, criterion="random") -> np.ndarray:
    bounds = [ranges[c] for c in INPUT_COLUMNS]
    return latin_hypercube(n, bounds, seed=seed, criterion=criterion)


def read_inputs_csv(path) -> np.ndarray:
    rows = list(csv.DictReader(open(path)))
    return np.array([[float(r[c]) for c in INPUT_COLUMNS] for r in rows], float)


def simulate_outputs(case, X: np.ndarray, chunk=100, backend="jax"):
    """Corre o simulador para cada linha de X (bar, K, kg/s); devolve dict com Y (n, 2), ok (n,), x_out,
    profiles (n, n_out, 2) em Pa e K, e info (passos aceites/rejeitados por amostra, tempo)."""
    from cmfsim.simulation import march_many

    X = np.asarray(X, float)
    n = len(X)
    profiles, acc, rej, t0 = [], [], [], time.time()
    x_out = None
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        var = {"P_in": X[a:b, 0] * 1e5, "T_in": X[a:b, 1], "m": X[a:b, 2]}
        x_out, U, info = march_many(case, var, backend=backend)
        profiles.append(U)
        acc.append(info["n_accepted"])
        rej.append(info["n_rejected"])
    U = np.concatenate(profiles)
    Y = np.stack([U[:, -1, 0] / 1e5, U[:, -1, 1]], axis=1)
    ok = np.isfinite(U).all(axis=(1, 2)) & (U[:, :, 0] > 0.0).all(axis=1) & (U[:, :, 1] > 0.0).all(axis=1)
    info = {
        "n_accepted": np.concatenate(acc),
        "n_rejected": np.concatenate(rej),
        "cpu": time.time() - t0,
        "backend": backend,
    }
    return {"Y": Y, "ok": ok, "x_out": np.asarray(x_out), "profiles": U, "info": info}


def generate(
    case_path,
    out,
    n=300,
    seed=0,
    segment=None,
    ranges=None,
    inputs_csv=None,
    criterion="random",
    chunk=100,
    overrides=None,
):
    """Gera e grava a massa de dados de um caso de trecho. Entradas: `ranges` (ou as da dissertação via
    `segment`) amostradas por hipercubo latino, ou lidas de `inputs_csv`. `overrides`: lista
    `caminho=valor` aplicada ao caso (p. ex. `outputs.dx_out=30` para perfis mais finos)."""
    from cmfsim.io.spec import Case, load_json
    from cmfsim.results import provenance

    case_dict = load_json(case_path, overrides)
    case = Case.from_dict(case_dict)
    if inputs_csv is not None:
        X = read_inputs_csv(inputs_csv)
        source = {"inputs": str(inputs_csv)}
        ranges = ranges or {c: [float(X[:, k].min()), float(X[:, k].max())] for k, c in enumerate(INPUT_COLUMNS)}
    else:
        ranges = ranges or thesis_ranges(segment or case_dict.get("meta", {}).get("segment"))
        X = sample_inputs(ranges, n, seed=seed, criterion=criterion)
        source = {"sampling": "latin_hypercube", "criterion": criterion, "seed": seed, "n": n}
    res = simulate_outputs(case, X, chunk=chunk, backend=case.backend if case.backend == "jax" else "jax")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "segment": segment or case_dict.get("meta", {}).get("segment"),
        "source": source,
        "ranges": ranges,
        "input_columns": INPUT_COLUMNS,
        "output_columns": OUTPUT_COLUMNS,
        "n_ok": int(res["ok"].sum()),
        "cpu": res["info"]["cpu"],
    }
    np.savez(
        out,
        format=DATASET_FORMAT,
        X=X,
        Y=res["Y"],
        ok=res["ok"],
        x_out=res["x_out"],
        profiles=res["profiles"],
        n_accepted=res["info"]["n_accepted"],
        n_rejected=res["info"]["n_rejected"],
        meta_json=json.dumps(meta, ensure_ascii=False),
        case_json=json.dumps(case_dict, ensure_ascii=False),
        provenance_json=json.dumps(provenance()),
    )
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("case",) + INPUT_COLUMNS + OUTPUT_COLUMNS + ("ok",))
        for k in range(len(X)):
            w.writerow((k, *[f"{v:.6g}" for v in X[k]], *[f"{v:.6g}" for v in res["Y"][k]], int(res["ok"][k])))
    return out, meta


def load_dataset(path) -> dict:
    d = np.load(path, allow_pickle=False)
    if str(d["format"]) != DATASET_FORMAT:
        raise ValueError(f"formato desconhecido: {d['format']}")
    out = {k: d[k] for k in ("X", "Y", "ok", "x_out", "profiles", "n_accepted", "n_rejected")}
    out["meta"] = json.loads(str(d["meta_json"]))
    out["case"] = json.loads(str(d["case_json"]))
    out["provenance"] = json.loads(str(d["provenance_json"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segment", choices=("well", "line", "riser"), help="faixas da Tabela 27/28/29")
    ap.add_argument("--inputs", help="CSV com colunas P_in_bar, T_in_K, m_kg_s (p. ex. o Apêndice B)")
    ap.add_argument("--criterion", default="random", choices=("random", "center", "maximin"))
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--set", action="append", default=[], help="override caminho=valor no caso (repetível)")
    a = ap.parse_args(argv)
    out, meta = generate(
        a.case,
        a.out,
        n=a.n,
        seed=a.seed,
        segment=a.segment,
        inputs_csv=a.inputs,
        criterion=a.criterion,
        chunk=a.chunk,
        overrides=a.set,
    )
    print(f"gravado {out}: {meta['n_ok']} amostras válidas em {meta['cpu']:.1f} s")


if __name__ == "__main__":
    main()
