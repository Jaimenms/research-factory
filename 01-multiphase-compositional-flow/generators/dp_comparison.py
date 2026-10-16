"""Gerador da comparação de perda de carga (§5.4, Tabelas 15–18, Apêndice A): dois casos base
(linha horizontal e vertical de 100 m, fluido CH4 70 % + C7+ 30 %, T_in 40 °C, parede adiabática)
e um lote com os 100 + 100 pontos da grelha D x P_in x m por overrides `--set`, para o modelo
mecanicista (diferenças finitas) e, em paralelo, o de não escorregamento (marcha). Fora do simulador.

Uso: uv run python -m generators.dp_comparison [--n 30] [--models mechanistic,noslip]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dissertation.fluids import fluid_block

ROOT = Path(__file__).resolve().parents[1]
D_IN = (4, 6, 8, 10)
P_BAR = (20, 55, 90, 125, 160)
M_KGS = (2, 4, 6, 8, 10)
INCH = 0.0254


def base_case(orientation: str, model: str, n: int) -> dict:
    vertical = orientation == "vertical"
    d = {
        "schema": "cmfsim-case/1",
        "name": f"dp_{orientation}_{model}_base",
        "description": f"Linha {orientation} de 100 m (Tabela 15), fluido CH4/C7+ 70/30 (Tabela 17), T_in 40 °C, U = 0; grelha D x P_in x m da Tabela 16 por overrides no lote.",
        "fluid": fluid_block("table17_dp_comparison"),
        "pipe": {
            "profile": [
                {
                    "x": 0.0,
                    "D": 4 * INCH,
                    "roughness": 4.5e-5,
                    "angle_deg": 90.0 if vertical else 0.0,
                    "U": 0.0,
                    "T_amb": 313.15,
                },
                {"x": 100.0},
            ]
        },
        "boundary": {"inlet": {"P": 20.0e5, "T": 313.15, "m": 2.0}},
        "outputs": {"dx_out": 5.0},
        "meta": {
            "source": "dissertação §5.4; alvo: dissertation/data/appendix_a_{orientation}.csv (coluna dissertation)"
        },
    }
    if model == "mechanistic":
        d["model"] = {"type": "mechanistic", "mass_split": "algebraic", "regime": {"regime": "automatic"}}
        d["solver"] = {
            "type": "fd",
            "n": n,
            "initial_guess": "march_noslip",
            "rtol": 1e-7,
            "atol": 1e-7,
            "first_step": 2.0,
            "backend": "jax",
            "newton": {"max_iter": 40, "tol": 1e-9},
        }
    else:
        d["model"] = {"type": "noslip", "regime": {"regime": "automatic"}}
        d["solver"] = {
            "type": "march",
            "integrator": "rk45",
            "rtol": 1e-8,
            "atol": 1e-8,
            "first_step": 2.0,
            "backend": "numpy",
        }
    return d


def batch(models, out_dir="runs/dp_comparison") -> dict:
    runs = []
    for orientation in ("horizontal", "vertical"):
        for model in models:
            case = f"cases/dp/dp_{orientation}_{model}_base.json"
            k = 0
            for D in D_IN:
                for Pb in P_BAR:
                    for m in M_KGS:
                        k += 1
                        runs.append(
                            {
                                "name": f"dp_{orientation}_{model}_{k:03d}",
                                "case": case,
                                "set": [
                                    f"pipe.profile.0.D={D * INCH}",
                                    f"boundary.inlet.P={Pb * 1e5}",
                                    f"boundary.inlet.m={float(m)}",
                                ],
                            }
                        )
    return {"out_dir": out_dir, "runs": runs}


#: lotes de sensibilidade sobre os mesmos 200 pontos (volume I, §4.2): nome -> (pasta de saída,
#: orientações, modelos, overrides extra). A rugosidade de 0,2 mm é a dos scripts originais dos casos
#: (legacy/issues L06.1); o atrito de Darcy em tensões de Fanning (×4) é a bandeira `wall_friction`.
SENSITIVITY = {
    "roughness": ("runs/dp_roughness", ("horizontal",), ("mechanistic", "noslip"), ["pipe.profile.0.roughness=0.0002"]),
    "darcy": (
        "runs/dp_comparison_darcy",
        ("horizontal", "vertical"),
        ("mechanistic",),
        ["model.regime.wall_friction=darcy_as_printed"],
    ),
}


def sensitivity_batch(name: str) -> dict:
    out_dir, orientations, models, extra = SENSITIVITY[name]
    base = batch(models)
    runs = [
        {**r, "set": r["set"] + extra}
        for r in base["runs"]
        if any(f"dp_{o}_" in r["name"] for o in orientations) and any(f"_{m}_" in r["name"] for m in models)
    ]
    return {"out_dir": out_dir, "runs": runs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30, help="nós do método das diferenças finitas")
    ap.add_argument("--models", default="mechanistic,noslip")
    ap.add_argument(
        "--sensitivity",
        choices=sorted(SENSITIVITY),
        help="escreve só o lote de sensibilidade batches/dp_<nome>.json (casos base inalterados)",
    )
    a = ap.parse_args(argv)
    if a.sensitivity:
        name = "dp_comparison_darcy" if a.sensitivity == "darcy" else f"dp_{a.sensitivity}"
        b = sensitivity_batch(a.sensitivity)
        (ROOT / "batches" / f"{name}.json").write_text(json.dumps(b, indent=1) + "\n")
        print(f"lote batches/{name}.json com {len(b['runs'])} corridas")
        return
    models = a.models.split(",")
    out = ROOT / "cases" / "dp"
    out.mkdir(parents=True, exist_ok=True)
    for orientation in ("horizontal", "vertical"):
        for model in models:
            (out / f"dp_{orientation}_{model}_base.json").write_text(
                json.dumps(base_case(orientation, model, a.n), indent=1, ensure_ascii=False) + "\n"
            )
    (ROOT / "batches" / "dp_comparison.json").write_text(json.dumps(batch(models), indent=1) + "\n")
    print("casos em", out, "e lote batches/dp_comparison.json com", 200 * len(models), "corridas")


if __name__ == "__main__":
    main()
