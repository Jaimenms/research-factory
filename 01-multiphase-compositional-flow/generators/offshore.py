"""Gerador do sistema de produção offshore (§5.5, Tabelas 19–22): poço vertical de 1500 m (D 0,127 m),
linha horizontal de 6000 m e riser vertical de 300 m (D 0,1524 m), um único tubo com perfil axial;
fluido de 9 componentes com C7+; 170 bar, 80 °C, 10 kg/s na entrada. Duas variantes: modelo de não
escorregamento (marcha) e mecanicista (diferenças finitas). Alvos: Figuras 32–37 digitalizadas.

Uso: uv run python -m generators.offshore [--n 78]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dissertation.fluids import fluid_block

ROOT = Path(__file__).resolve().parents[1]


def profile():
    return [
        {"x": 0.0, "D": 0.127, "roughness": 2.0e-4, "angle_deg": 90.0, "U": 7.0, "T_amb": 313.15},
        {"x": 1500.0, "D": 0.1524, "roughness": 6.0e-4, "angle_deg": 0.0, "U": 5.0, "T_amb": 285.15},
        {"x": 7500.0, "D": 0.1524, "roughness": 6.0e-4, "angle_deg": 90.0, "U": 5.0, "T_amb": 285.15},
        {"x": 7800.0},
    ]


def noslip_case(dx_out=50.0) -> dict:
    return {
        "schema": "cmfsim-case/1",
        "name": "offshore_5_5_noslip",
        "description": "Sistema offshore (§5.5): poço 1500 m vertical, linha 6000 m horizontal, riser 300 m vertical; fluido da Tabela 19 com C7+; 170 bar, 80 °C, 10 kg/s; modelo de não escorregamento por marcha.",
        "fluid": fluid_block("table19_offshore"),
        "pipe": {"profile": profile()},
        "boundary": {"inlet": {"P": 170.0e5, "T": 353.15, "m": 10.0}},
        "model": {
            "type": "noslip",
            "friction": "homogeneous",
            "derivatives": "autodiff",
            "regime": {"regime": "automatic"},
        },
        "solver": {
            "type": "march",
            "integrator": "rk45",
            "rtol": 1e-7,
            "atol": 1e-7,
            "first_step": 5.0,
            "backend": "jax",
        },
        "outputs": {"dx_out": dx_out},
        "meta": {
            "source": "dissertação §5.5, Tabelas 19–22; alvos: Figuras 32–37 (references/figures/data/fig_32..37.json)"
        },
    }


def mechanistic_case(n=78) -> dict:
    return {
        "schema": "cmfsim-case/1",
        "$extends": "offshore_5_5_noslip.json",
        "name": "offshore_5_5_slip",
        "description": "Variante com escorregamento (§3.2): diferenças finitas com Newton, padrão automático.",
        "model": {"type": "mechanistic", "mass_split": "algebraic", "regime": {"regime": "automatic"}},
        "solver": {
            "type": "fd",
            "n": n,
            "initial_guess": "march_noslip",
            "rtol": 1e-7,
            "atol": 1e-7,
            "first_step": 5.0,
            "backend": "jax",
            "newton": {"max_iter": 40, "tol": 1e-9},
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=78)
    a = ap.parse_args(argv)
    out = ROOT / "cases" / "offshore"
    out.mkdir(parents=True, exist_ok=True)
    (out / "offshore_5_5_noslip.json").write_text(json.dumps(noslip_case(), indent=1, ensure_ascii=False) + "\n")
    (out / "offshore_5_5_slip.json").write_text(json.dumps(mechanistic_case(a.n), indent=1, ensure_ascii=False) + "\n")
    b = json.load(open(ROOT / "batches" / "dissertation.json"))
    names = {r["name"] for r in b["runs"]}
    for nm in ("offshore_5_5_noslip", "offshore_5_5_slip"):
        if nm not in names:
            b["runs"].append({"name": nm, "case": f"cases/offshore/{nm}.json"})
    (ROOT / "batches" / "dissertation.json").write_text(json.dumps(b, indent=1) + "\n")
    print("casos em", out, "e lote batches/dissertation.json atualizado")


if __name__ == "__main__":
    main()
