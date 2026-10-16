"""Gerador dos casos do gasoduto com formação de condensado (§5.6, Tabelas 23–25): um caso base e
seis variantes por `$extends` (T_in 60/80 °C x U 1/3/6 W/m²K na Tabela 25). Fora do simulador.

Os casos usam U_eff = U_FACTOR x U da Tabela 25. Os perfis da Figura 39 são exponenciais puras com a
mesma razão a/U nos seis casos e 3,42x a do balanço de energia com os U impressos; com o fator, as
Figuras 38–40 ficam reproduzidas (T a 0,1–0,3 °C, P a 0,04 bar, início da condensação a < 1 km) e a
§5.5, com o mesmo modelo, não precisa de fator nenhum. Decisão do dono (2026-10-15): erro de
apresentação de U na Tabela 25; ver docs/issues/004.

Uso: uv run python -m generators.condensate [--out cases/condensate] [--dx-out 100]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dissertation.fluids import fluid_block

ROOT = Path(__file__).resolve().parents[1]
CASES = [(1, 60.0, 1.0), (2, 60.0, 3.0), (3, 60.0, 6.0), (4, 80.0, 1.0), (5, 80.0, 3.0), (6, 80.0, 6.0)]
U_FACTOR = 3.42  # U_eff / U da Tabela 25 (ajuste às Figuras 38–40; issue 004)


def base_case(dx_out=100.0, flash=None) -> dict:
    return {
        "schema": "cmfsim-case/1",
        "name": "condensate_5_6_base",
        "description": "Gasoduto de 8 km e 12 in com gás rico (Tabela 23), 70 bar e 15 kg/s na entrada, T_amb 25 °C (Tabela 24); modelo de não escorregamento por marcha em x (§3.1, §5.6).",
        "fluid": fluid_block("table23_rich_gas", flash=flash),
        "pipe": {
            "profile": [
                {
                    "x": 0.0,
                    "D": 0.3048,
                    "roughness": 4.57e-5,
                    "angle_deg": 0.0,
                    "U": U_FACTOR * 1.0,
                    "T_amb": 298.15,
                },
                {"x": 8000.0},
            ]
        },
        "boundary": {"inlet": {"P": 70.0e5, "T": 333.15, "m": 15.0}},
        "model": {
            "type": "noslip",
            "friction": "homogeneous",
            "derivatives": "autodiff",
            "dP": 1000.0,
            "dT": 0.01,
            "regime": {"regime": "automatic"},
        },
        "solver": {
            "type": "march",
            "integrator": "rk45",
            "rtol": 1e-8,
            "atol": 1e-8,
            "first_step": 10.0,
            "backend": "numpy",
        },
        "outputs": {"dx_out": dx_out},
        "meta": {
            "source": "dissertação §5.6, Tabelas 23–25; alvos: Figuras 38–40 (references/figures/data/fig_38..40.json), Tabela 26",
            "U_table25": 1.0,
            "U_factor": U_FACTOR,
            "U_note": "U = U_factor x U da Tabela 25: os perfis da Figura 39 exigem 3,42x o U impresso (issue 004)",
        },
    }


def variant(case_no, T_in_C, U, U_table=None) -> dict:
    """`U` é o valor efetivo escrito no caso; `U_table` (opcional) é o valor impresso na Tabela 25."""
    desc_U = f"U = {U:g} W/(m²K)" if U_table is None else f"U = {U_table:.0f} W/(m²K) da Tabela 25 x {U_FACTOR} = {U:g}"
    d = {
        "schema": "cmfsim-case/1",
        "$extends": "condensate_5_6_base.json",
        "name": f"condensate_5_6_case{case_no}",
        "description": f"Caso {case_no} da Tabela 25: T_in = {T_in_C:.0f} °C, {desc_U}.",
        "pipe": {
            "profile": [
                {"x": 0.0, "D": 0.3048, "roughness": 4.57e-5, "angle_deg": 0.0, "U": U, "T_amb": 298.15},
                {"x": 8000.0},
            ]
        },
        "boundary": {"inlet": {"P": 70.0e5, "T": 273.15 + T_in_C, "m": 15.0}},
    }
    if U_table is not None:
        d["meta"] = {"U_table25": U_table, "U_factor": U_FACTOR}
    return d


def write_all(out_dir: Path, dx_out=100.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "condensate_5_6_base.json").write_text(
        json.dumps(base_case(dx_out), indent=1, ensure_ascii=False) + "\n"
    )
    for n, T, U in CASES:
        (out_dir / f"condensate_5_6_case{n}.json").write_text(
            json.dumps(variant(n, T, round(U_FACTOR * U, 6), U_table=U), indent=1, ensure_ascii=False) + "\n"
        )
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "cases" / "condensate")
    ap.add_argument("--dx-out", type=float, default=100.0)
    a = ap.parse_args(argv)
    print("escritos em", write_all(a.out, a.dx_out))


if __name__ == "__main__":
    main()
