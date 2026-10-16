"""Gerador dos casos do modelo substituto (§5.7): os três trechos do sistema offshore da §5.5 (Tabela 21)
simulados isoladamente com o fluido da Tabela 6 e o modelo de não escorregamento, mais o sistema
completo com esse fluido (referência das Figuras 57–58). As condições de entrada escritas nos casos
são o centro das faixas das Tabelas 27–29; a massa de dados varia P_in, T_in e m por cima delas
(`surrogate.dataset`). Fora do simulador.

Uso: uv run python -m generators.surrogate [--out cases/surrogate]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dissertation.fluids import fluid_block

ROOT = Path(__file__).resolve().parents[1]
FLUID = "table6_validation"
SEGMENTS = {
    # Tabela 21: D, rugosidade, inclinação, U, T_amb, comprimento; centro das faixas das Tabelas 27–29
    "well": dict(D=0.127, roughness=2.0e-4, angle_deg=90.0, U=7.0, T_amb=313.15, L=1500.0, P_in=165.0e5, T_in=353.15),
    "line": dict(D=0.1524, roughness=6.0e-4, angle_deg=0.0, U=5.0, T_amb=285.15, L=6000.0, P_in=80.0e5, T_in=343.15),
    "riser": dict(D=0.1524, roughness=6.0e-4, angle_deg=90.0, U=5.0, T_amb=285.15, L=300.0, P_in=60.0e5, T_in=323.15),
}
M_CENTER = 13.5


def _solver():
    return {
        "type": "march",
        "integrator": "rk45",
        "rtol": 1e-7,
        "atol": 1e-7,
        "first_step": 5.0,
        "backend": "jax",
    }


def _model():
    return {"type": "noslip", "friction": "homogeneous", "derivatives": "autodiff", "regime": {"regime": "automatic"}}


def segment_case(name: str, n_out=10) -> dict:
    s = SEGMENTS[name]
    return {
        "schema": "cmfsim-case/1",
        "name": f"surrogate_{name}",
        "description": f"Trecho «{name}» do sistema offshore (Tabela 21) isolado, fluido da Tabela 6, não escorregamento; entrada no centro das faixas da Tabela {dict(well=27, line=28, riser=29)[name]} (§5.7).",
        "fluid": fluid_block(FLUID),
        "pipe": {
            "profile": [
                {
                    "x": 0.0,
                    "D": s["D"],
                    "roughness": s["roughness"],
                    "angle_deg": s["angle_deg"],
                    "U": s["U"],
                    "T_amb": s["T_amb"],
                },
                {"x": s["L"]},
            ]
        },
        "boundary": {"inlet": {"P": s["P_in"], "T": s["T_in"], "m": M_CENTER}},
        "model": _model(),
        "solver": _solver(),
        "outputs": {"dx_out": s["L"] / n_out},
        "meta": {"source": "dissertação §5.7 (Tabelas 21, 27–29); amostras do autor no Apêndice B", "segment": name},
    }


def system_case(dx_out=50.0) -> dict:
    return {
        "schema": "cmfsim-case/1",
        "name": "surrogate_offshore_table6",
        "description": "Sistema offshore completo (Tabela 21) com o fluido da Tabela 6, não escorregamento, 170 bar, 80 °C, 20 kg/s (vazão lida na legenda das figuras): referência física para o substituto encadeado (Figuras 57–58, Tabela 33).",
        "fluid": fluid_block(FLUID),
        "pipe": {
            "profile": [
                {"x": 0.0, "D": 0.127, "roughness": 2.0e-4, "angle_deg": 90.0, "U": 7.0, "T_amb": 313.15},
                {"x": 1500.0, "D": 0.1524, "roughness": 6.0e-4, "angle_deg": 0.0, "U": 5.0, "T_amb": 285.15},
                {"x": 7500.0, "D": 0.1524, "roughness": 6.0e-4, "angle_deg": 90.0, "U": 5.0, "T_amb": 285.15},
                {"x": 7800.0},
            ]
        },
        "boundary": {"inlet": {"P": 170.0e5, "T": 353.15, "m": 20.0}},
        "model": _model(),
        "solver": _solver(),
        "outputs": {"dx_out": dx_out},
        "meta": {"source": "dissertação §5.7 (Figuras 57–58, Tabela 33); P e T da Tabela 22, m = 20 kg/s das figuras"},
    }


def write_all(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SEGMENTS:
        (out_dir / f"{name}.json").write_text(json.dumps(segment_case(name), indent=1, ensure_ascii=False) + "\n")
    (out_dir / "offshore_table6_noslip.json").write_text(json.dumps(system_case(), indent=1, ensure_ascii=False) + "\n")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "cases" / "surrogate")
    a = ap.parse_args(argv)
    print("escritos em", write_all(a.out))


if __name__ == "__main__":
    main()
