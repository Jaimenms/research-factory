"""Blocos `fluid` autocontidos para os casos JSON a partir das tabelas da dissertação
(`dissertation/data/components.json` e `fluids.json`)."""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def fluid_block(name: str, flash=None, constants="thesis") -> dict:
    """`name`: chave de `fluids.json` (p. ex. `table23_rich_gas`)."""
    comps = json.load(open(DATA / "components.json"))["components"]
    fl = json.load(open(DATA / "fluids.json"))["fluids"][name]
    records = []
    for cid, z in fl.items():
        c = comps[cid]
        records.append(
            dict(
                id=cid,
                z=float(z),
                MW=c["MW"],
                Pc=c["Pc"],
                Tc=c["Tc"],
                omega=c["omega"],
                Tb=c["Tb"],
                Hf=c["Hf"],
                cp=list(c["cp"]),
                Nc_eff=c["Nc_eff"],
            )
        )
    d = {
        "type": "compositional",
        "components": records,
        "kij": "zero",
        "eos": {"type": "peng_robinson", "constants": constants},
    }
    if flash:
        d["flash"] = dict(flash)
    return d
