"""Casos em JSON (`cmfsim-case/1`): leitura com herança (`$extends`) e overrides por caminho,
validação pelo schema, e construção dos objetos que `simulate` consome. O caso é autocontido:
o bloco `fluid.components` traz as propriedades de cada componente (nada é lido de bases de
dados externas pelo simulador)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from ..closures import ClosureOptions
from ..models.mechanistic import MechanisticOptions
from ..models.noslip import NoSlipOptions
from ..solvers.newton import NewtonOptions
from ..pipe import PipeProfile
from ..solvers.rk45 import RK45Options
from ..thermo import ComponentSet, FlashOptions
from .extends import apply_overrides, merge_case

SCHEMA_ID = "cmfsim-case/1"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "cmfsim-case-1.schema.json"


def load_json(path, overrides=None) -> dict:
    """Carrega um caso, resolvendo `$extends` (caminhos relativos ao ficheiro) e aplicando overrides."""
    path = Path(path)
    d = json.load(open(path))
    chain = []
    while "$extends" in d:
        bases = d.pop("$extends")
        if isinstance(bases, str):
            bases = [bases]
        merged = {}
        for b in bases:
            bp = (path.parent / b).resolve()
            base = load_json(bp)
            chain.append(str(bp))
            merged = merge_case(merged, base) if merged else base
        d = merge_case(merged, d)
    if chain:
        d.setdefault("meta", {})["extends"] = chain
    if overrides:
        d = apply_overrides(d, overrides)
    return d


def validate_case(d: dict):
    """Valida contra o schema (jsonschema 2020-12); devolve a lista de violações (vazia se válido)."""
    import jsonschema

    schema = json.load(open(SCHEMA_PATH))
    v = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in e.absolute_path)}: {e.message}"
        for e in sorted(v.iter_errors(d), key=lambda e: list(e.absolute_path))
    ]


def _opts_from(cls, d: dict, skip=()):
    names = {f.name for f in fields(cls)} - set(skip)
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Case:
    name: str
    cs: ComponentSet
    z: np.ndarray
    flash: FlashOptions
    pipe: PipeProfile
    inlet: dict  # P, T, m
    model_type: str
    noslip: NoSlipOptions
    closures: ClosureOptions
    solver_type: str
    rk45: RK45Options
    backend: str
    dx_out: float
    mechanistic: MechanisticOptions = MechanisticOptions()
    newton: NewtonOptions = NewtonOptions()
    n_nodes: int = 30
    initial_guess: str = "march_noslip"
    outlet: dict = field(default_factory=dict)
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, validate=True) -> "Case":
        if validate:
            errs = validate_case(d)
            if errs:
                raise ValueError("caso inválido:\n  " + "\n  ".join(errs))
        f = d["fluid"]
        comps = f["components"]
        kij = f.get("kij", "zero")
        cs = ComponentSet.from_records(comps, kij=None if kij == "zero" else np.asarray(kij, float))
        if f.get("eos", {}).get("constants", "thesis") == "exact":
            cs = cs.with_exact_pr_constants()
        if f.get("cp_temperature", "kelvin") != "kelvin":
            from dataclasses import replace as _replace

            cs = _replace(cs, cp_temperature=f["cp_temperature"])
        z = np.array([c["z"] for c in comps], float)
        z = z / z.sum()
        flash = _opts_from(FlashOptions, f.get("flash", {}))
        pipe = PipeProfile.from_json(d["pipe"])
        m = d["model"]
        s = d["solver"]
        rk = _opts_from(RK45Options, {k: (np.inf if v is None else v) for k, v in s.items()})
        return cls(
            name=d.get("name", "case"),
            cs=cs,
            z=z,
            flash=flash,
            pipe=pipe,
            inlet=dict(d["boundary"]["inlet"]),
            model_type=m["type"],
            noslip=_opts_from(NoSlipOptions, m),
            closures=_opts_from(ClosureOptions, m.get("regime", {})),
            solver_type=s["type"],
            rk45=rk,
            backend=s.get("backend", "numpy"),
            dx_out=float(d.get("outputs", {}).get("dx_out", pipe.L / 100.0)),
            mechanistic=_opts_from(MechanisticOptions, m),
            newton=_opts_from(NewtonOptions, s.get("newton", {})),
            n_nodes=int(s.get("n", d.get("pipe", {}).get("n", 30))),
            initial_guess=s.get("initial_guess", "march_noslip"),
            outlet=dict(d["boundary"].get("outlet", {})),
            raw=copy.deepcopy(d),
        )


def case_from_json(path, overrides=None, validate=True) -> Case:
    return Case.from_dict(load_json(path, overrides), validate=validate)


def case_to_json(d: dict, path):
    """Grava um dicionário de caso (validado) com formatação estável."""
    errs = validate_case(d)
    if errs:
        raise ValueError("caso inválido:\n  " + "\n  ".join(errs))
    with open(path, "w") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
        f.write("\n")
