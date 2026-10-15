"""Linha de comando: `cmfsim run caso.json [--out corrida.npz] [--backend jax] [--set caminho=valor]`,
`cmfsim validate casos...`, `cmfsim batch lote.json`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io.spec import Case, load_json, validate_case
from .results import Results
from .simulation import simulate


def cmd_run(a):
    d = load_json(a.case, a.set)
    case = Case.from_dict(d)
    res = simulate(case, backend=a.backend)
    out = a.out or Path("runs") / f"{case.name}.npz"
    res.save(out)
    print("gravado", out)


def cmd_validate(a):
    rc = 0
    for f in a.cases:
        errs = validate_case(load_json(f))
        if errs:
            rc = 1
            print(f"{f}: INVÁLIDO")
            for e in errs:
                print("  ", e)
        else:
            print(f"{f}: ok")
    sys.exit(rc)


def cmd_batch(a):
    b = json.load(open(a.batch))
    out_dir = Path(b.get("out_dir", "runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in b["runs"]:
        if a.only and r["name"] not in a.only:
            continue
        out = out_dir / f"{r['name']}.npz"
        if out.exists() and not a.force:
            print("existe, saltado:", out)
            continue
        d = load_json(r["case"], r.get("set"))
        case = Case.from_dict(d)
        try:
            res = simulate(case, backend=r.get("backend"))
            res.save(out)
            print("gravado", out)
        except Exception as e:  # uma entrada que rebenta não trava o lote
            print(f"FALHOU {r['name']}: {type(e).__name__}: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cmfsim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="simula um caso JSON e grava a corrida")
    p.add_argument("case")
    p.add_argument("--out")
    p.add_argument("--backend", choices=["numpy", "jax"])
    p.add_argument("--set", action="append", default=[], help="override caminho=valor (repetível)")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("validate", help="valida casos contra o schema")
    p.add_argument("cases", nargs="+")
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("batch", help="corre um lote de casos")
    p.add_argument("batch")
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", nargs="*")
    p.set_defaults(fn=cmd_batch)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
