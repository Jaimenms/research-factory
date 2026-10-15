"""Cria um artigo novo em `papers/NN-<slug>/` a partir de `papers/_template/` (copia, numera e substitui os
marcadores @@SLUG@@ e @@TITLE@@).

uso: uv run python tools/new_paper.py <slug> [--title "Working title"]
     ex.: uv run python tools/new_paper.py surrogate-models --title "Surrogate models for ..."
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers"


def next_number():
    nums = [int(m.group(1)) for p in PAPERS.iterdir() if (m := re.match(r"^(\d{2})-", p.name))]
    return max(nums, default=0) + 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slug", help="nome curto em inglês, minúsculas e hífens (ex.: surrogate-models)")
    parser.add_argument("--title", default="Working title", help="título de trabalho do artigo")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", args.slug):
        sys.exit("slug inválido: use minúsculas, dígitos e hífens")
    name = f"{next_number():02d}-{args.slug}"
    dst = PAPERS / name
    if dst.exists():
        sys.exit(f"já existe: {dst}")
    shutil.copytree(PAPERS / "_template", dst)
    for path in dst.rglob("*"):
        if path.is_file() and path.suffix in {".tex", ".py", ".bib", ""}:
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("@@SLUG@@", name).replace("@@TITLE@@", args.title), encoding="utf-8")
    (dst / "figures" / "static").mkdir(exist_ok=True)
    (dst / "figures" / "static" / ".gitkeep").touch()
    print(f"artigo criado: {dst.relative_to(ROOT)}  (cd {dst.relative_to(ROOT)} && make)")


if __name__ == "__main__":
    main()
