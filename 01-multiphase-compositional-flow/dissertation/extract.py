"""Extrai do PDF da dissertação os dados numéricos impressos: o Apêndice A (Tabelas 36 e 37,
200 perdas de carga) para CSV. Reprodutível: `uv run python -m dissertation.extract`.

Uso: python -m dissertation.extract [--pdf caminho] [--out dissertation/data]
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE.parent / "references" / "tese-gustavo-carvalho.pdf"
PAGES_APPENDIX_A = range(168, 173)  # páginas impressas 168–172
D_IN = (4, 6, 8, 10)
P_BAR = (20, 55, 90, 125, 160)
M_KGS = (2, 4, 6, 8, 10)
COLUMNS = {
    "horizontal": ("dissertation", "beggs_brill", "tulsa", "htfs", "gregory"),
    "vertical": ("dissertation", "beggs_brill", "tulsa", "htfs", "aziz"),
}
_NUM = re.compile(r"^-?\d+(,\d+)?(E[+-]\d+)?$")


def page_text(pdf: Path, pages) -> str:
    import pymupdf

    doc = pymupdf.open(str(pdf))
    return "\n".join(doc[p - 1].get_text("text") for p in pages)


def _tokens(text: str):
    """Tokens numéricos, saltando os números de página (linha isolada igual ao número da página)."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _NUM.match(s):
            out.append(s)
        else:
            out.append(None)  # marcador de texto (quebra sequências)
    return out


def _to_float(tok: str) -> float:
    return float(tok.replace(",", "."))


def parse_table(text: str, orientation: str):
    """Linhas de 9 tokens (caso, D, P, m, cinco perdas de carga) a partir do texto da tabela."""
    toks = _tokens(text)
    rows, i, expected = [], 0, 1
    while i < len(toks) and expected <= 100:
        if toks[i] is not None and toks[i] == str(expected):
            cand = toks[i : i + 9]
            if all(t is not None for t in cand) and len(cand) == 9:
                case, D, P, m = (int(_to_float(t)) for t in cand[:4])
                vals = [_to_float(t) for t in cand[4:]]
                if D in D_IN and P in P_BAR and m in M_KGS:
                    rows.append((case, D, P, m, *vals))
                    expected += 1
                    i += 9
                    continue
        i += 1
    if len(rows) != 100:
        raise ValueError(f"{orientation}: esperados 100 casos, encontrados {len(rows)}")
    # a grelha completa 4 x 5 x 5, na ordem impressa (D, P, m)
    grid = [(D, P, m) for D in D_IN for P in P_BAR for m in M_KGS]
    if [r[1:4] for r in rows] != grid:
        raise ValueError(f"{orientation}: ordem (D, P, m) diferente da grelha esperada")
    return rows


def extract_appendix_a(pdf: Path = PDF, out_dir: Path = HERE / "data"):
    text = page_text(pdf, PAGES_APPENDIX_A)
    i36 = text.index("Tabela 36")
    i37 = text.index("Tabela 37")
    tables = {"horizontal": text[i36:i37], "vertical": text[i37:]}
    written = []
    for orientation, chunk in tables.items():
        rows = parse_table(chunk, orientation)
        path = out_dir / f"appendix_a_{orientation}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("case", "D_in", "P_in_bar", "m_kg_s") + tuple(f"dp_bar_{c}" for c in COLUMNS[orientation]))
            for r in rows:
                w.writerow(r)
        written.append(path)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, default=PDF)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for p in extract_appendix_a(a.pdf, a.out):
        print("escrito", p)


if __name__ == "__main__":
    main()
