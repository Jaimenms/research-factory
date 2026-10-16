"""Extrai do PDF da dissertação os dados numéricos impressos: o Apêndice A (Tabelas 36 e 37,
200 perdas de carga), o Apêndice B (Tabelas 38–40, 300 amostras do modelo físico por trecho para o
substituto) e o Apêndice C (Tabelas 41–43, R² das 135 configurações da grelha por trecho) para CSV.
Reprodutível: `uv run python -m dissertation.extract`.

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
PAGES_APPENDIX_B = range(173, 193)  # Tabelas 38–40: poço, linha de escoamento, riser
PAGES_APPENDIX_C = range(193, 203)  # Tabelas 41–43: grelha de hiperparâmetros por trecho (até à última página)
SEGMENTS = ("well", "line", "riser")
ACTIVATIONS = ("relu", "sigmoid", "tanh")
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


def _split_tables(text: str, titles):
    """Fatias do texto entre títulos consecutivos («Tabela 38», ...)."""
    idx = [text.index(t) for t in titles]
    return [text[a:b] for a, b in zip(idx, idx[1:] + [len(text)])]


def parse_samples(text: str, n_expected=300, max_gap=5, min_rows=295):
    """Apêndice B: linhas (caso, P_in bar, T_in K, m kg/s, P_out bar, T_out K) com casos 0..299. A
    tabela impressa salta alguns números de caso (p. ex. o 253 na Tabela 38), por isso aceita-se o
    próximo inteiro até `max_gap` à frente; os casos em falta ficam registados."""
    toks = _tokens(text)
    rows, i, expected = [], 0, 0
    while i < len(toks) and expected < n_expected:
        t = toks[i]
        if t is not None and "," not in t and expected <= int(_to_float(t)) < expected + max_gap:
            cand = toks[i : i + 6]
            if len(cand) == 6 and all(c is not None and ("," in c or k == 0) for k, c in enumerate(cand)):
                case = int(cand[0])
                rows.append((case, *[_to_float(c) for c in cand[1:]]))
                expected = case + 1
                i += 6
                continue
        i += 1
    if len(rows) < min_rows:
        raise ValueError(f"Apêndice B: esperadas ~{n_expected} amostras, encontradas {len(rows)}")
    return rows


def extract_appendix_b(pdf: Path = PDF, out_dir: Path = HERE / "data"):
    text = page_text(pdf, PAGES_APPENDIX_B)
    chunks = _split_tables(text, ("Tabela 38", "Tabela 39", "Tabela 40"))
    written = []
    for seg, chunk in zip(SEGMENTS, chunks):
        rows = parse_samples(chunk)
        path = out_dir / f"appendix_b_{seg}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("case", "P_in_bar", "T_in_K", "m_kg_s", "P_out_bar", "T_out_K"))
            w.writerows(rows)
        written.append(path)
    return written


_WORD = re.compile(r"^(relu|sigmoid|tanh|TRUE|FALSE)$")


def _tokens_c(text: str):
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(s if (_NUM.match(s) or _WORD.match(s)) else None)
    return out


def parse_grid(text: str, n_expected=135, max_gap=5, min_rows=130):
    """Apêndice C: linhas (caso, neurónios, taxa, ativação, camadas, R² P, R² T, soma, ambos > 0,9);
    tolera números de caso saltados como `parse_samples`."""
    toks = _tokens_c(text)
    rows, i, expected = [], 0, 0
    while i < len(toks) and expected < n_expected:
        t = toks[i]
        if t is not None and _NUM.match(t) and "," not in t and expected <= int(t) < expected + max_gap:
            cand = toks[i : i + 9]
            if (
                len(cand) == 9
                and all(t is not None for t in cand)
                and cand[3] in ACTIVATIONS
                and cand[8] in ("TRUE", "FALSE")
            ):
                expected = int(cand[0])
                rows.append(
                    (
                        expected,
                        int(_to_float(cand[1])),
                        _to_float(cand[2]),
                        cand[3],
                        int(_to_float(cand[4])),
                        _to_float(cand[5]),
                        _to_float(cand[6]),
                        _to_float(cand[7]),
                        cand[8] == "TRUE",
                    )
                )
                expected += 1
                i += 9
                continue
        i += 1
    if len(rows) < min_rows:
        raise ValueError(f"Apêndice C: esperadas ~{n_expected} configurações, encontradas {len(rows)}")
    return rows


def extract_appendix_c(pdf: Path = PDF, out_dir: Path = HERE / "data"):
    text = page_text(pdf, PAGES_APPENDIX_C)
    chunks = _split_tables(text, ("Tabela 41", "Tabela 42", "Tabela 43"))
    written = []
    for seg, chunk in zip(SEGMENTS, chunks):
        rows = parse_grid(chunk)
        path = out_dir / f"appendix_c_{seg}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ("case", "neurons", "learning_rate", "activation", "layers", "r2_P", "r2_T", "r2_sum", "both_above_0_9")
            )
            w.writerows(rows)
        written.append(path)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, default=PDF)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for fn in (extract_appendix_a, extract_appendix_b, extract_appendix_c):
        for p in fn(a.pdf, a.out):
            print("escrito", p)


if __name__ == "__main__":
    main()
