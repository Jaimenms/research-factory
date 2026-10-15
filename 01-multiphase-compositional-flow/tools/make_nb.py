"""Converte fontes .py com marcadores `# %%` / `# %% [markdown]` em notebooks .ipynb e, com
`--execute`, executa-os com o nbconvert (padrão copiado de mfsim/tools/make_nb.py).

uso: uv run python tools/make_nb.py notebooks_src/thesis/nb_5_1_thermo.py [--execute]
Os notebooks ficam em notebooks/<sub>/<nome>.ipynb e correm com a raiz do projeto como
diretório de trabalho (as fontes usam ROOT = Path.cwd() quando não há __file__).
"""

import re
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]


def convert(src: Path, dst: Path):
    text = src.read_text(encoding="utf-8")
    parts = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    for i in range(1, len(parts), 2):
        marker, body = parts[i].strip(), parts[i + 1]
        body = body.strip("\n")
        if not body.strip():
            continue
        if "markdown" in marker:
            lines = [re.sub(r"^# ?", "", ln) for ln in body.splitlines()]
            cells.append(new_markdown_cell("\n".join(lines)))
        else:
            cells.append(new_code_cell(body))
    nb = new_notebook(
        cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, dst)
    return dst


def build(src, execute=False):
    src = Path(src).resolve()
    try:
        rel = src.relative_to((ROOT / "notebooks_src").resolve())
    except ValueError:
        rel = Path(src.name)
    dst = ROOT / "notebooks" / rel.with_suffix(".ipynb")
    convert(src, dst)
    print("notebook:", dst)
    if execute:
        cmd = [
            "uv",
            "run",
            "--project",
            str(ROOT),
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=3600",
            str(dst),
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)
        print("executado:", dst)
    return dst


if __name__ == "__main__":
    build(sys.argv[1], "--execute" in sys.argv)
