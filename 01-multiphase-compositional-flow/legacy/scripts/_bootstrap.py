"""Arranque comum dos scripts que correm o código original (legacy/masters_code) sem o alterar.

- não escreve .pyc no submódulo (sys.dont_write_bytecode);
- põe os substitutos de dasslc e scikit-learn no caminho (stubs/) e troca o módulo do flash por redes
  (flash_rna, que importa torch) por um módulo vazio;
- muda o diretório para a raiz do código original, que lê data/*.csv por caminho relativo;
- `OUT` é runs/legacy/ do projeto (fora do git).

Variantes ("patches") aplicam-se em memória, por monkeypatch ou por exec de uma cópia do código-fonte,
nunca editando os ficheiros do submódulo.
"""

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY = os.path.abspath(os.path.join(HERE, "..", "masters_code"))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(PROJECT, "runs", "legacy")

sys.dont_write_bytecode = True
for p in (LEGACY, os.path.join(HERE, "stubs")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.makedirs(OUT, exist_ok=True)
os.chdir(LEGACY)

_rna = types.ModuleType("app.thermodynamic_v2.flash.flash_rna")
_rna.RNAFlash = None
sys.modules.setdefault("app.thermodynamic_v2.flash.flash_rna", _rna)

import matplotlib  # noqa: E402

matplotlib.use("Agg")


def out(name):
    """Caminho absoluto em runs/legacy/."""
    return os.path.join(OUT, name)


def project(*parts):
    """Caminho absoluto dentro do projeto cmfsim."""
    return os.path.join(PROJECT, *parts)
