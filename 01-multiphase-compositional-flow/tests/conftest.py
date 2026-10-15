"""Fixtures comuns: raiz do projeto, dados da dissertação, conjuntos de componentes e fluidos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dissertation" / "data"


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def components_db():
    return json.load(open(DATA / "components.json"))["components"]


@pytest.fixture(scope="session")
def fluids_db():
    return json.load(open(DATA / "fluids.json"))["fluids"]


def make_fluid(components_db, fluid):
    """(ComponentSet, z) para um fluido {nome: fração molar}."""
    from cmfsim.thermo import ComponentSet

    names = list(fluid)
    cs = ComponentSet.from_records([dict(id=k, **components_db[k]) for k in names])
    z = np.array([fluid[k] for k in names])
    return cs, z / z.sum()


@pytest.fixture(scope="session")
def fluid_table6(components_db, fluids_db):
    return make_fluid(components_db, fluids_db["table6_validation"])


@pytest.fixture(scope="session")
def fluid_table23(components_db, fluids_db):
    return make_fluid(components_db, fluids_db["table23_rich_gas"])


@pytest.fixture(scope="session")
def fluid_table19(components_db, fluids_db):
    return make_fluid(components_db, fluids_db["table19_offshore"])
