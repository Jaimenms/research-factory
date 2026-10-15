"""Resultados de uma corrida: perfis ao longo de x (P, T e tudo o que deles se deriva) e gravação
em npz v1 com o caso completo (`case_json`), `info_json` e proveniência, de modo que uma corrida
gravada reconstrói o caso sem outra fonte."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RUN_FORMAT = "cmfsim-run/1"


@dataclass
class Results:
    x: np.ndarray
    P: np.ndarray
    T: np.ndarray
    profiles: dict = field(default_factory=dict)  # nome -> array (n,) ou (n, k)
    info: dict = field(default_factory=dict)
    case_dict: dict = field(default_factory=dict)

    def __getitem__(self, key):
        if key == "x":
            return self.x
        if key == "P":
            return self.P
        if key == "T":
            return self.T
        return self.profiles[key]

    def keys(self):
        return ["x", "P", "T"] + list(self.profiles)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"x": self.x, "P": self.P, "T": self.T}
        arrays.update({f"prof__{k}": np.asarray(v) for k, v in self.profiles.items()})
        np.savez(
            path,
            format=RUN_FORMAT,
            info_json=json.dumps(self.info, default=_json_default),
            case_json=json.dumps(self.case_dict, ensure_ascii=False),
            provenance_json=json.dumps(provenance()),
            **arrays,
        )
        return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def provenance():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        commit = ""
    return dict(
        time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        commit=commit,
        python=sys.version.split()[0],
        platform=platform.platform(),
        argv=sys.argv,
    )


def load_run(path) -> Results:
    d = np.load(path, allow_pickle=False)
    if str(d["format"]) != RUN_FORMAT:
        raise ValueError(f"formato de corrida desconhecido: {d['format']}")
    profiles = {k[len("prof__") :]: d[k] for k in d.files if k.startswith("prof__")}
    return Results(
        x=d["x"],
        P=d["P"],
        T=d["T"],
        profiles=profiles,
        info=json.loads(str(d["info_json"])),
        case_dict=json.loads(str(d["case_json"])),
    )
