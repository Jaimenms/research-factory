"""Conjunto de componentes: propriedades dos puros (e pseudocomponentes) em arrays, carregado do
bloco `fluid.components` de um caso JSON (autocontido: cada componente traz as suas
propriedades) ou construído em código.

Unidades SI em base molar por kmol: MW [kg/kmol], Pc [Pa], Tc [K], Tb [K], Hf [J/kmol] a
298.15 K, Cp de gás ideal na forma de Smith et al. (2018), Cp/R = A + B T + C T^2 + D T^-2,
com B, C e D já nas unidades de T em K (isto é, os valores das tabelas multiplicados por
1e-3, 1e-6 e 1e5). `Nc_eff` é o número de carbonos efetivo de Allan & Teja (1991).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

R_GAS = 8314.462618  # J/(kmol K)
T_REF = 298.15  # K, referência da entalpia de formação
FIELDS = ("MW", "Pc", "Tc", "omega", "Tb", "Hf", "cpA", "cpB", "cpC", "cpD", "Nc_eff")


@dataclass(frozen=True)
class ComponentSet:
    names: tuple
    MW: np.ndarray
    Pc: np.ndarray
    Tc: np.ndarray
    omega: np.ndarray
    Tb: np.ndarray
    Hf: np.ndarray
    cpA: np.ndarray
    cpB: np.ndarray
    cpC: np.ndarray
    cpD: np.ndarray
    Nc_eff: np.ndarray
    kij: np.ndarray = field(default=None)
    # constantes de Peng–Robinson: as impressas na dissertação (106–107) por omissão; as exatas
    # (0.4572355289, 0.0777960739) só para comparar com oráculos externos
    omega_a: float = 0.45724
    omega_b: float = 0.07780

    @property
    def n(self) -> int:
        return len(self.names)

    def __post_init__(self):
        n = self.n
        for f in FIELDS:
            a = np.asarray(getattr(self, f), float)
            if a.shape != (n,):
                raise ValueError(f"{f}: forma {a.shape}, esperada ({n},)")
            object.__setattr__(self, f, a)
        k = np.zeros((n, n)) if self.kij is None else np.asarray(self.kij, float)
        if k.shape != (n, n):
            raise ValueError(f"kij: forma {k.shape}, esperada ({n}, {n})")
        object.__setattr__(self, "kij", k)

    def with_exact_pr_constants(self) -> "ComponentSet":
        from dataclasses import replace

        return replace(self, omega_a=0.4572355289213822, omega_b=0.07779607390388846)

    @classmethod
    def from_records(cls, records, kij=None) -> "ComponentSet":
        """`records`: lista de dicionários com `id`, `MW`, `Pc`, `Tc`, `omega`, `Tb`, `Hf`, `cp` [A, B, C, D]
        e opcionalmente `Nc_eff` (default: número de carbonos estimado por (MW - 2)/14)."""
        names = tuple(r["id"] for r in records)
        cp = np.array([r["cp"] for r in records], float)
        nc = np.array([r.get("Nc_eff", max(1.0, (r["MW"] - 2.0) / 14.0)) for r in records], float)
        return cls(
            names=names,
            MW=[r["MW"] for r in records],
            Pc=[r["Pc"] for r in records],
            Tc=[r["Tc"] for r in records],
            omega=[r["omega"] for r in records],
            Tb=[r["Tb"] for r in records],
            Hf=[r["Hf"] for r in records],
            cpA=cp[:, 0],
            cpB=cp[:, 1],
            cpC=cp[:, 2],
            cpD=cp[:, 3],
            Nc_eff=nc,
            kij=kij,
        )

    def to_records(self):
        out = []
        for i, nm in enumerate(self.names):
            out.append(
                dict(
                    id=nm,
                    MW=float(self.MW[i]),
                    Pc=float(self.Pc[i]),
                    Tc=float(self.Tc[i]),
                    omega=float(self.omega[i]),
                    Tb=float(self.Tb[i]),
                    Hf=float(self.Hf[i]),
                    cp=[float(self.cpA[i]), float(self.cpB[i]), float(self.cpC[i]), float(self.cpD[i])],
                    Nc_eff=float(self.Nc_eff[i]),
                )
            )
        return out


def composition_array(z, n_points=None):
    """Composição global como array (n_points, nc) normalizado; aceita (nc,) ou (n, nc)."""
    z = np.asarray(z, float)
    if z.ndim == 1:
        z = z[None, :]
    z = z / z.sum(axis=1, keepdims=True)
    if n_points is not None and z.shape[0] == 1 and n_points != 1:
        z = np.repeat(z, n_points, axis=0)
    return z
