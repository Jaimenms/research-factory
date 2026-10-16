"""Estado termodinâmico das fases num conjunto de pontos (P, T, z): flash e propriedades por fase
(massa específica, entalpia por kg, viscosidade, tensão superficial), com as frações
vaporizadas molar e mássica. É o que os modelos de escoamento consomem."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from ..backend import NUMPY
from .components import ComponentSet, composition_array
from .flash import FlashOptions, flash_pt
from .ideal import h_ideal
from .pr import density, molar_mass, residual_enthalpy
from .surface_tension import surface_tension
from .viscosity import allan_teja_viscosity, pedersen_viscosity


@dataclass
class PhaseState:
    """Arrays por ponto (n,) ou (n, nc). beta: fração vaporizada molar truncada a [0, 1]; w_G: mássica;
    rho_k [kg/m3]; h_k [J/kg]; mu_k [Pa s]; sigma [N/m]; two_phase 0/1."""

    beta: np.ndarray
    beta_raw: np.ndarray
    w_G: np.ndarray
    x: np.ndarray
    y: np.ndarray
    K: np.ndarray
    MW_L: np.ndarray
    MW_V: np.ndarray
    rho_L: np.ndarray
    rho_V: np.ndarray
    h_L: np.ndarray
    h_V: np.ndarray
    mu_L: np.ndarray
    mu_V: np.ndarray
    sigma: np.ndarray
    two_phase: np.ndarray
    iterations: np.ndarray
    residual: np.ndarray
    lnK: np.ndarray = None  # ln K convergido (não truncado): estimativa inicial do flash vizinho (arranque quente)
    polish_iterations: np.ndarray = None  # iterações do polimento de Newton por ponto

    def names(self):
        return [f.name for f in fields(self)]

    def to_numpy(self, bk):
        return PhaseState(
            **{
                f.name: (None if getattr(self, f.name) is None else bk.to_numpy(getattr(self, f.name)))
                for f in fields(self)
            }
        )


def mass_vapor_fraction(beta, MW_L, MW_V):
    """Título w_G a partir da fração molar beta e das massas molares das fases."""
    return beta * MW_V / (beta * MW_V + (1.0 - beta) * MW_L)


def phase_state(T, P, z, cs: ComponentSet, opts: FlashOptions = FlashOptions(), bk=NUMPY, lnK0=None) -> PhaseState:
    """Flash e propriedades em (n,) pontos; z (nc,) ou (n, nc)."""
    xp = bk.xp
    T = xp.asarray(T, float)
    P = xp.asarray(P, float)
    z = xp.asarray(composition_array(z, T.shape[0]) if not bk.is_jax or isinstance(z, np.ndarray) else z, float)
    if z.ndim == 1:
        z = xp.broadcast_to(z[None, :], (T.shape[0], cs.n))
    fl = flash_pt(T, P, z, cs, opts, bk, lnK0=lnK0)
    x, y = fl["x"], fl["y"]
    MW_L = molar_mass(x, cs, bk)
    MW_V = molar_mass(y, cs, bk)
    h_L = (h_ideal(T, x, cs, bk) + residual_enthalpy(T, P, x, cs, "liquid", bk)) / MW_L
    h_V = (h_ideal(T, y, cs, bk) + residual_enthalpy(T, P, y, cs, "vapor", bk)) / MW_V
    return PhaseState(
        beta=fl["beta"],
        beta_raw=fl["beta_raw"],
        w_G=mass_vapor_fraction(fl["beta"], MW_L, MW_V),
        x=x,
        y=y,
        K=fl["K"],
        MW_L=MW_L,
        MW_V=MW_V,
        rho_L=density(T, P, x, cs, "liquid", bk),
        rho_V=density(T, P, y, cs, "vapor", bk),
        h_L=h_L,
        h_V=h_V,
        mu_L=allan_teja_viscosity(T, x, cs, bk),
        mu_V=pedersen_viscosity(T, P, y, cs, bk),
        sigma=surface_tension(T, x, cs, bk),
        two_phase=fl["two_phase"],
        iterations=fl["iterations"],
        residual=fl["residual"],
        lnK=fl["lnK"],
        polish_iterations=fl["polish_iterations"],
    )
