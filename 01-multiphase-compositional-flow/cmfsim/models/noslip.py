"""Modelo de não escorregamento (§2.2.4, §3.1): v_L = v_G, fases em equilíbrio (holdup = fração
volumétrica do flash), incógnitas P e T. Em vez de expandir as derivadas totais termo a termo
(150–163), o modelo escreve os fluxos conservados Phi(P, T) = [G v + P, h_mix] e resolve o 2x2
J u' = S com J = dPhi/du: no backend JAX, J vem da diferenciação automática (`jax.linearize`: um
flash por avaliação, com tangente, e duas avaliações lineares); no numpy, por diferenças finitas
de 1.ª ordem para trás com dP e dT como na dissertação (169–170, três flashes por avaliação), que
é o caminho fiel.

Termo dissipativo F (−dP/dx sem aceleração): `friction="homogeneous"` (a dissertação não diz como
calculou o atrito no modelo de não escorregamento, B15 em docs/05): Churchill na mistura homogénea
(rho_mix e mu_mix ponderadas em volume) mais a gravidade rho_mix g sin theta.

Os parâmetros do caso (tubo por troço, vazão) são argumentos (`params`) e não atributos: uma
função compilada serve todos os casos com a mesma estrutura. O lado direito aceita e devolve a
variável auxiliar `aux` = ln K do último flash (arranque quente, issue 007)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend import NUMPY
from ..closures.common import G, WALL_FRICTION_MULT
from ..closures.friction import fanning, vabs
from ..thermo import ComponentSet, FlashOptions, phase_state
from .base import FluxModel


@dataclass(frozen=True)
class NoSlipOptions:
    friction: str = "homogeneous"
    derivatives: str = "autodiff"  # autodiff (JAX) | finite_difference (dP, dT, como a dissertação)
    dP: float = 1000.0  # Pa
    dT: float = 0.01  # K
    kinetic_energy: bool = False  # termo cinético na energia (opção A12)
    wall_friction: str = "fanning"  # fanning | darcy_as_printed (ver ClosureOptions)


def mixture_state(P, T, z, cs: ComponentSet, flash: FlashOptions, bk=NUMPY, lnK0=None):
    """Estado da mistura homogénea em (n,) pontos: rho_mix, h_mix, mu_mix, lambda_G e o PhaseState."""
    st = phase_state(T, P, z, cs, flash, bk, lnK0=lnK0)
    wG = st.w_G
    vol_G = wG / st.rho_V
    vol_L = (1.0 - wG) / st.rho_L
    rho_mix = 1.0 / (vol_G + vol_L)
    lam_G = vol_G * rho_mix
    h_mix = wG * st.h_V + (1.0 - wG) * st.h_L
    mu_mix = lam_G * st.mu_V + (1.0 - lam_G) * st.mu_L
    return rho_mix, h_mix, mu_mix, lam_G, st


class NoSlipModel(FluxModel):
    """Modelo de não escorregamento sobre a interface `FluxModel`: u = (P, T), Phi = [G v + P, h_mix
    (+ v^2/2 com `kinetic_energy`)], S = [-(F_atrito + rho g sin theta), Q_w/m - g sin theta]."""

    def __init__(
        self, cs: ComponentSet, z, flash: FlashOptions = FlashOptions(), opts: NoSlipOptions = NoSlipOptions(), bk=NUMPY
    ):
        self.cs, self.z = cs, np.asarray(z, float)
        self.flash, self.opts, self.bk = flash, opts, bk
        self.derivatives, self.fd_dP, self.fd_dT = opts.derivatives, opts.dP, opts.dT

    @staticmethod
    def params_from(pipe_arrays, m, bk=NUMPY):
        return FluxModel.point_params(pipe_arrays, m, bk)

    def fluxes_sources(self, P, T, D, eps, theta, U, T_amb, m, lnK0=None):
        """Vetorizado em n pontos: Phi (n, 2), S (n, 2), ln K (n, nc), estado (dict)."""
        xp = self.bk.xp
        bk = self.bk
        n = P.shape[0]
        A = xp.pi * D**2 / 4.0
        Gm = m / A
        zz = xp.broadcast_to(xp.asarray(self.z)[None, :], (n, self.cs.n))
        rho_mix, h_mix, mu_mix, lam_G, st = mixture_state(P, T, zz, self.cs, self.flash, bk, lnK0=lnK0)
        v = Gm / rho_mix
        e = h_mix + (0.5 * v**2 if self.opts.kinetic_energy else 0.0)
        phi = xp.stack([Gm * v + P, e], axis=1)
        f, _ = fanning(v, rho_mix, mu_mix, D, eps, bk=bk)
        f = f * WALL_FRICTION_MULT[self.opts.wall_friction]
        F_fric = f * rho_mix * v * vabs(v, bk) / 2.0 * 4.0 / D
        F_grav = rho_mix * G * xp.sin(theta)
        Qw = -U * xp.pi * D * (T - T_amb)
        S = xp.stack([-(F_fric + F_grav), Qw / m - G * xp.sin(theta)], axis=1)
        state = dict(st=st, rho_mix=rho_mix, h_mix=h_mix, mu_mix=mu_mix, lam_G=lam_G, v=v, A=A)
        return phi, S, st.lnK, state

    # compatibilidade com o código anterior (testes e pós-processamento)
    def state(self, u, seg, params, lnK0=None):
        return self.point(u, seg, params, lnK0)

    def fluxes(self, u, seg, params, lnK0=None):
        return self.point(u, seg, params, lnK0)[0]

    def sources(self, u, seg, params, lnK0=None):
        return self.point(u, seg, params, lnK0)[1]
