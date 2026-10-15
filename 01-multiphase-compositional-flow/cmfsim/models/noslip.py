"""Modelo de não escorregamento (§2.2.4, §3.1): v_L = v_G, fases em equilíbrio (holdup = fração
volumétrica do flash), incógnitas P e T. Em vez de expandir as derivadas totais termo a termo
(150–163), o modelo escreve os fluxos conservados Phi(P, T) = [G v + P, h_mix] e resolve o 2x2
J u' = S com J = dPhi/du: no backend JAX, J vem da diferenciação automática (um flash por
avaliação, derivadas exatas); no numpy, por diferenças finitas de 1.ª ordem para trás com dP e dT
como na dissertação (169–170, três flashes por avaliação), que é o caminho fiel.

Termo dissipativo F (−dP/dx sem aceleração): `friction="homogeneous"` (a dissertação não diz como
calculou o atrito no modelo de não escorregamento, B15 em docs/05): Churchill na mistura homogénea
(rho_mix e mu_mix ponderadas em volume) mais a gravidade rho_mix g sin theta."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend import NUMPY
from ..closures.common import G
from ..closures.friction import fanning, vabs
from ..thermo import ComponentSet, FlashOptions, phase_state


@dataclass(frozen=True)
class NoSlipOptions:
    friction: str = "homogeneous"
    derivatives: str = "autodiff"  # autodiff (JAX) | finite_difference (dP, dT, como a dissertação)
    dP: float = 1000.0  # Pa
    dT: float = 0.01  # K
    kinetic_energy: bool = False  # termo cinético na energia (opção A12)


def mixture_state(P, T, z, cs: ComponentSet, flash: FlashOptions, bk=NUMPY):
    """Estado da mistura homogénea em (n,) pontos: rho_mix, h_mix, mu_mix, lambda_G e o PhaseState."""
    st = phase_state(T, P, z, cs, flash, bk)
    wG = st.w_G
    vol_G = wG / st.rho_V
    vol_L = (1.0 - wG) / st.rho_L
    rho_mix = 1.0 / (vol_G + vol_L)
    lam_G = vol_G * rho_mix
    h_mix = wG * st.h_V + (1.0 - wG) * st.h_L
    mu_mix = lam_G * st.mu_V + (1.0 - lam_G) * st.mu_L
    return rho_mix, h_mix, mu_mix, lam_G, st


class NoSlipModel:
    """du/dx para u = (P, T) num tubo com parâmetros por troço (dict de arrays D, roughness, theta, U, T_amb)."""

    def __init__(
        self,
        cs: ComponentSet,
        z,
        m,
        pipe_arrays,
        flash: FlashOptions = FlashOptions(),
        opts: NoSlipOptions = NoSlipOptions(),
        bk=NUMPY,
    ):
        self.cs, self.z, self.m = cs, np.asarray(z, float), float(m)
        self.pipe = {k: bk.xp.asarray(v, float) for k, v in pipe_arrays.items()}
        self.flash, self.opts, self.bk = flash, opts, bk

    def _params(self, seg):
        p = self.pipe
        return p["D"][seg], p["roughness"][seg], p["theta"][seg], p["U"][seg], p["T_amb"][seg]

    def fluxes(self, u, seg):
        """Phi = [G v + P, h_mix (+ v^2/2)] com G = m/A."""
        xp = self.bk.xp
        D = self._params(seg)[0]
        A = xp.pi * D**2 / 4.0
        Gm = self.m / A
        P, T = u[0], u[1]
        rho_mix, h_mix, mu_mix, lam_G, st = mixture_state(
            P[None], T[None], self.z[None, :], self.cs, self.flash, self.bk
        )
        v = Gm / rho_mix[0]
        e = h_mix[0] + (0.5 * v**2 if self.opts.kinetic_energy else 0.0)
        return xp.stack([Gm * v + P, e])

    def sources(self, u, seg):
        """S = [-F, Q_w/m - g sin theta]."""
        xp = self.bk.xp
        bk = self.bk
        D, eps, theta, U, T_amb = self._params(seg)
        A = xp.pi * D**2 / 4.0
        P, T = u[0], u[1]
        rho_mix, h_mix, mu_mix, lam_G, st = mixture_state(P[None], T[None], self.z[None, :], self.cs, self.flash, bk)
        rho = rho_mix[0]
        v = self.m / (A * rho)
        f, _ = fanning(v, rho, mu_mix[0], D, eps, bk=bk)
        F_fric = f * rho * v * vabs(v, bk) / 2.0 * 4.0 / D
        F_grav = rho * G * xp.sin(theta)
        Qw = -U * xp.pi * D * (T - T_amb)
        return xp.stack([-(F_fric + F_grav), Qw / self.m - G * xp.sin(theta)])

    def jacobian(self, u, seg):
        xp = self.bk.xp
        if self.opts.derivatives == "autodiff" and self.bk.has_ad:
            cols = []
            for k in range(2):
                e = self.bk.set(xp.zeros((2,)), k, 1.0)
                cols.append(self.bk.jvp(lambda uu: self.fluxes(uu, seg), (u,), (e,))[1])
            return xp.stack(cols, axis=1)
        d = (self.opts.dP, self.opts.dT)
        phi0 = self.fluxes(u, seg)
        cols = []
        for k in range(2):
            e = self.bk.set(xp.zeros((2,)), k, d[k])
            cols.append((phi0 - self.fluxes(u - e, seg)) / d[k])  # para trás, como (169–170)
        return xp.stack(cols, axis=1)

    def rhs(self, x, u, seg):
        xp = self.bk.xp
        J = self.jacobian(u, seg)
        S = self.sources(u, seg)
        return xp.linalg.solve(J, S)
