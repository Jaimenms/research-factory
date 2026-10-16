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


class NoSlipModel:
    """du/dx para u = (P, T). `params`: dict de arrays por troço (D, roughness, theta, U, T_amb) e `m`."""

    def __init__(
        self, cs: ComponentSet, z, flash: FlashOptions = FlashOptions(), opts: NoSlipOptions = NoSlipOptions(), bk=NUMPY
    ):
        self.cs, self.z = cs, np.asarray(z, float)
        self.flash, self.opts, self.bk = flash, opts, bk

    @staticmethod
    def params_from(pipe_arrays, m, bk=NUMPY):
        xp = bk.xp
        p = {k: xp.asarray(v, float) for k, v in pipe_arrays.items()}
        p["m"] = xp.asarray(float(m))
        return p

    @staticmethod
    def _seg(params, seg):
        p = params
        return p["D"][seg], p["roughness"][seg], p["theta"][seg], p["U"][seg], p["T_amb"][seg]

    def state(self, u, seg, params, lnK0=None):
        """Um só flash: fluxos conservados Phi (2,), fontes S (2,) e ln K convergido (nc,)."""
        xp = self.bk.xp
        bk = self.bk
        D, eps, theta, U, T_amb = self._seg(params, seg)
        m = params["m"]
        A = xp.pi * D**2 / 4.0
        Gm = m / A
        P, T = u[0], u[1]
        lnK0 = None if lnK0 is None else lnK0[None, :]
        rho_mix, h_mix, mu_mix, lam_G, st = mixture_state(
            P[None], T[None], self.z[None, :], self.cs, self.flash, bk, lnK0=lnK0
        )
        rho = rho_mix[0]
        v = Gm / rho
        e = h_mix[0] + (0.5 * v**2 if self.opts.kinetic_energy else 0.0)
        phi = xp.stack([Gm * v + P, e])
        f, _ = fanning(v, rho, mu_mix[0], D, eps, bk=bk)
        f = f * WALL_FRICTION_MULT[self.opts.wall_friction]
        F_fric = f * rho * v * vabs(v, bk) / 2.0 * 4.0 / D
        F_grav = rho * G * xp.sin(theta)
        Qw = -U * xp.pi * D * (T - T_amb)
        S = xp.stack([-(F_fric + F_grav), Qw / m - G * xp.sin(theta)])
        return phi, S, st.lnK[0]

    def fluxes(self, u, seg, params, lnK0=None):
        return self.state(u, seg, params, lnK0)[0]

    def sources(self, u, seg, params, lnK0=None):
        return self.state(u, seg, params, lnK0)[1]

    def rhs(self, x, u, seg, params, aux=None):
        """du/dx = J^-1 S. Com `aux` (ln K do flash anterior) devolve (du, ln K novo)."""
        xp = self.bk.xp
        lnK0 = aux if (aux is not None and self.flash.warm_start) else None
        if self.opts.derivatives == "autodiff" and self.bk.has_ad:
            import jax

            (phi, S, lnK), f_lin = jax.linearize(lambda uu: self.state(uu, seg, params, lnK0), u)
            cols = [f_lin(self.bk.set(xp.zeros((2,)), k, 1.0))[0] for k in range(2)]
            J = xp.stack(cols, axis=1)
        else:
            d = (self.opts.dP, self.opts.dT)
            phi, S, lnK = self.state(u, seg, params, lnK0)
            cols = []
            for k in range(2):
                e = self.bk.set(xp.zeros((2,)), k, d[k])
                cols.append((phi - self.fluxes(u - e, seg, params, lnK0)) / d[k])  # para trás, como (169–170)
            J = xp.stack(cols, axis=1)
        du = xp.linalg.solve(J, S)
        return (du, lnK) if aux is not None else du
