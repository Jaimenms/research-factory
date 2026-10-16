"""Interface comum dos modelos de escoamento: ambos (não escorregamento e mecanicista) se escrevem
como d(Phi)/dx = S com fechamentos locais, Phi = [fluxo de quantidade de movimento + P, entalpia
específica da mistura] e S = [-F, Q_w/m - g sin(theta)]. Sobre esta interface assentam os três
métodos: a marcha (J u' = S com J = dPhi/du, integrador RK45), as diferenças finitas nos fluxos
(M Phi = S nos nós, Tabela 4 da dissertação) e os volumes finitos (Phi_{i+1} - Phi_i = integral de
S na célula). Uma implementação para numpy e JAX; os parâmetros do tubo são argumentos (`params`)."""

from __future__ import annotations

import numpy as np

from ..backend import NUMPY

N_FLUX = 2  # [quantidade de movimento, energia]


class FluxModel:
    """Mixin: a classe concreta implementa `fluxes_sources(P, T, D, eps, theta, U, T_amb, m, lnK0)`,
    vetorizada em n pontos, devolvendo (Phi (n, 2), S (n, 2), ln K (n, nc), estado) com o estado
    (PhaseState e termos do fechamento) para o pós-processamento."""

    bk = NUMPY

    # --- parâmetros por troço (marcha) --------------------------------------------------------
    @staticmethod
    def point_params(pipe_arrays, m, bk=NUMPY):
        """`params` da marcha: arrays por troço (D, roughness, theta, U, T_amb) e a vazão `m`."""
        xp = bk.xp
        p = {k: xp.asarray(v, float) for k, v in pipe_arrays.items()}
        p["m"] = xp.asarray(float(m))
        return p

    @staticmethod
    def _seg(params, seg):
        p = params
        return p["D"][seg], p["roughness"][seg], p["theta"][seg], p["U"][seg], p["T_amb"][seg]

    # --- um ponto (marcha) --------------------------------------------------------------------
    def point(self, u, seg, params, lnK0=None):
        """Phi (2,), S (2,) e ln K (nc,) num ponto u = (P, T) do troço `seg`."""
        D, eps, theta, U, T_amb = self._seg(params, seg)
        one = lambda v: v[None]
        lnK0 = None if lnK0 is None else lnK0[None, :]
        phi, S, lnK, _ = self.fluxes_sources(
            one(u[0]), one(u[1]), one(D), one(eps), one(theta), one(U), one(T_amb), params["m"], lnK0
        )
        return phi[0], S[0], lnK[0]

    def rhs(self, x, u, seg, params, aux=None):
        """du/dx = J^-1 S com J = dPhi/du: `jax.linearize` (um flash com tangente) no JAX; no numpy,
        diferenças de 1.ª ordem para trás com dP e dT (169–170, como a dissertação). Com `aux`
        (ln K do flash anterior) devolve (du, ln K novo)."""
        xp = self.bk.xp
        lnK0 = aux if (aux is not None and self.flash.warm_start) else None
        if self.derivatives == "autodiff" and self.bk.has_ad:
            import jax

            (phi, S, lnK), f_lin = jax.linearize(lambda uu: self.point(uu, seg, params, lnK0), u)
            cols = [f_lin(self.bk.set(xp.zeros((2,)), k, 1.0))[0] for k in range(2)]
            J = xp.stack(cols, axis=1)
        else:
            d = (self.fd_dP, self.fd_dT)
            phi, S, lnK = self.point(u, seg, params, lnK0)
            cols = []
            for k in range(2):
                e = self.bk.set(xp.zeros((2,)), k, d[k])
                cols.append((phi - self.point(u - e, seg, params, lnK0)[0]) / d[k])
            J = xp.stack(cols, axis=1)
        du = xp.linalg.solve(J, S)
        return (du, lnK) if aux is not None else du

    # --- vetorizado (sistemas nodais) --------------------------------------------------------
    def nodes(self, P, T, node_params, m, lnK0=None):
        """Phi (n, 2), S (n, 2), ln K (n, nc) e estado em n nós com parâmetros por nó."""
        q = node_params
        return self.fluxes_sources(P, T, q["D"], q["roughness"], q["theta"], q["U"], q["T_amb"], m, lnK0)
