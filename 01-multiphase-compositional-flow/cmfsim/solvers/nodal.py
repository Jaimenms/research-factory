"""Sistemas nodais sobre a interface `FluxModel` (incógnitas P e T por nó), com dois esquemas:

- `fd`: (M Phi)_i = S_i em cada nó, com M o operador de derivada de 2.ª ordem (Tabela 4 da
  dissertação, igual ao `numpy.gradient`): o método 3.2 aplicado aos fluxos, com a repartição
  algébrica de massa; o resíduo de um nó usa os nós a distância <= 2 (banda 2);
- `fv`: Phi_{i+1} - Phi_i = (h/2)(S_i + S_{i+1}) em cada célula (volumes finitos com a fonte pelo
  trapézio): conservativo por construção, sem derivar os fluxos através dos vincos (orvalho,
  mudança de padrão); banda 1.

Malha: nós por troço, pontos de quebra duplicados (continuidade de P e T). Condições de contorno:
P e T na entrada, ou T na entrada e P na saída. Layout de `u`: nó a nó, [P, T] escalados por
[P_ref, T_ref]. Uma implementação para numpy e JAX; os parâmetros do caso vão em `params`."""

from __future__ import annotations

import numpy as np

from ..backend import NUMPY
from ..thermo import phase_state
from .newton import band_colors

N_VARS = 2
I_P, I_T = 0, 1


def mesh(pipe, n_nodes: int):
    """Nós por troço (>= 3 cada, proporcionais ao comprimento) e malha x (lista por troço)."""
    L = np.diff(pipe.x_break)
    counts = np.maximum(3, np.round(n_nodes * L / L.sum()).astype(int) + 1)
    x_seg = [np.linspace(pipe.x_break[k], pipe.x_break[k + 1], counts[k]) for k in range(pipe.n_segments)]
    return tuple(int(c) for c in counts), x_seg


class NodalSystem:
    """Resíduos escalados F(u, params) = 0 para um `FluxModel`. A estrutura (nós por troço, esquema,
    condição de saída) é estática; malha, tubo por nó, entrada/saída e escalas vão em `params`."""

    def __init__(self, model, n_seg, outlet_P: bool, scheme: str, bk=NUMPY):
        if scheme not in ("fd", "fv"):
            raise ValueError(f"esquema desconhecido: {scheme}")
        self.model, self.bk, self.scheme = model, bk, scheme
        self.n_seg = tuple(int(n) for n in n_seg)
        self.offsets = np.concatenate([[0], np.cumsum(self.n_seg)])
        self.n_total = int(self.offsets[-1])
        self.first_nodes = self.offsets[:-1].tolist()
        self.last_nodes = (self.offsets[1:] - 1).tolist()
        self.outlet_P = bool(outlet_P)
        self.width = 2 if scheme == "fd" else 1
        n = self.n_total
        # máscaras estáticas (numpy): que linha vai em cada (nó, variável)
        self.is_first = np.zeros(n, dtype=bool)
        self.is_first[self.first_nodes] = True
        self.is_last_inner = np.zeros(n, dtype=bool)  # último nó de um troço que não é o final
        self.is_last_inner[self.last_nodes[:-1]] = True

    # --- parâmetros --------------------------------------------------------------------------
    @classmethod
    def params_from(cls, pipe, inlet, outlet, n_nodes, cs, z, flash, edge_order=2, bk=NUMPY):
        from ..models.mechanistic import gradient_matrix

        xp = bk.xp
        counts, x_seg = mesh(pipe, n_nodes)
        n_total = int(sum(counts))
        offsets = np.concatenate([[0], np.cumsum(counts)])
        Mb = np.zeros((n_total, n_total))
        dx = np.zeros(n_total)  # tamanho da célula que termina no nó i (0 no primeiro nó de cada troço)
        for k, xs in enumerate(x_seg):
            a, b = offsets[k], offsets[k + 1]
            Mb[a:b, a:b] = gradient_matrix(xs, edge_order)
            dx[a + 1 : b] = np.diff(xs)
        seg_of_node = np.concatenate([np.full(c, k) for k, c in enumerate(counts)])
        arr = pipe.arrays()
        outlet = outlet or {}
        m = float(inlet["m"])
        P_ref = float(inlet["P"]) if "P" in inlet else float(outlet["P"])
        T_ref = float(inlet["T"])
        st_in = phase_state(np.asarray([T_ref]), np.asarray([P_ref]), np.asarray(z)[None, :], cs, flash, NUMPY)
        h_in = float(st_in.w_G[0] * st_in.h_V[0] + (1.0 - st_in.w_G[0]) * st_in.h_L[0])
        p = dict(
            x=np.concatenate(x_seg),
            M=Mb,
            dx=dx,
            D=arr["D"][seg_of_node],
            roughness=arr["roughness"][seg_of_node],
            theta=arr["theta"][seg_of_node],
            U=arr["U"][seg_of_node],
            T_amb=arr["T_amb"][seg_of_node],
            m=m,
            P_ref=P_ref,
            T_ref=T_ref,
            scale=np.array([P_ref, T_ref]),
            h_ref=max(abs(h_in), 1e5),
            L=float(pipe.L),
            P_in=float(inlet.get("P", P_ref)),
            T_in=T_ref,
            P_out=float(outlet.get("P", P_ref)),
        )
        return {k: xp.asarray(v, float) for k, v in p.items()}

    # --- estado ------------------------------------------------------------------------------
    def unpack(self, u, params):
        U = u.reshape(self.n_total, N_VARS) * params["scale"][None, :]
        return U[:, I_P], U[:, I_T]

    def pack(self, P, T, params):
        xp = self.bk.xp
        return (xp.stack([P, T], axis=1) / params["scale"][None, :]).reshape(-1)

    # --- resíduos ----------------------------------------------------------------------------
    def residual(self, u, params, aux=None):
        """F(u) escalado (2 n,); com `aux` (ln K por nó) devolve (F, ln K novo) (arranque quente)."""
        xp = self.bk.xp
        bk = self.bk
        P, T = self.unpack(u, params)
        lnK0 = aux if (aux is not None and self.model.flash.warm_start) else None
        phi, S, lnK, _ = self.model.nodes(P, T, params, params["m"], lnK0)
        P_ref, h_ref, L = params["P_ref"], params["h_ref"], params["L"]
        if self.scheme == "fd":
            R = params["M"] @ phi - S
            R = R * xp.asarray([L / P_ref, L / h_ref])[None, :]
        else:
            # balanço da célula (i-1, i) atribuído ao nó i (direita); nos troços interiores e com P na
            # saída, a linha de momento usa a célula (i, i+1) atribuída ao nó i (esquerda)
            dxr = params["dx"][:, None]
            R_right = phi - xp.roll(phi, 1, axis=0) - 0.5 * dxr * (S + xp.roll(S, 1, axis=0))
            dxl = xp.roll(params["dx"], -1)[:, None]
            R_left = xp.roll(phi, -1, axis=0) - phi - 0.5 * dxl * (xp.roll(S, -1, axis=0) + S)
            R = R_right
            if self.outlet_P:
                R = bk.set(R, (slice(None), I_P), R_left[:, I_P])
            R = R / xp.asarray([P_ref, h_ref])[None, :]
        bc = self.boundary_rows(P, T, params)
        R = xp.where(bc["mask"], bc["values"], R)
        return (R.reshape(-1), lnK) if aux is not None else R.reshape(-1)

    def boundary_rows(self, P, T, params):
        """Linhas de contorno e de continuidade (ver docstring do módulo)."""
        xp = self.bk.xp
        bk = self.bk
        n = self.n_total
        mask = xp.zeros((n, N_VARS), dtype=bool)
        vals = xp.zeros((n, N_VARS))
        P_ref, T_ref = params["P_ref"], params["T_ref"]
        for k, i0 in enumerate(self.first_nodes):
            if k == 0:
                rows = {I_T: (T[i0] - params["T_in"]) / T_ref}
                if not self.outlet_P:
                    rows[I_P] = (P[i0] - params["P_in"]) / P_ref
            else:
                j = i0 - 1
                rows = {I_T: (T[i0] - T[j]) / T_ref}
                if not self.outlet_P or self.scheme == "fd":
                    rows[I_P] = (P[i0] - P[j]) / P_ref
            for v, val in rows.items():
                mask = bk.set(mask, (i0, v), True)
                vals = bk.set(vals, (i0, v), val)
        if self.outlet_P:
            if self.scheme == "fv":
                # com atribuição à esquerda, a continuidade de P fica no último nó de cada troço interior
                for j in self.last_nodes[:-1]:
                    mask = bk.set(mask, (j, I_P), True)
                    vals = bk.set(vals, (j, I_P), (P[j + 1] - P[j]) / P_ref)
            last = self.n_total - 1
            mask = bk.set(mask, (last, I_P), True)
            vals = bk.set(vals, (last, I_P), (P[last] - params["P_out"]) / P_ref)
        return dict(mask=mask, values=vals)

    # --- estimativa inicial, esparsidade, limites ---------------------------------------------
    def initial_guess(self, params, P=None, T=None):
        """Perfis P(x), T(x) nos nós (arrays) ou planos; devolve (u0, ln K por nó)."""
        xp = self.bk.xp
        n = self.n_total
        P = xp.full((n,), params["P_ref"]) if P is None else xp.asarray(P, float)
        T = xp.full((n,), params["T_ref"]) if T is None else xp.asarray(T, float)
        zz = xp.broadcast_to(xp.asarray(self.model.z)[None, :], (n, self.model.cs.n))
        st = phase_state(T, P, zz, self.model.cs, self.model.flash, self.bk)
        return self.pack(P, T, params), st.lnK

    def sparsity(self):
        return band_colors(self.n_total, N_VARS, self.width)

    def bounds(self, params):
        xp = self.bk.xp
        lo = xp.tile(xp.asarray([1e4, 150.0]) / params["scale"], self.n_total)
        hi = xp.tile(xp.asarray([1e9, 1500.0]) / params["scale"], self.n_total)
        return lo, hi
