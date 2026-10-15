"""Modelo mecanicista com escorregamento (§2.2.1–2.2.3, §3.2): incógnitas m_G, m_L, P e T em cada
nó; equações de massa por fase com a transferência de massa à taxa do equilíbrio (25), momento da
mistura com o termo dissipativo F do padrão identificado (179) e energia da mistura (182), escritas
como resíduos (188–190) com derivadas de 2.ª ordem (Tabela 4, equivalentes ao `numpy.gradient`
com `edge_order`). O holdup vem dos fechamentos mecanicistas (§2.4.1) em cada avaliação, com o
flash a fornecer a repartição de massa, as propriedades de fase e as derivadas de w_G
(`mass_split="ode"`, a dissertação) ou, como melhoria, a relação algébrica m_G = m w_G(P, T)
(`mass_split="algebraic"`), com que as equações de massa deixam de ser diferenciais.

Malha: nós por troço do tubo (uniformes dentro de cada troço), pontos de quebra como nós
partilhados por continuidade (m_G, m_L, P, T iguais); condições de contorno na entrada (P, T, m) ou
T e m na entrada com P na saída. Layout do vetor de incógnitas: por troço, nó a nó, [m_G, m_L, P, T]
escalados por [m, m, P_ref, T_ref]. Uma implementação para numpy e JAX."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend import NUMPY
from ..closures import ClosureOptions, FlowInputs, flow_closure
from ..closures.common import G, WALL_FRICTION_MULT
from ..closures.friction import fanning, vabs
from ..thermo import ComponentSet, FlashOptions, phase_state

N_VARS = 4
I_MG, I_ML, I_P, I_T = 0, 1, 2, 3


@dataclass(frozen=True)
class MechanisticOptions:
    mass_split: str = "ode"  # ode (eq. 25) | algebraic (m_G = m w_G)
    mass_basis: str = (
        "mass"  # base da fração vaporizada em (25): mass | mole (volume não é dimensionalmente admissível)
    )
    edge_order: int = 2  # ordem das diferenças nos extremos (Tabela 4: 2; `numpy.gradient` por omissão: 1)
    fd_step_dP: float = 1000.0  # derivadas de w em (25) por diferenças (backend sem AD), como 169–170
    fd_step_dT: float = 0.01
    single_phase_blend: bool = True  # F de fase única (Churchill no fluido presente) onde w_G -> 0 ou 1


def gradient_matrix(x, edge_order=2):
    """Operador de derivada (n, n) de 2.ª ordem numa malha não uniforme (o que `numpy.gradient` faz);
    nos extremos, avançada/recuada de ordem `edge_order`."""
    x = np.asarray(x, float)
    n = x.size
    M = np.zeros((n, n))
    h = np.diff(x)
    for i in range(1, n - 1):
        hl, hr = h[i - 1], h[i]
        M[i, i - 1] = -hr / (hl * (hl + hr))
        M[i, i] = (hr - hl) / (hl * hr)
        M[i, i + 1] = hl / (hr * (hl + hr))
    if edge_order == 1 or n < 3:
        M[0, 0], M[0, 1] = -1.0 / h[0], 1.0 / h[0]
        M[-1, -2], M[-1, -1] = -1.0 / h[-1], 1.0 / h[-1]
    else:
        h0, h1 = h[0], h[1]
        M[0, 0] = -(2.0 * h0 + h1) / (h0 * (h0 + h1))
        M[0, 1] = (h0 + h1) / (h0 * h1)
        M[0, 2] = -h0 / (h1 * (h0 + h1))
        hm, hn = h[-2], h[-1]
        M[-1, -3] = hn / (hm * (hm + hn))
        M[-1, -2] = -(hm + hn) / (hm * hn)
        M[-1, -1] = (2.0 * hn + hm) / (hn * (hm + hn))
    return M


class MechanisticModel:
    """Resíduos escalados F(u, params) = 0 do sistema discretizado; `u` é o vetor plano de todos os
    troços. A estrutura (nós por troço, condições de contorno) é estática; os parâmetros do caso
    (malha, operador de derivada, tubo por nó, entrada/saída, escalas) vão em `params`
    (`params_from`), para que uma só compilação sirva todos os casos com a mesma estrutura."""

    def __init__(
        self,
        cs: ComponentSet,
        z,
        n_seg: tuple,
        outlet_P: bool,
        flash: FlashOptions,
        closures: ClosureOptions,
        opts: MechanisticOptions,
        bk=NUMPY,
    ):
        self.cs, self.z, self.bk = cs, np.asarray(z, float), bk
        self.flash, self.closures, self.opts = flash, closures, opts
        self.n_seg = tuple(int(n) for n in n_seg)
        self.offsets = np.concatenate([[0], np.cumsum(self.n_seg)])
        self.n_total = int(self.offsets[-1])
        self.first_nodes = self.offsets[:-1].tolist()
        self.last_node = self.n_total - 1
        self.outlet_P = bool(outlet_P)

    # --- estrutura e parâmetros ------------------------------------------------------------
    @staticmethod
    def structure(pipe, n_nodes: int):
        """Nós por troço (>= 3 cada, proporcionais ao comprimento) e malha x (concatenada)."""
        L = np.diff(pipe.x_break)
        counts = np.maximum(3, np.round(n_nodes * L / L.sum()).astype(int) + 1)
        x_seg = [np.linspace(pipe.x_break[k], pipe.x_break[k + 1], counts[k]) for k in range(pipe.n_segments)]
        return tuple(int(c) for c in counts), x_seg

    @classmethod
    def params_from(cls, pipe, inlet, outlet, n_nodes, cs, z, flash, opts: MechanisticOptions, bk=NUMPY):
        xp = bk.xp
        counts, x_seg = cls.structure(pipe, n_nodes)
        n_total = int(sum(counts))
        offsets = np.concatenate([[0], np.cumsum(counts)])
        Mb = np.zeros((n_total, n_total))
        for k, xs in enumerate(x_seg):
            a, b = offsets[k], offsets[k + 1]
            Mb[a:b, a:b] = gradient_matrix(xs, opts.edge_order)
        seg_of_node = np.concatenate([np.full(n, k) for k, n in enumerate(counts)])
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
            D=arr["D"][seg_of_node],
            roughness=arr["roughness"][seg_of_node],
            theta=arr["theta"][seg_of_node],
            U=arr["U"][seg_of_node],
            T_amb=arr["T_amb"][seg_of_node],
            m=m,
            P_ref=P_ref,
            T_ref=T_ref,
            scale=np.array([m, m, P_ref, T_ref]),
            h_ref=max(abs(h_in), 1e5),
            L=float(pipe.L),
            P_in=float(inlet.get("P", P_ref)),
            T_in=T_ref,
            mG_in=float(inlet.get("m_G", m * st_in.w_G[0])),
            P_out=float(outlet.get("P", P_ref)),
        )
        return {k: xp.asarray(v, float) for k, v in p.items()}

    # --- estado -----------------------------------------------------------------------------
    def unpack(self, u, params):
        U = u.reshape(self.n_total, N_VARS) * params["scale"][None, :]
        return U[:, I_MG], U[:, I_ML], U[:, I_P], U[:, I_T]

    def pack(self, mG, mL, P, T, params):
        xp = self.bk.xp
        return (xp.stack([mG, mL, P, T], axis=1) / params["scale"][None, :]).reshape(-1)

    def node_state(self, mG, mL, P, T, params):
        """Flash, fechamento e termos por nó. Devolve dict de arrays (n,)."""
        xp = self.bk.xp
        bk = self.bk
        n = P.shape[0]
        zz = xp.broadcast_to(xp.asarray(self.z)[None, :], (n, self.cs.n))
        st = phase_state(T, P, zz, self.cs, self.flash, bk)
        D = params["D"]
        A = xp.pi * D**2 / 4.0
        v_SG = mG / (st.rho_V * A)
        v_SL = mL / (st.rho_L * A)
        inp = FlowInputs.build(
            bk=bk,
            D=D,
            eps=params["roughness"],
            theta=params["theta"],
            rho_L=st.rho_L,
            rho_G=st.rho_V,
            mu_L=st.mu_L,
            mu_G=st.mu_V,
            sigma=xp.maximum(st.sigma, 1e-4),
            v_SL=xp.maximum(v_SL, 1e-9),
            v_SG=xp.maximum(v_SG, 1e-9),
            f_wall_mult=WALL_FRICTION_MULT[self.closures.wall_friction],
        )
        res, w = flow_closure(inp, self.closures, bk)
        # velocidades de fase do próprio fechamento (v_L H_L = v_SL, v_G (1 - H_L) = v_SG): o fluxo de
        # momento escreve-se m_k v_k / A, suave quando uma fase some (m_k -> 0 com v_k finito);
        # escrevê-lo como rho H v^2 = rho v_S^2 / H com H num piso absoluto dava uma parábola de
        # curvatura 1/H nos nós monofásicos e uma Jacobiana quase singular (o problema de
        # convergência que a dissertação relata para o aparecimento e desaparecimento de fase)
        HL = xp.clip(res.H_L, 1e-9, 1.0 - 1e-9)
        vL = res.v_L
        vG = res.v_G
        F = res.F
        if self.opts.single_phase_blend:
            # mistura C1 para o atrito de fase única (fluido presente) quando w -> 0 ou 1
            wG = xp.clip(mG / (mG + mL), 0.0, 1.0)
            rho1 = xp.where(wG > 0.5, st.rho_V, st.rho_L)
            mu1 = xp.where(wG > 0.5, st.mu_V, st.mu_L)
            v1 = (mG + mL) / (rho1 * A)
            f1, _ = fanning(v1, rho1, mu1, D, params["roughness"], bk=bk)
            f1 = f1 * WALL_FRICTION_MULT[self.closures.wall_friction]
            F1 = f1 * rho1 * v1 * vabs(v1, bk) / 2.0 * 4.0 / D + rho1 * G * xp.sin(params["theta"])
            uu = xp.clip((xp.minimum(wG, 1.0 - wG) - 1e-4) / 1e-3, 0.0, 1.0)
            s = uu * uu * (3.0 - 2.0 * uu)  # 0 em fase única (< 1e-4), 1 a partir de 1,1e-3
            F = s * F + (1.0 - s) * F1
        return dict(st=st, HL=HL, vL=vL, vG=vG, F=F, v_SL=v_SL, v_SG=v_SG, w=w, closure=res, A=A)

    # --- derivadas de w_G em (25) --------------------------------------------------------------
    def dw_dPT(self, P, T):
        xp = self.bk.xp
        bk = self.bk
        n = P.shape[0]
        zz = xp.broadcast_to(xp.asarray(self.z)[None, :], (n, self.cs.n))

        def w_of(PP, TT):
            st = phase_state(TT, PP, zz, self.cs, self.flash, bk)
            return st.w_G if self.opts.mass_basis == "mass" else st.beta

        if bk.has_ad:
            wP = bk.jvp(lambda PP: w_of(PP, T), (P,), (xp.ones_like(P),))[1]
            wT = bk.jvp(lambda TT: w_of(P, TT), (T,), (xp.ones_like(T),))[1]
        else:
            dP, dT = self.opts.fd_step_dP, self.opts.fd_step_dT
            w0 = w_of(P, T)
            wP = (w0 - w_of(P - dP, T)) / dP
            wT = (w0 - w_of(P, T - dT)) / dT
        return wP, wT

    # --- resíduos ----------------------------------------------------------------------------
    def residual(self, u, params):
        xp = self.bk.xp
        mG, mL, P, T = self.unpack(u, params)
        ns = self.node_state(mG, mL, P, T, params)
        st = ns["st"]
        M = params["M"]
        m = params["m"]
        L = params["L"]
        A = ns["A"]
        # massas (175–178)
        if self.opts.mass_split == "ode":
            wP, wT = self.dw_dPT(P, T)
            MGL = m * (wP * (M @ P) + wT * (M @ T))  # (25) em base mássica: m dw/dx
            r_mG = M @ mG - MGL
            r_mL = M @ mL + MGL
        else:
            w = st.w_G if self.opts.mass_basis == "mass" else st.beta
            r_mG = (mG - m * w) / L  # algébrica, na mesma escala (1/m)
            r_mL = (mL - (m - mG)) / L
        # momento (179–181): rho_L H_L v_L^2 + rho_G H_G v_G^2 + P = (m_L v_L + m_G v_G)/A + P
        Y_mom = (mL * ns["vL"] + mG * ns["vG"]) / A + P
        r_mom = M @ Y_mom + ns["F"]
        # energia (182–184): fluxo por unidade de área
        Y_en = (mL * st.h_L + mG * st.h_V) / A
        Qw = -params["U"] * xp.pi * params["D"] * (T - params["T_amb"])
        r_en = M @ Y_en - (Qw / A - G * xp.sin(params["theta"]) * m / A)
        h_ref = params["h_ref"]
        R = xp.stack(
            [r_mG * L / m, r_mL * L / m, r_mom * L / params["P_ref"], r_en * L / (m * h_ref / A[0])],
            axis=1,
        )
        bc = self.boundary_rows(mG, mL, P, T, params)
        R = xp.where(bc["mask"], bc["values"], R)
        return R.reshape(-1)

    def boundary_rows(self, mG, mL, P, T, params):
        """Linhas de contorno: no primeiro nó do troço 0, u - u_spec; nos primeiros nós dos outros
        troços, continuidade com o último nó do troço anterior; com P na saída, a linha de P do
        primeiro nó fica com o resíduo de momento e o último nó recebe P - P_out."""
        xp = self.bk.xp
        bk = self.bk
        n = self.n_total
        mask = xp.zeros((n, N_VARS), dtype=bool)
        vals = xp.zeros((n, N_VARS))
        m, P_ref, T_ref = params["m"], params["P_ref"], params["T_ref"]
        for k, i0 in enumerate(self.first_nodes):
            if k == 0:
                mG_spec = params["mG_in"]
                rows = {
                    I_MG: (mG[i0] - mG_spec) / m,
                    I_ML: (mL[i0] - (m - mG_spec)) / m,
                    I_T: (T[i0] - params["T_in"]) / T_ref,
                }
                if not self.outlet_P:
                    rows[I_P] = (P[i0] - params["P_in"]) / P_ref
            else:
                j = i0 - 1
                rows = {
                    I_MG: (mG[i0] - mG[j]) / m,
                    I_ML: (mL[i0] - mL[j]) / m,
                    I_P: (P[i0] - P[j]) / P_ref,
                    I_T: (T[i0] - T[j]) / T_ref,
                }
            for v, val in rows.items():
                mask = bk.set(mask, (i0, v), True)
                vals = bk.set(vals, (i0, v), val)
        if self.outlet_P:
            mask = bk.set(mask, (self.last_node, I_P), True)
            vals = bk.set(vals, (self.last_node, I_P), (P[self.last_node] - params["P_out"]) / P_ref)
        return dict(mask=mask, values=vals)

    # --- estimativa inicial, esparsidade, limites ---------------------------------------------
    def initial_guess(self, params, P=None, T=None):
        """Perfis P(x), T(x) (arrays nos nós) ou planos; m_G da repartição do flash."""
        xp = self.bk.xp
        n = self.n_total
        P = xp.full((n,), params["P_ref"]) if P is None else xp.asarray(P, float)
        T = xp.full((n,), params["T_ref"]) if T is None else xp.asarray(T, float)
        zz = xp.broadcast_to(xp.asarray(self.z)[None, :], (n, self.cs.n))
        st = phase_state(T, P, zz, self.cs, self.flash, self.bk)
        mG = params["m"] * st.w_G
        return self.pack(mG, params["m"] - mG, P, T, params)

    def sparsity(self):
        """Cores e máscara: o resíduo de um nó usa os nós a distância <= 2 (derivadas de 2.ª ordem
        nos extremos), e as linhas de continuidade ligam nós adjacentes."""
        from ..solvers.newton import band_colors

        return band_colors(self.n_total, N_VARS, 2)

    def bounds(self, params):
        xp = self.bk.xp
        lo = xp.tile(xp.asarray([-0.01, -0.01, 1e4]) / xp.asarray([1.0, 1.0, params["P_ref"]]), 1)
        lo = xp.tile(xp.concatenate([lo, xp.asarray([150.0]) / params["T_ref"]]), self.n_total)
        hi = xp.tile(
            xp.concatenate([xp.asarray([1.01, 1.01, 1e4]), xp.asarray([1500.0]) / params["T_ref"]]), self.n_total
        )
        return lo, hi
