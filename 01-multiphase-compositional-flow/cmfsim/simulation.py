"""Caminho único de execução: `simulate(case)` lê tudo do caso (`Case`, de um JSON) e devolve
`Results` com os perfis ao longo do tubo. Modelos: `noslip` (marcha em x, §3.1). O modelo
mecanicista com escorregamento (§3.2) entra na fase 4."""

from __future__ import annotations

import time

import numpy as np

from .backend import get_backend
from .closures import ClosureOptions, FlowInputs, flow_closure, pattern_code
from .io.spec import Case
from .models.noslip import NoSlipModel, mixture_state
from .results import Results
from .solvers.rk45 import integrate
from .thermo import phase_state


def build_noslip(case: Case, bk):
    return NoSlipModel(case.cs, case.z, case.inlet["m"], case.pipe.arrays(), case.flash, case.noslip, bk=bk)


def march_noslip(case: Case, bk, x_out=None):
    """Integra o modelo de não escorregamento de x = 0 a L; devolve (x_out, U (n, 2), info)."""
    xp = bk.xp
    model = build_noslip(case, bk)
    x_out = case.pipe.output_grid(case.dx_out) if x_out is None else np.asarray(x_out, float)
    segs = case.pipe.interval_segments(x_out)
    u0 = xp.asarray([case.inlet["P"], case.inlet["T"]], float)
    scale = xp.asarray([1e5, 1.0])
    t0 = time.time()
    if bk.is_jax:
        import jax

        run = jax.jit(
            lambda u: integrate(model.rhs, u, xp.asarray(x_out), xp.asarray(segs), case.rk45, scale=scale, bk=bk)
        )
        U, info = run(u0)
        jax.block_until_ready(U)
    else:
        U, info = integrate(model.rhs, u0, x_out, segs, case.rk45, scale=scale, bk=bk)
    info = {k: int(bk.to_numpy(v)) for k, v in info.items()}
    info["cpu"] = time.time() - t0
    return x_out, bk.to_numpy(U), info


def postprocess(case: Case, x, P, T, bk):
    """Perfis derivados de (P, T): flash, holdup de não escorregamento, velocidades, padrão (só reporte)."""
    from .backend import NUMPY

    bkn = NUMPY  # pós-processamento em numpy (fora do caminho crítico)
    seg = case.pipe.segment_of(x)
    D = case.pipe.D[seg]
    eps = case.pipe.roughness[seg]
    theta = case.pipe.theta[seg]
    A = np.pi * D**2 / 4.0
    rho_mix, h_mix, mu_mix, lam_G, st = mixture_state(P, T, case.z[None, :], case.cs, case.flash, bkn)
    m = case.inlet["m"]
    v = m / (rho_mix * A)
    m_G = st.w_G * m
    m_L = (1.0 - st.w_G) * m
    v_SG = m_G / (st.rho_V * A)
    v_SL = m_L / (st.rho_L * A)
    # identificação de padrão só para reporte (Figura 37); em zonas monofásicas os critérios
    # degeneram (sigma = 0, v_SL = 0): pisos numéricos e código NaN onde two_phase = 0
    inp = FlowInputs.build(
        bk=bkn,
        D=D,
        eps=eps,
        theta=theta,
        rho_L=st.rho_L,
        rho_G=st.rho_V,
        mu_L=st.mu_L,
        mu_G=st.mu_V,
        sigma=np.maximum(st.sigma, 1e-4),
        v_SL=np.maximum(v_SL, 1e-9),
        v_SG=np.maximum(v_SG, 1e-9),
    )
    with np.errstate(all="ignore"):
        _, w = flow_closure(inp, case.closures, bkn)
    code = np.where(st.two_phase > 0, pattern_code(w, bkn), np.nan)
    prof = dict(
        z=case.pipe.elevation(x),
        D=D,
        theta=theta,
        beta=st.beta,
        w_G=st.w_G,
        two_phase=st.two_phase,
        lambda_G=lam_G,
        H_L=1.0 - lam_G,
        rho_L=st.rho_L,
        rho_V=st.rho_V,
        rho_mix=rho_mix,
        h_L=st.h_L,
        h_V=st.h_V,
        h_mix=h_mix,
        mu_L=st.mu_L,
        mu_V=st.mu_V,
        sigma=st.sigma,
        v=v,
        v_SL=v_SL,
        v_SG=v_SG,
        m_G=m_G,
        m_L=m_L,
        x_comp=st.x,
        y_comp=st.y,
        pattern_weights=w,
        pattern_code=code,
        flash_iterations=st.iterations,
    )
    return prof


def simulate(case: Case, backend=None, progress=True) -> Results:
    bk = get_backend(backend or case.backend)
    if case.model_type != "noslip" or case.solver_type != "march":
        raise NotImplementedError(f"modelo {case.model_type} com solver {case.solver_type}: fase 4")
    x, U, info = march_noslip(case, bk)
    P, T = U[:, 0], U[:, 1]
    prof = postprocess(case, x, P, T, bk)
    info.update(backend=bk.name, model=case.model_type, solver=case.solver_type, n_out=int(x.size))
    if progress:
        print(
            f"[{case.name}] {bk.name} {case.solver_type}: passos {info['n_accepted']} (+{info['n_rejected']} rejeitados) | {info['cpu']:.2f} s"
        )
    return Results(x=x, P=P, T=T, profiles=prof, info=info, case_dict=case.raw)
