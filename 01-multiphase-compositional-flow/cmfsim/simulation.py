"""Caminho único de execução: `simulate(case)` lê tudo do caso (`Case`, de um JSON) e devolve
`Results` com os perfis ao longo do tubo. Modelos: `noslip` (marcha em x, §3.1). O modelo
mecanicista com escorregamento (§3.2) entra na fase 4."""

from __future__ import annotations

import time

import numpy as np

from .backend import get_backend
from .closures import ClosureOptions, FlowInputs, flow_closure, pattern_code
from .io.spec import Case
from .models.mechanistic import MechanisticModel
from .models.noslip import NoSlipModel, mixture_state
from .solvers.newton import newton_solve
from .results import Results
from .solvers.rk45 import integrate
from .thermo import phase_state

_COMPILED = {}  # funções compiladas por estrutura (fluido, opções, malha): uma compilação serve muitos casos


def _fluid_key(case: Case):
    cs = case.cs
    return (
        (cs.names, cs.omega_a, cs.omega_b, cs.cp_temperature)
        + tuple(
            np.asarray(getattr(cs, f)).tobytes()
            for f in ("MW", "Pc", "Tc", "omega", "Tb", "Hf", "cpA", "cpB", "cpC", "cpD", "Nc_eff", "kij")
        )
        + (np.asarray(case.z).tobytes(), case.flash)
    )


def build_noslip(case: Case, bk):
    model = NoSlipModel(case.cs, case.z, case.flash, case.noslip, bk=bk)
    params = NoSlipModel.params_from(case.pipe.arrays(), case.inlet["m"], bk)
    return model, params


def march_noslip(case: Case, bk, x_out=None):
    """Integra o modelo de não escorregamento de x = 0 a L; devolve (x_out, U (n, 2), info)."""
    xp = bk.xp
    model, params = build_noslip(case, bk)
    x_out = case.pipe.output_grid(case.dx_out) if x_out is None else np.asarray(x_out, float)
    segs = case.pipe.interval_segments(x_out)
    u0 = xp.asarray([case.inlet["P"], case.inlet["T"]], float)
    scale = xp.asarray([1e5, 1.0])
    t0 = time.time()
    if bk.is_jax:
        import jax

        key = ("march", _fluid_key(case), case.noslip, case.rk45, x_out.tobytes(), segs.tobytes())
        run = _COMPILED.get(key)
        if run is None:
            xo, sg = xp.asarray(x_out), xp.asarray(segs)
            run = jax.jit(lambda u, p: integrate(model.rhs, u, xo, sg, case.rk45, scale=scale, bk=bk, args=(p,)))
            _COMPILED[key] = run
        U, info = run(u0, params)
        jax.block_until_ready(U)
    else:
        U, info = integrate(model.rhs, u0, x_out, segs, case.rk45, scale=scale, bk=bk, args=(params,))
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


def build_mechanistic(case: Case, bk):
    counts, _ = MechanisticModel.structure(case.pipe, case.n_nodes)
    model = MechanisticModel(
        case.cs, case.z, counts, bool(case.outlet), case.flash, case.closures, case.mechanistic, bk=bk
    )
    params = MechanisticModel.params_from(
        case.pipe, case.inlet, case.outlet or None, case.n_nodes, case.cs, case.z, case.flash, case.mechanistic, bk
    )
    return model, params


def solve_fd(case: Case, bk):
    """Método 3.2: diferenças finitas + Newton; devolve (modelo, params, u, info)."""
    xp = bk.xp
    model, params = build_mechanistic(case, bk)
    t0 = time.time()
    x_nodes = bk.to_numpy(params["x"])
    if case.initial_guess == "march_noslip" and not case.outlet:
        xg, Ug, _ = march_noslip(case, bk, x_out=x_nodes)
        u0 = model.initial_guess(params, P=Ug[:, 0], T=Ug[:, 1])
    else:
        u0 = model.initial_guess(params)
    colors, n_colors, mask = model.sparsity()
    if bk.is_jax:
        import jax

        key = ("fd", _fluid_key(case), case.closures, case.mechanistic, case.newton, model.n_seg, model.outlet_P)
        solve = _COMPILED.get(key)
        if solve is None:
            cc, mm = xp.asarray(colors), xp.asarray(mask)

            def solve(u, p):
                return newton_solve(
                    lambda uu: model.residual(uu, p), u, cc, n_colors, mm, case.newton, model.bounds(p), bk
                )

            solve = jax.jit(solve)
            _COMPILED[key] = solve
        u, info = solve(u0, params)
        jax.block_until_ready(u)
    else:
        u, info = newton_solve(
            lambda uu: model.residual(uu, params), u0, colors, n_colors, mask, case.newton, model.bounds(params), bk
        )
    out = {}
    for k, v in info.items():
        v = bk.to_numpy(v)
        out[k] = float(v) if k == "residual" else bool(v) if k == "converged" else int(v)
    out["cpu"] = time.time() - t0
    return model, params, bk.to_numpy(u), out


def postprocess_mechanistic(case: Case, u):
    from .backend import NUMPY

    bkn = NUMPY
    model, params = build_mechanistic(case, bkn)
    mG, mL, P, T = model.unpack(np.asarray(u), params)
    with np.errstate(all="ignore"):
        ns = model.node_state(mG, mL, P, T, params)
    st = ns["st"]
    x = np.asarray(params["x"])
    w = ns["w"]
    vol_G = st.w_G / st.rho_V
    prof = dict(
        z=case.pipe.elevation(x),
        D=np.asarray(params["D"]),
        theta=np.asarray(params["theta"]),
        beta=st.beta,
        w_G=mG / (mG + mL),
        w_G_eq=st.w_G,
        two_phase=st.two_phase,
        H_L=ns["HL"],
        H_L_noslip=1.0 - vol_G / (vol_G + (1.0 - st.w_G) / st.rho_L),
        rho_L=st.rho_L,
        rho_V=st.rho_V,
        h_L=st.h_L,
        h_V=st.h_V,
        mu_L=st.mu_L,
        mu_V=st.mu_V,
        sigma=st.sigma,
        v_L=ns["vL"],
        v_G=ns["vG"],
        v_SL=ns["v_SL"],
        v_SG=ns["v_SG"],
        m_G=mG,
        m_L=mL,
        F=ns["F"],
        x_comp=st.x,
        y_comp=st.y,
        pattern_weights=w,
        pattern_code=np.where(st.two_phase > 0, pattern_code(w, bkn), np.nan),
        flash_iterations=st.iterations,
    )
    return x, P, T, prof


def simulate(case: Case, backend=None, progress=True) -> Results:
    bk = get_backend(backend or case.backend)
    if case.model_type == "noslip" and case.solver_type == "march":
        x, U, info = march_noslip(case, bk)
        P, T = U[:, 0], U[:, 1]
        prof = postprocess(case, x, P, T, bk)
        info.update(backend=bk.name, model=case.model_type, solver=case.solver_type, n_out=int(x.size))
        if progress:
            print(
                f"[{case.name}] {bk.name} {case.solver_type}: passos {info['n_accepted']} "
                f"(+{info['n_rejected']} rejeitados) | {info['cpu']:.2f} s"
            )
    elif case.model_type == "mechanistic" and case.solver_type == "fd":
        model, params, u, info = solve_fd(case, bk)
        x, P, T, prof = postprocess_mechanistic(case, u)
        info.update(
            backend=bk.name,
            model=case.model_type,
            solver=case.solver_type,
            n_out=int(x.size),
            n_nodes=int(model.n_total),
        )
        if progress:
            print(
                f"[{case.name}] {bk.name} fd: {model.n_total} nós, Newton {info['iterations']} it, "
                f"|F| {info['residual']:.1e}, convergiu {info['converged']} | {info['cpu']:.2f} s"
            )
    else:
        raise NotImplementedError(f"modelo {case.model_type} com solver {case.solver_type}")
    return Results(x=x, P=P, T=T, profiles=prof, info=info, case_dict=case.raw)


def march_many(case: Case, variations: dict, backend="jax"):
    """Muitos casos com a mesma estrutura numa só chamada compilada e vetorizada (`vmap`): base
    `case` e `variations` = dict de arrays (N,) com chaves `D`, `roughness`, `U`, `T_amb` (por troço:
    forma (N, n_troços) ou (N,) para um tubo de um troço), `m`, `P_in`, `T_in`. Devolve (x_out,
    U (N, n_out, 2), info). É o caminho para gerar dados de treino: o custo é o do caso mais lento
    do lote, não a soma."""
    import jax
    import jax.numpy as jnp

    bk = get_backend(backend)
    xp = bk.xp
    model, params0 = build_noslip(case, bk)
    x_out = case.pipe.output_grid(case.dx_out)
    segs = case.pipe.interval_segments(x_out)
    N = len(next(iter(variations.values())))
    ns = case.pipe.n_segments

    def expand(key, default):
        v = variations.get(key)
        if v is None:
            return jnp.broadcast_to(jnp.asarray(default)[None], (N,) + jnp.asarray(default).shape)
        v = jnp.asarray(v, float)
        return jnp.broadcast_to(v[:, None], (N, ns)) if (ns > 1 and v.ndim == 1) or ns == 1 and v.ndim == 1 else v

    params = {k: expand(k, params0[k]) for k in ("D", "roughness", "theta", "U", "T_amb")}
    params["m"] = jnp.asarray(variations.get("m", np.full(N, case.inlet["m"])), float)
    P_in = jnp.asarray(variations.get("P_in", np.full(N, case.inlet["P"])), float)
    T_in = jnp.asarray(variations.get("T_in", np.full(N, case.inlet["T"])), float)
    u0 = jnp.stack([P_in, T_in], axis=1)
    xo, sg = xp.asarray(x_out), xp.asarray(segs)
    scale = xp.asarray([1e5, 1.0])
    key = ("march_many", _fluid_key(case), case.noslip, case.rk45, x_out.tobytes(), segs.tobytes())
    run = _COMPILED.get(key)
    if run is None:
        single = lambda u, p: integrate(model.rhs, u, xo, sg, case.rk45, scale=scale, bk=bk, args=(p,))
        run = jax.jit(jax.vmap(single))
        _COMPILED[key] = run
    t0 = time.time()
    U, info = run(u0, params)
    jax.block_until_ready(U)
    info = {k: np.asarray(v) for k, v in info.items()}
    info["cpu"] = time.time() - t0
    return x_out, np.asarray(U), info
