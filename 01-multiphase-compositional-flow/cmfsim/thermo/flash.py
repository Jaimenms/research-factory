"""Flash P-T de líquido e vapor (eqs. 113–125 da dissertação): Rachford–Rice (118) com Newton
salvaguardado por bisseção, substituições sucessivas K_i <- K_i f_L,i/f_V,i (124) a partir da
estimativa de Wilson (125), critério de paragem (123). Como melhoria com bandeira, um
polimento de Newton no sistema completo G(ln K, beta) = 0 depois das substituições: além de
levar a solução a precisão de máquina em duas iterações, é ele que fornece as derivadas
exatas d(beta, x, y)/d(P, T, z) pelo teorema da função implícita (o resultado das
substituições entra com `stop_gradient` e o passo de Newton transporta a sensibilidade).
Uma implementação para numpy e JAX (`bk`), vetorizada em pontos.

Flash negativo: beta pode sair de [0, 1] dentro dos limites (1/(1 - K_max), 1/(1 - K_min));
fora de [0, 1] a mistura é monofásica, beta é truncado, a fase presente tem a composição
global e a fase ausente fica com a composição incipiente (normalizada), o que mantém todas
as propriedades definidas e contínuas até à fronteira de fases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend import NUMPY
from .components import ComponentSet
from .pr import ln_phi


@dataclass(frozen=True)
class FlashOptions:
    tol: float = 1e-6  # critério (123) das substituições sucessivas: soma de (f_L/f_V - 1)^2; o polimento de
    # Newton leva depois a 1e-26 (tol = 1e-12 e polish_iter = 0 reproduzem a dissertação tal e qual);
    # 1e-3 + Newton é 3x mais rápido do que 1e-6 + Newton com o mesmo resultado (issue 007)
    max_iter: int = 200  # substituições sucessivas
    rr_tol: float = 1e-14
    rr_max_iter: int = 60
    polish_iter: int = 30  # máximo de iterações de Newton amortecido no sistema completo (0 = só a dissertação)
    polish_tol: float = 1e-26  # paragem do polimento (mesmo critério 123)
    fd_step: float = 1e-7  # passo relativo das diferenças finitas da Jacobiana (backend sem AD)
    stability_test: bool = False  # teste de estabilidade de Michelsen (TPD) antes do flash: melhoria com bandeira
    stability_iter: int = 40
    warm_start: bool = True  # os modelos passam o K convergido do ponto anterior como estimativa inicial (issue 007)


def wilson_K(T, P, cs: ComponentSet, bk=NUMPY):
    """Estimativa inicial de Wilson (125): K_i = Pc_i/P exp(5.37 (1 + w_i)(1 - Tc_i/T)); (n, nc)."""
    xp = bk.xp
    T = xp.asarray(T, float)[:, None]
    P = xp.asarray(P, float)[:, None]
    Pc = xp.asarray(cs.Pc)[None, :]
    Tc = xp.asarray(cs.Tc)[None, :]
    w = xp.asarray(cs.omega)[None, :]
    return Pc / P * xp.exp(5.37 * (1.0 + w) * (1.0 - Tc / T))


def rr_bounds(K, bk=NUMPY):
    """Limites do flash negativo: beta em (1/(1 - K_max), 1/(1 - K_min)); sem raiz possível
    (todos os K de um lado de 1) o limite correspondente é +-1e3."""
    xp = bk.xp
    Kmax = xp.max(K, axis=1)
    Kmin = xp.min(K, axis=1)
    lo = xp.where(Kmax > 1.0, 1.0 / (1.0 - xp.where(Kmax > 1.0, Kmax, 2.0)), -1e3)
    hi = xp.where(Kmin < 1.0, 1.0 / (1.0 - xp.where(Kmin < 1.0, Kmin, 0.5)), 1e3)
    return lo, hi


def rachford_rice(z, K, bk=NUMPY, tol=1e-14, max_iter=60):
    """Resolve sum_i z_i (K_i - 1)/(1 + beta (K_i - 1)) = 0 em beta (118) por Newton com bisseção
    nos limites do flash negativo. z, K de forma (n, nc); devolve beta (n,) não truncado."""
    xp = bk.xp
    Km1 = K - 1.0
    lo0, hi0 = rr_bounds(K, bk)
    eps = 1e-10 * (hi0 - lo0)
    lo0 = lo0 + eps
    hi0 = hi0 - eps
    beta0 = xp.clip(xp.full(lo0.shape, 0.5), lo0, hi0)

    def f_df(beta):
        den = 1.0 + beta[:, None] * Km1
        den = xp.where(xp.abs(den) < 1e-300, 1e-300, den)
        f = xp.sum(z * Km1 / den, axis=1)
        df = -xp.sum(z * Km1**2 / den**2, axis=1)
        return f, df

    def cond(c):
        beta, lo, hi, it, done = c
        return xp.logical_and(xp.logical_not(xp.all(done)), it < max_iter)

    def body(c):
        beta, lo, hi, it, done = c
        f, df = f_df(beta)
        lo = xp.where(f > 0.0, beta, lo)
        hi = xp.where(f > 0.0, hi, beta)
        df_safe = xp.where(xp.abs(df) < 1e-300, -1e-300, df)
        b_newton = beta - f / df_safe
        # convergido: |f| pequeno ou passo de Newton nulo; nesse caso fica-se em beta (sem saltar
        # para o ponto médio: quando o intervalo colapsou sobre a raiz, beta é um dos extremos)
        done_n = xp.logical_or(
            done, xp.logical_or(xp.abs(f) < tol, xp.abs(b_newton - beta) < 1e-15 * xp.maximum(1.0, xp.abs(beta)))
        )
        inside = xp.logical_and(b_newton >= lo, b_newton <= hi)  # inclusivo: a raiz pode estar no extremo
        bn = xp.where(inside, b_newton, 0.5 * (lo + hi))
        bn = xp.where(done_n, beta, bn)
        return bn, lo, hi, it + 1, done_n

    init = (beta0, lo0, hi0, xp.asarray(0), xp.zeros(beta0.shape, dtype=bool))
    beta, lo, hi, it, done = bk.while_loop(cond, body, init)
    return beta


def phase_compositions(z, K, beta, bk=NUMPY, clamp=True):
    """x, y (116–117). Com `clamp`, beta é truncado a [0, 1] e a fase ausente recebe a composição
    incipiente normalizada (beta <= 0: y ~ K z; beta >= 1: x ~ z/K). Sem `clamp` (iteração do flash
    negativo) usa-se o beta tal qual: dentro dos limites de Rachford–Rice as composições são
    positivas; fora deles (sem raiz) são truncadas a um piso antes de normalizar."""
    xp = bk.xp
    b = (xp.clip(beta, 0.0, 1.0) if clamp else beta)[:, None]
    den = 1.0 + b * (K - 1.0)
    x = z / xp.where(xp.abs(den) < 1e-300, 1e-300, den)
    y = K * x
    x = xp.maximum(x, 1e-300)
    y = xp.maximum(y, 1e-300)
    x = x / xp.sum(x, axis=1, keepdims=True)
    y = y / xp.sum(y, axis=1, keepdims=True)
    return x, y


def _unpack(u, nc):
    return u[:, :nc], u[:, nc]


def _G(u, z, T, P, cs, bk):
    """Sistema completo do flash: G = [ln K - ln phi_L(x) + ln phi_V(y); sum(y - x)] (n, nc + 1),
    com x, y de (K, beta) não truncado (flash negativo) e normalizados para a EoS."""
    xp = bk.xp
    nc = cs.n
    lnK, beta = _unpack(u, nc)
    K = xp.exp(lnK)
    den = 1.0 + beta[:, None] * (K - 1.0)
    x_raw = z / den
    y_raw = K * x_raw
    x = x_raw / xp.sum(x_raw, axis=1, keepdims=True)
    y = y_raw / xp.sum(y_raw, axis=1, keepdims=True)
    g1 = lnK - ln_phi(T, P, x, cs, "liquid", bk) + ln_phi(T, P, y, cs, "vapor", bk)
    g2 = xp.sum(y_raw - x_raw, axis=1)
    return xp.concatenate([g1, g2[:, None]], axis=1)


def _jacobian_dirs(G, u, bk, fd_step):
    """J (n, nd, nd) de G(u) (n, nd) por nd tangentes (jvp no JAX; diferenças centrais no numpy)."""
    xp = bk.xp
    n, nd = u.shape
    cols = []
    for k in range(nd):
        e = xp.zeros((n, nd)).at[:, k].set(1.0) if bk.is_jax else bk.set(xp.zeros((n, nd)), (slice(None), k), 1.0)
        if bk.has_ad:
            cols.append(bk.jvp(G, (u,), (e,))[1])
        else:
            h = fd_step * xp.maximum(1.0, xp.abs(u[:, k]))[:, None]
            cols.append((G(u + h * e) - G(u - h * e)) / (2.0 * h))
    return xp.stack(cols, axis=2)


def stability_tpd(T, P, z, cs: ComponentSet, opts: FlashOptions, bk=NUMPY):
    """Teste de estabilidade de Michelsen (1982): duas fases de tentativa (vapor W = K z, líquido
    W = z/K, com K de Wilson) iteradas por substituições sucessivas na forma não normalizada
    ln W_i = ln z_i + ln phi_i(z) - ln phi_i(w). Devolve o tpd* mínimo (n,) (negativo = z instável,
    duas fases) e a estimativa de K da tentativa mais instável (n, nc)."""
    xp = bk.xp
    lnK0 = xp.log(wilson_K(T, P, cs, bk))
    d_V = ln_phi(T, P, z, cs, "vapor", bk)
    d_L = ln_phi(T, P, z, cs, "liquid", bk)
    lnz = xp.log(xp.maximum(z, 1e-300))

    def trial(lnW0, phase_W, d_ref):
        def body(i, lnW):
            W = xp.exp(lnW)
            w = W / xp.sum(W, axis=1, keepdims=True)
            return lnz + d_ref - ln_phi(T, P, w, cs, phase_W, bk)

        lnW = bk.fori_loop(0, opts.stability_iter, body, lnW0)
        W = xp.exp(lnW)
        w = W / xp.sum(W, axis=1, keepdims=True)
        # tpd* = 1 + sum W_i (ln W_i + ln phi_i(w) - ln z_i - ln phi_i(z) - 1)
        tpd = 1.0 + xp.sum(W * (lnW + ln_phi(T, P, w, cs, phase_W, bk) - lnz - d_ref - 1.0), axis=1)
        return tpd, lnW

    tpd_v, lnW_v = trial(lnz + lnK0, "vapor", d_L)  # tentativa vapor contra z como líquido
    tpd_l, lnW_l = trial(lnz - lnK0, "liquid", d_V)  # tentativa líquido contra z como vapor
    use_v = tpd_v <= tpd_l
    tpd = xp.where(use_v, tpd_v, tpd_l)
    lnK = xp.where(use_v[:, None], lnW_v - lnz, lnz - lnW_l)
    lnK = xp.where(xp.isfinite(lnK), lnK, lnK0)
    return tpd, lnK


def flash_pt(T, P, z, cs: ComponentSet, opts: FlashOptions = FlashOptions(), bk=NUMPY, lnK0=None):
    """Flash P-T. T, P (n,), z (n, nc). Devolve dict com K (n, nc), beta_raw (não truncado), beta
    (truncado), x, y (n, nc), two_phase (n,) 0/1, iterations (n,), residual (n,) (critério 123).
    `lnK0` (n, nc): estimativa inicial de ln K (arranque quente: o K convergido de um estado vizinho);
    pontos com `lnK0` não finito ou trivial (max |ln K| < 1e-3, solução trivial de um estado
    supercrítico) voltam à estimativa de Wilson. Com arranque quente o teste de estabilidade não
    corre (o flash negativo transporta a solução através da fronteira de fases)."""
    xp = bk.xp
    T = xp.asarray(T, float)
    P = xp.asarray(P, float)
    z = xp.asarray(z, float)
    nc = cs.n
    lnK_w = xp.log(wilson_K(T, P, cs, bk))
    warm = lnK0 is not None
    if warm:
        lnK0 = xp.asarray(lnK0, float)
        usable = xp.logical_and(xp.all(xp.isfinite(lnK0), axis=1), xp.max(xp.abs(lnK0), axis=1) > 1e-3)
        lnK0 = xp.where(usable[:, None], lnK0, lnK_w)
    else:
        lnK0 = lnK_w
    if opts.stability_test and not warm:
        # z instável: parte do K da fase de tentativa (converge para a solução bifásica certa);
        # z estável: fica com Wilson e o flash negativo decide a fase
        tpd, lnK_st = stability_tpd(T, P, z, cs, opts, bk)
        unstable = tpd < -1e-8
        lnK0 = xp.where(unstable[:, None], lnK_st, lnK0)

    def ss_step(lnK):
        K = xp.exp(lnK)
        beta = rachford_rice(z, K, bk, opts.rr_tol, opts.rr_max_iter)
        x, y = phase_compositions(z, K, beta, bk, clamp=False)  # flash negativo: beta tal qual
        lphiL = ln_phi(T, P, x, cs, "liquid", bk)
        lphiV = ln_phi(T, P, y, cs, "vapor", bk)
        ratio = xp.exp(lphiL - lnK - lphiV)  # f_L/f_V com o K usado em x, y
        res = xp.sum((ratio - 1.0) ** 2, axis=1)
        return lphiL - lphiV, beta, res

    def cond(c):
        lnK, beta, it, res, n_it = c
        return xp.logical_and(xp.any(res > opts.tol), it < opts.max_iter)

    def body(c):
        lnK, beta, it, res, n_it = c
        lnK_new, beta_new, res_new = ss_step(lnK)
        keep = res <= opts.tol  # pontos já convergidos ficam parados
        lnK_out = xp.where(keep[:, None], lnK, lnK_new)
        return (
            lnK_out,
            xp.where(keep, beta, beta_new),
            it + 1,
            xp.where(keep, res, res_new),
            xp.where(keep, n_it, n_it + 1),
        )

    init = (lnK0, xp.full(T.shape, 0.5), xp.asarray(0), xp.full(T.shape, xp.inf), xp.zeros(T.shape, dtype=int))
    lnK, beta, it, res, iterations = bk.while_loop(cond, body, init)
    lnK = bk.stop_gradient(lnK)
    beta = bk.stop_gradient(beta)

    # polimento de Newton amortecido no sistema completo (melhoria com bandeira; derivadas pelo TFI)
    if opts.polish_iter > 0:
        lo, hi = rr_bounds(xp.exp(lnK), bk)
        # só pontos com uma raiz genuína do flash negativo (longe dos polos de Rachford–Rice, onde
        # a bisseção encosta quando não há raiz)
        margin = 1e-6 * (hi - lo)
        two_phase_like = xp.logical_and(beta > lo + margin, beta < hi - margin)
        eye = xp.eye(nc + 1)[None, :, :]

        def G(uu):
            return _G(uu, z, T, P, cs, bk)

        def gnorm(g):
            return xp.sum(g * g, axis=1)

        def pcond(c):
            u, k, active, n_pol = c
            return xp.logical_and(xp.any(active), k < opts.polish_iter)

        def pbody(c):
            u, k, active, n_pol = c
            g = G(u)
            J = _jacobian_dirs(G, u, bk, opts.fd_step)
            # fora da região bifásica (ou com Jacobiana não finita) resolve-se um sistema
            # identidade com lado direito nulo: passo zero, sem matriz singular
            finite = xp.logical_and(xp.all(xp.isfinite(J), axis=(1, 2)), xp.all(xp.isfinite(g), axis=1))
            mask = xp.logical_and(active, finite)
            J_safe = xp.where(mask[:, None, None], J, eye)
            g_safe = xp.where(mask[:, None], g, 0.0)
            du = -xp.einsum("nij,nj->ni", bk.robust_inv(J_safe), g_safe)  # pseudo-inversa onde singular
            du = xp.where(xp.all(xp.isfinite(du), axis=1)[:, None], du, 0.0)
            # amortecimento: o melhor de s = 1, 1/2, 1/4 (em |G|^2); nunca pior do que ficar parado,
            # exceto num ponto já convergido à entrada (arranque quente que cai sobre a solução): aí
            # aplica-se o passo completo, nulo em valor, porque é ele que transporta a derivada
            # exata (teorema da função implícita) através do `stop_gradient` das substituições
            g0 = gnorm(g)
            converged0 = g0 <= opts.polish_tol
            best_u, best_n = u, g0
            for s in (1.0, 0.5, 0.25):
                u_s = u + s * du
                n_s = gnorm(G(u_s))
                n_s = xp.where(xp.isfinite(n_s), n_s, xp.inf)
                if s == 1.0:
                    better = xp.logical_or(n_s < best_n, converged0)
                else:
                    better = xp.logical_and(n_s < best_n, xp.logical_not(converged0))
                best_u = xp.where(better[:, None], u_s, best_u)
                best_n = xp.where(better, n_s, best_n)
            u_new = xp.where(mask[:, None], best_u, u)
            # continua só enquanto há progresso (no piso de arredondamento |G|^2 deixa de descer)
            still = xp.logical_and(mask, xp.logical_and(best_n > opts.polish_tol, best_n < g0))
            return u_new, k + 1, still, n_pol + xp.where(mask, 1, 0)

        u0 = xp.concatenate([lnK, beta[:, None]], axis=1)
        u, _, _, polish_iterations = bk.while_loop(
            pcond, pbody, (u0, xp.asarray(0), two_phase_like, xp.zeros(T.shape, dtype=int))
        )
        lnK, beta = _unpack(u, nc)
    else:
        polish_iterations = xp.zeros(T.shape, dtype=int)
    K = xp.exp(lnK)
    beta = xp.where(xp.isfinite(beta), beta, 0.5)
    # ln K a guardar para o arranque quente do vizinho: o convergido só onde o flash negativo tem
    # raiz genuína (dentro dos limites de Rachford–Rice, com margem); onde a bisseção encostou ao
    # limite (sem raiz) o K é arbitrário e deriva de chamada para chamada, e reentrar na região
    # bifásica a partir dele falha: aí guarda-se Wilson (o arranque frio)
    # solução trivial (K ~ 1: puros e misturas supercríticas): as fases são idênticas, não há
    # bifásico; Rachford–Rice tem f = 0 em todo o beta e fica no ponto de partida (0,5)
    trivial = xp.max(xp.abs(lnK), axis=1) < 1e-5
    lo_g, hi_g = rr_bounds(K, bk)
    genuine = xp.logical_and(beta > lo_g + 1e-6 * (hi_g - lo_g), beta < hi_g - 1e-6 * (hi_g - lo_g))
    genuine = xp.logical_and(genuine, xp.logical_not(trivial))
    lnK_store = xp.where(xp.logical_and(genuine, xp.all(xp.isfinite(lnK), axis=1))[:, None], lnK, lnK_w)
    two_phase_b = xp.logical_and(xp.logical_and(beta > 0.0, beta < 1.0), xp.logical_not(trivial))
    if opts.stability_test and not warm:
        # a estabilidade manda: z estável -> fase única mesmo que o flash negativo tenha raiz em (0, 1)
        two_phase_b = xp.logical_and(two_phase_b, unstable)
        beta = xp.where(unstable, beta, xp.where(beta >= 0.5, 1.0, 0.0))
    # rótulo da fase única: pelo sinal do flash negativo, exceto na solução trivial, em que decide
    # a temperatura pseudocrítica de Kay (T > sum z_i Tc_i: vapor denso)
    T_pc = z @ xp.asarray(cs.Tc)
    vapor = xp.where(trivial, T > T_pc, beta >= 0.5)
    beta_single = xp.where(vapor, 1.0, 0.0)
    beta_raw = xp.where(two_phase_b, beta, xp.where(trivial, beta_single, beta))
    beta_c = xp.where(two_phase_b, beta, beta_single)
    # resíduo (123) do estado não truncado: medida de convergência do flash (negativo incluído)
    residual = xp.sum(_G(xp.concatenate([lnK, beta[:, None]], axis=1), z, T, P, cs, bk)[:, :nc] ** 2, axis=1)
    residual = xp.where(xp.isfinite(residual), residual, xp.inf)
    x, y = phase_compositions(z, K, beta_c, bk)
    two_phase = xp.where(two_phase_b, 1.0, 0.0)
    return dict(
        K=K,
        lnK=lnK_store,
        beta_raw=beta_raw,
        beta=beta_c,
        x=x,
        y=y,
        two_phase=two_phase,
        iterations=iterations,
        polish_iterations=polish_iterations,
        residual=residual,
    )
