"""Newton amortecido para sistemas algébricos não lineares F(u) = 0 (método 3.2 da dissertação,
no lugar do Powell híbrido sem Jacobiana): Jacobiana por cores (grupos de colunas sem
sobreposição no padrão de esparsidade: `jvp` no JAX, diferenças centrais no numpy), resolução
densa, busca linear por bisseção em |F|², limites opcionais. Função pura sobre o backend; o número
de iterações é dinâmico (`while_loop`)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend import NUMPY


@dataclass(frozen=True)
class NewtonOptions:
    max_iter: int = 40
    tol: float = 1e-10  # em |F|_rms (resíduos já escalados)
    fd_step: float = 1e-6  # passo relativo das diferenças (backend sem AD)
    line_search: int = 6  # bisseções do passo
    min_step_fraction: float = 1e-3


def colored_jacobian(residual, u, colors, n_colors, bk=NUMPY, fd_step=1e-7):
    """J (n, n) densa a partir de `n_colors` direções: a coluna j recebe o produto J e_{colors==c}
    restrito às linhas que lhe pertencem. Exige que colunas da mesma cor não partilhem linhas
    (padrão de esparsidade conhecido); `rows_of(j)` implícito via máscara `S` não é necessário
    porque usamos o padrão `band`: linhas i com |i_nó - j_nó| <= largura (ver `band_colors`)."""
    xp = bk.xp
    n = u.shape[0]
    colors = xp.asarray(colors)
    J = xp.zeros((n, n))
    seeds = (colors[None, :] == xp.arange(n_colors)[:, None]).astype(float)  # (n_colors, n)
    if bk.has_ad:
        import jax

        V = jax.vmap(lambda s: jax.jvp(residual, (u,), (s,))[1])(seeds)  # uma só cópia traçada do resíduo
        return V, seeds
    else:
        # diferenças avançadas (um só lado dos pisos/vincos, como a diferenciação automática escolhe
        # um ramo); passo uniforme por cor nas variáveis escaladas (todas de ordem 1)
        F0 = residual(u)
        cols = [(residual(u + fd_step * seeds[c]) - F0) / fd_step for c in range(n_colors)]
    V = xp.stack(cols, axis=0)  # (n_colors, n): V[c, i] = sum_{j: colors[j]=c} J[i, j]
    return V, seeds


def assemble_from_colors(V, seeds, mask, bk=NUMPY):
    """J[i, j] = V[colors[j], i] onde mask[i, j] (padrão de esparsidade), 0 fora."""
    xp = bk.xp
    # Jfull[i, j] = sum_c seeds[c, j] V[c, i]
    Jfull = xp.einsum("cj,ci->ij", seeds, V)
    return xp.where(mask, Jfull, 0.0)


def newton_solve(residual, u0, colors, n_colors, mask, opts: NewtonOptions = NewtonOptions(), bounds=None, bk=NUMPY):
    """Resolve residual(u) = 0. `colors` (n,) inteiros, `mask` (n, n) booleana (padrão de esparsidade).
    Devolve (u, info dict com iterações, |F| final e convergência)."""
    xp = bk.xp
    u0 = xp.asarray(u0, float)
    n = u0.shape[0]

    def fnorm(F):
        return xp.sqrt(xp.mean(F * F))

    def cond(c):
        u, F, it, done = c
        return xp.logical_and(xp.logical_not(done), it < opts.max_iter)

    def body(c):
        u, F, it, done = c
        V, seeds = colored_jacobian(residual, u, colors, n_colors, bk, opts.fd_step)
        J = assemble_from_colors(V, seeds, mask, bk)
        du = -xp.linalg.solve(J, F)
        du = xp.where(xp.isfinite(du), du, 0.0)
        f0 = fnorm(F)

        # busca linear por bisseção: o primeiro s em {1, 1/2, ...} que reduz |F| (laço do backend:
        # o resíduo é traçado uma só vez)
        def ls_body(i, c):
            best_u, best_F, best_n, found = c
            s = 0.5**i
            u_s = u + s * du
            if bounds is not None:
                u_s = xp.clip(u_s, bounds[0], bounds[1])
            F_s = residual(u_s)
            n_s = fnorm(F_s)
            n_s = xp.where(xp.isfinite(n_s), n_s, xp.inf)
            better = xp.logical_and(n_s < f0, xp.logical_not(found))
            return (
                xp.where(better, u_s, best_u),
                xp.where(better, F_s, best_F),
                xp.where(better, n_s, best_n),
                xp.logical_or(found, better),
            )

        best_u, best_F, best_n, found = bk.fori_loop(0, opts.line_search, ls_body, (u, F, f0, xp.asarray(False)))
        stalled = xp.logical_not(found)
        done_n = xp.logical_or(best_n < opts.tol, stalled)
        return best_u, best_F, it + 1, done_n

    F0 = residual(u0)
    u, F, it, done = bk.while_loop(cond, body, (u0, F0, xp.asarray(0), xp.asarray(False)))
    return u, dict(iterations=it, residual=fnorm(F), converged=fnorm(F) < opts.tol)


def band_colors(n_nodes, n_vars, width):
    """Cores e máscara para um padrão em banda: o resíduo do nó i depende dos nós |i - j| <= width.
    Cor de (nó j, variável v) = (j mod (2 width + 1)) * n_vars + v; n_colors = (2 width + 1) n_vars."""
    period = 2 * width + 1
    node = np.repeat(np.arange(n_nodes), n_vars)
    var = np.tile(np.arange(n_vars), n_nodes)
    colors = (node % period) * n_vars + var
    mask = np.abs(node[:, None] - node[None, :]) <= width
    return colors, period * n_vars, mask
