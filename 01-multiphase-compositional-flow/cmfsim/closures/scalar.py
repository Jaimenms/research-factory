"""Resolvedor escalar vetorizado para as equações implícitas de holdup (41, 59, 79, 82):
bisseção num intervalo com mudança de sinal (número fixo de iterações, sem ramos em valores)
seguida de polimento de Newton; o resultado da bisseção entra com `stop_gradient`, de modo que
em JAX a derivada da raiz em relação aos parâmetros vem do passo de Newton (teorema da função
implícita), exata e barata."""

from __future__ import annotations

from ..backend import NUMPY


def solve_bracketed(residual, lo, hi, bk=NUMPY, n_bisect=30, n_newton=3, fd_step=1e-7):
    """Raiz de residual(h) = 0 com residual (n,) -> (n,), lo e hi arrays (n,) com sinais opostos
    (se não houver mudança de sinal devolve o extremo mais próximo da raiz). A derivada para o
    Newton é por `jvp` (JAX) ou diferenças centrais (numpy)."""
    xp = bk.xp
    lo = xp.asarray(lo, float)
    hi = xp.asarray(hi, float)
    lo0, hi0 = lo, hi
    f_lo = residual(lo)
    s_lo = f_lo > 0.0

    def body(i, c):
        lo, hi, s_lo = c
        mid = 0.5 * (lo + hi)
        s_mid = residual(mid) > 0.0
        same = s_mid == s_lo
        return xp.where(same, mid, lo), xp.where(same, hi, mid), s_lo

    f_hi = residual(hi)
    no_change = (f_hi > 0.0) == s_lo
    lo, hi, _ = bk.fori_loop(0, n_bisect, body, (lo, hi, s_lo))
    h = 0.5 * (lo + hi)
    # sem mudança de sinal não há raiz no intervalo: devolve o extremo com menor |resíduo|
    h = xp.where(no_change, xp.where(xp.abs(f_lo) < xp.abs(f_hi), lo0, hi0), h)
    h = bk.stop_gradient(h)
    for _ in range(n_newton):
        f = residual(h)
        if bk.has_ad:
            df = bk.jvp(residual, (h,), (xp.ones_like(h),))[1]
        else:
            d = fd_step * xp.maximum(1.0, xp.abs(h))
            df = (residual(h + d) - residual(h - d)) / (2.0 * d)
        df = xp.where(xp.abs(df) < 1e-300, 1e-300, df)
        step = f / df
        step = xp.where(xp.isfinite(step), step, 0.0)
        h = xp.where(no_change, h, xp.clip(h - step, lo0, hi0))
    return h
