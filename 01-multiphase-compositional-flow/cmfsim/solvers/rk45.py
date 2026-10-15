"""Integrador próprio em x (decisão A5): Runge–Kutta encaixado de Dormand–Prince 5(4) com passo
adaptativo, escrito sobre o backend (laços `while_loop`/`fori_loop`, sem ramos em valores) para
compilar em JAX. Integra de ponto de saída a ponto de saída (o último passo de cada intervalo
acaba exatamente no ponto), o que trata os pontos de quebra do tubo como eventos e dispensa
saída densa. O lado direito recebe o índice do intervalo de saída (`seg`): os parâmetros do
tubo são constantes dentro de cada intervalo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend import NUMPY

# coeficientes de Dormand–Prince 5(4)
_C = (0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0)
_A = (
    (),
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
)
_A_MAT = np.zeros((7, 7))
for _i, _row in enumerate(_A):
    _A_MAT[_i, : len(_row)] = _row
_B5 = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0)
_B4 = (5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0)


@dataclass(frozen=True)
class RK45Options:
    rtol: float = 1e-8
    atol: float = 1e-8  # no estado escalado (u/scale)
    first_step: float = 1.0  # m
    max_step: float = np.inf
    max_steps_per_interval: int = 10000
    safety: float = 0.9


def integrate(rhs, u0, x_out, seg_of_interval, opts: RK45Options = RK45Options(), scale=None, bk=NUMPY):
    """Integra du/dx = rhs(x, u, seg) de x_out[0] a x_out[-1]; devolve U (n_out, nu) nos pontos de saída
    e um dict com passos aceites e rejeitados. `seg_of_interval` (n_out - 1,) é o índice do troço de cada
    intervalo; `scale` (nu,) escala o erro local (tolerância relativa a |u|/scale mais atol)."""
    xp = bk.xp
    u0 = xp.asarray(u0, float)
    x_out = xp.asarray(x_out, float)
    seg_of_interval = xp.asarray(seg_of_interval)
    nu = u0.shape[0]
    n_out = x_out.shape[0]
    sc = xp.ones((nu,)) if scale is None else xp.asarray(scale, float)

    A_mat = xp.asarray(_A_MAT)
    B5 = xp.asarray(_B5)
    B4 = xp.asarray(_B4)
    C = xp.asarray(_C)

    def step(x, u, h, seg):
        """Um passo de Dormand–Prince; os estágios correm num `fori_loop` para que o lado direito
        (com o flash dentro) seja traçado e compilado uma só vez."""

        def stage(i, ks):
            ui = u + h * (A_mat[i] @ ks)
            ki = rhs(x + C[i] * h, ui, seg)
            return bk.set(ks, i, ki)

        ks = bk.fori_loop(0, 7, stage, xp.zeros((7, nu)))
        u5 = u + h * (B5 @ ks)
        u4 = u + h * (B4 @ ks)
        tol = opts.atol + opts.rtol * xp.maximum(xp.abs(u), xp.abs(u5)) / sc
        err = xp.sqrt(xp.mean(((u5 - u4) / sc / tol) ** 2))
        return u5, err

    def interval(i, carry):
        u, h, n_acc, n_rej = carry
        xa = x_out[i]
        xb = x_out[i + 1]
        seg = seg_of_interval[i]

        def cond(c):
            x, u, h, n_acc, n_rej, k = c
            return xp.logical_and(x < xb - 1e-12 * xp.maximum(1.0, xp.abs(xb)), k < opts.max_steps_per_interval)

        def body(c):
            x, u, h, n_acc, n_rej, k = c
            h_try = xp.minimum(xp.minimum(h, opts.max_step), xb - x)
            u_new, err = step(x, u, h_try, seg)
            err = xp.where(xp.isfinite(err), err, xp.inf)
            accept = err <= 1.0
            factor = xp.clip(opts.safety * xp.where(err > 0, err, 1e-16) ** (-0.2), 0.2, 5.0)
            h_next = xp.where(accept, h_try * factor, h_try * xp.minimum(factor, 0.9))
            x = xp.where(accept, x + h_try, x)
            u = xp.where(accept, u_new, u)
            return x, u, h_next, n_acc + accept, n_rej + (1 - accept), k + 1

        x, u, h, n_acc, n_rej, k = bk.while_loop(cond, body, (xa, u, h, n_acc, n_rej, xp.asarray(0)))
        return u, h, n_acc, n_rej

    U = xp.zeros((n_out, nu))
    U = bk.set(U, 0, u0)

    def outer(i, carry):
        U, u, h, n_acc, n_rej = carry
        u, h, n_acc, n_rej = interval(i, (u, h, n_acc, n_rej))
        U = bk.set(U, i + 1, u)
        return U, u, h, n_acc, n_rej

    init = (U, u0, xp.asarray(opts.first_step, float), xp.asarray(0), xp.asarray(0))
    U, u, h, n_acc, n_rej = bk.fori_loop(0, n_out - 1, outer, init)
    return U, dict(n_accepted=n_acc, n_rejected=n_rej)
