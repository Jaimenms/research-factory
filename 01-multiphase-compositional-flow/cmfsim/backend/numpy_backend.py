"""Copiado de mfsim/backend (doutorado, 2026-10-14, commit 535e6f8) em 2026-10-15; alterações: nenhuma.

Backend numpy: execução imediata com a semântica exata do código numpy anterior."""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from .base import Backend


class NumpyBackend(Backend):
    name = "numpy"
    is_jax = False
    has_ad = False
    xp = np

    def asarray(self, x, dtype=float):
        return np.asarray(x, dtype)

    def to_numpy(self, x):
        return np.asarray(x)

    def set(self, a, idx, v):
        out = np.array(a, copy=True)
        out[idx] = v
        return out

    def add(self, a, idx, v):
        out = np.array(a, copy=True)
        out[idx] = out[idx] + v
        return out

    def while_loop(self, cond, body, init):
        c = init
        while cond(c):
            c = body(c)
        return c

    def fori_loop(self, lo, hi, body, init):
        c = init
        for i in range(lo, hi):
            c = body(i, c)
        return c

    def stop_gradient(self, x):
        return x

    def safe_sqrt(self, x):
        return np.sqrt(x)

    def safe_pow(self, x, p):
        return x**p

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(np.clip(-z, -700.0, 700.0)))

    def robust_inv(self, A):
        A = np.asarray(A, float)
        n = A.shape[0]
        try:
            invA = np.linalg.inv(A)
            if not np.all(np.isfinite(invA)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            invA = np.empty_like(A)
            for i in range(n):
                try:
                    invA[i] = np.linalg.inv(A[i])
                    if not np.all(np.isfinite(invA[i])):
                        raise np.linalg.LinAlgError
                except np.linalg.LinAlgError:
                    try:
                        invA[i] = np.linalg.pinv(A[i], rcond=1e-12)
                    except np.linalg.LinAlgError:
                        invA[i] = np.nan
        return invA

    def masked_apply(self, fn, mask, inp, out_cls):
        n = inp.size
        out = out_cls.zeros(n)
        mask = np.asarray(mask, bool)
        if not mask.any():
            return out
        sub = fn(inp.subset(mask))
        return out_cls(**{f.name: self.set(getattr(out, f.name), mask, getattr(sub, f.name)) for f in fields(out_cls)})

    def errstate(self, **kw):
        return np.errstate(**kw)


NUMPY = NumpyBackend()
