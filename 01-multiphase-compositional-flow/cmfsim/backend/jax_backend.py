"""Copiado de mfsim/backend (doutorado, 2026-10-14, commit 535e6f8) em 2026-10-15; alterações: cache em CMFSIM_JAX_CACHE; registo de pytrees do mfsim removido (o cmfsim regista as suas dataclasses com `JAX.register_container` onde as define).

Backend JAX: float64, cache persistente de compilação, pytrees para as dataclasses de
arrays, e as mesmas primitivas do numpy em forma traçável."""

from __future__ import annotations

import contextlib
import os
from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .base import Backend

jax.config.update("jax_enable_x64", True)
# cache persistente: a compilação (1-5 s por modelo) é feita uma vez por máquina
_cache_dir = os.environ.get("CMFSIM_JAX_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "cmfsim_jax"))
try:
    os.makedirs(_cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:  # pragma: no cover
    pass


@jax.custom_jvp
def _sigmoid(z):
    # a mesma fórmula do numpy (valores bit-iguais); a tangente s (1 - s) evita o NaN do
    # exp(clip) encostado ao corte
    return 1.0 / (1.0 + jnp.exp(jnp.clip(-z, -700.0, 700.0)))


@_sigmoid.defjvp
def _sigmoid_jvp(primals, tangents):
    (z,), (t,) = primals, tangents
    s = _sigmoid(z)
    return s, s * (1.0 - s) * t


class JaxBackend(Backend):
    name = "jax"
    is_jax = True
    has_ad = True
    xp = jnp

    def asarray(self, x, dtype=float):
        return jnp.asarray(x, dtype)

    def to_numpy(self, x):
        return np.asarray(x)

    def set(self, a, idx, v):
        return jnp.asarray(a).at[idx].set(v)

    def add(self, a, idx, v):
        return jnp.asarray(a).at[idx].add(v)

    def while_loop(self, cond, body, init):
        return lax.while_loop(cond, body, init)

    def fori_loop(self, lo, hi, body, init):
        return lax.fori_loop(lo, hi, body, init)

    def stop_gradient(self, x):
        return lax.stop_gradient(x)

    def jvp(self, f, primals, tangents):
        return jax.jvp(f, primals, tangents)

    def safe_sqrt(self, x):
        pos = x > 0
        return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0)

    def safe_pow(self, x, p):
        pos = x > 0
        return jnp.where(pos, jnp.where(pos, x, 1.0) ** p, 0.0 if p > 0 else jnp.inf)

    def sigmoid(self, z):
        return _sigmoid(z)

    def robust_inv(self, A):
        invA = jnp.linalg.inv(A)
        bad = ~jnp.all(jnp.isfinite(invA), axis=(1, 2))
        pinvA = jnp.linalg.pinv(A, rtol=1e-12)
        return jnp.where(bad[:, None, None], pinvA, invA)

    def masked_apply(self, fn, mask, inp, out_cls):
        sub = fn(inp)
        return out_cls(**{f.name: jnp.where(mask, getattr(sub, f.name), 0.0) for f in fields(out_cls)})

    def errstate(self, **kw):
        return contextlib.nullcontext()

    def register_container(self, cls):
        names = [f.name for f in fields(cls)]
        try:
            jax.tree_util.register_dataclass(cls, data_fields=names, meta_fields=[])
        except ValueError:
            pass  # já registada
        return cls


JAX = JaxBackend()


# dataclasses de arrays do pacote como pytrees (uma vez, ao carregar o backend)
def _register_containers():
    from ..thermo.state import PhaseState

    for cls in (PhaseState,):
        JAX.register_container(cls)


_register_containers()


def _register_closure_containers():
    from ..closures.common import ClosureResult, FlowInputs

    for cls in (FlowInputs, ClosureResult):
        JAX.register_container(cls)


_register_closure_containers()
