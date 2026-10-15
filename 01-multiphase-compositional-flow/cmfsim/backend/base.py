"""Copiado de mfsim/backend (doutorado, 2026-10-14, commit 535e6f8) em 2026-10-15; alterações: nenhuma.

Interface do backend (classe base abstrata) partilhada por numpy e JAX."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, replace


class Backend(ABC):
    name: str = "base"
    is_jax: bool = False
    has_ad: bool = False
    xp = None  # numpy | jax.numpy

    # --- construção / conversão -------------------------------------------
    @abstractmethod
    def asarray(self, x, dtype=float): ...

    @abstractmethod
    def to_numpy(self, x): ...

    # --- atualização funcional ----------------------------------------------
    @abstractmethod
    def set(self, a, idx, v):
        """Cópia de `a` com `a[idx] = v` (numpy: copia; JAX: `a.at[idx].set(v)`)."""

    @abstractmethod
    def add(self, a, idx, v):
        """Cópia de `a` com `a[idx] += v`."""

    # --- laços ----------------------------------------------------------------
    @abstractmethod
    def while_loop(self, cond, body, init):
        """`while cond(c): c = body(c)`; `c` é um tuplo de arrays de forma fixa."""

    @abstractmethod
    def fori_loop(self, lo, hi, body, init):
        """`for i in range(lo, hi): c = body(i, c)`."""

    # --- diferenciação ----------------------------------------------------------
    @abstractmethod
    def stop_gradient(self, x): ...

    def jvp(self, f, primals, tangents):
        raise NotImplementedError(f"o backend {self.name} não tem diferenciação automática")

    # --- funções com derivada segura ----------------------------------------
    @abstractmethod
    def safe_sqrt(self, x):
        """sqrt(x) para x >= 0, com derivada finita em 0 (0 para x <= 0)."""

    @abstractmethod
    def safe_pow(self, x, p):
        """x**p para x >= 0 com derivada finita em 0 (0 se x <= 0 e p > 0; inf se p <= 0)."""

    @abstractmethod
    def sigmoid(self, z):
        """1/(1 + exp(-z)) com o corte de |z| <= 700 do numpy (valores iguais nos dois backends)."""

    @abstractmethod
    def robust_inv(self, A):
        """Inversa de matrizes (n, m, m); as singulares (inv não finita) recebem a pseudo-inversa."""

    # --- avaliação sob máscara ---------------------------------------------------
    @abstractmethod
    def masked_apply(self, fn, mask, inp, out_cls):
        """`out_cls.zeros(n)` com `fn(inp)` nos pontos `mask` (numpy: avalia só no subconjunto,
        como o `blend_closures` de hoje; JAX: avalia em todos e mascara com `where`)."""

    # --- contexto numpy ---------------------------------------------------------
    @abstractmethod
    def errstate(self, **kw): ...

    def register_container(self, cls):
        """Regista uma dataclass de arrays como pytree (JAX); no numpy não faz nada."""
        return cls
