"""Backend numérico injetado (copiado do padrão do mfsim/doutorado): numpy (eager) ou JAX (jit/jacfwd/vmap).

Uma só implementação da termodinâmica, dos fechamentos, dos modelos e dos solvers corre nos
dois: as funções usam `bk.xp` (numpy ou jax.numpy) para a aritmética e as primitivas do
backend para o que difere (atualização funcional `set/add`, laços `while_loop/fori_loop`,
`stop_gradient`, raízes e sigmoide com derivada estável, inversa robusta, aplicação sob
máscara, `jvp`).

Regras: funções puras de arrays e configuração estática; sem escrita in-place; `if` em Python
só sobre configuração, nunca sobre valores (usar `xp.where`); formas estáticas; `np.` só em
constantes de construção; NaN tratado com `where`; float64.
"""

from .base import Backend
from .numpy_backend import NUMPY, NumpyBackend

__all__ = ["Backend", "NUMPY", "NumpyBackend", "get_backend"]


def get_backend(name="numpy") -> Backend:
    """`"numpy"` (default) ou `"jax"` (import preguiçoso; liga float64 e a cache de compilação)."""
    if name in (None, "numpy", NUMPY):
        return NUMPY
    if isinstance(name, Backend):
        return name
    if name == "jax":
        from .jax_backend import JAX

        return JAX
    raise ValueError(f"backend desconhecido: {name}")
