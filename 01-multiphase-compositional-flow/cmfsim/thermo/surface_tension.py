"""Tensão superficial por Brock & Bird (1955) para os puros (142–143) e média molar na fase
líquida (144). Pc em bar e Tc em K dão sigma em dyn/cm; aqui convertido a N/m. Componentes
supercríticos (T >= Tc) contribuem com zero (a potência (1 - Tr)^(11/9) é truncada)."""

from __future__ import annotations

from ..backend import NUMPY
from .components import ComponentSet


def pure_surface_tension(T, cs: ComponentSet, bk=NUMPY):
    """sigma_i [N/m]; T (n,) -> (n, nc)."""
    xp = bk.xp
    T = xp.asarray(T, float)[:, None]
    Tc = xp.asarray(cs.Tc)[None, :]
    Pc_bar = xp.asarray(cs.Pc)[None, :] / 1e5
    Tbr = xp.asarray(cs.Tb)[None, :] / Tc
    Q = 0.1196 * (1.0 + Tbr * xp.log(Pc_bar / 1.01325) / (1.0 - Tbr)) - 0.279
    one_m_tr = xp.maximum(1.0 - T / Tc, 0.0)
    sigma_dyn = Pc_bar ** (2.0 / 3.0) * Tc ** (1.0 / 3.0) * Q * bk.safe_pow(one_m_tr, 11.0 / 9.0)
    return sigma_dyn * 1e-3


def surface_tension(T, x, cs: ComponentSet, bk=NUMPY):
    """sigma da mistura líquida [N/m] (144); x (n, nc)."""
    xp = bk.xp
    return xp.einsum("ni,ni->n", xp.asarray(x, float), pure_surface_tension(T, cs, bk))
