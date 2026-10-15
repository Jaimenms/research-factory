"""cmfsim: escoamento bifásico composicional em tubulações, regime permanente (reprodução da
dissertação de G. A. Carvalho, UERJ 2022, com a arquitetura do mfsim e backend numpy/JAX)."""

from .io.spec import Case, case_from_json
from .results import Results, load_run
from .simulation import simulate

__all__ = ["Case", "case_from_json", "Results", "load_run", "simulate"]
