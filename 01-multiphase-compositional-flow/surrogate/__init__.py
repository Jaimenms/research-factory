"""Modelo substituto (capítulo 4, §5.7): massa de dados de treino e teste gerada pelo simulador
(`surrogate.dataset`, por cima de `cmfsim.simulation.march_many`), amostragem por hipercubo latino
(`surrogate.sampling`), rede multicamada em JAX com Adam (`surrogate.model`, `surrogate.train`),
busca em grelha (`surrogate.grid`) e predição encadeada poço → linha → riser (`surrogate.predict`).
Fora do simulador: só usa a API pública de `cmfsim`."""

from .model import MLP, Hyperparameters
from .train import evaluate, r2_score, split_indices, train

__all__ = ["MLP", "Hyperparameters", "evaluate", "r2_score", "split_indices", "train"]
