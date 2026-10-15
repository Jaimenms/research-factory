"""Um único tubo com parâmetros que variam axialmente (decisão A9): perfil por pontos de quebra em x,
cada um com diâmetro, rugosidade, inclinação, coeficiente global de troca térmica e temperatura
ambiente constantes até ao ponto seguinte; o último ponto só diz onde o tubo acaba. Os pontos de
quebra são eventos para a marcha (reinício do passo) e nós da malha das diferenças finitas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PipeProfile:
    x_break: np.ndarray  # (ns + 1,) pontos de quebra, x_break[0] = 0, x_break[-1] = L
    D: np.ndarray  # (ns,) por troço
    roughness: np.ndarray
    theta: np.ndarray  # rad, positivo ascendente
    U: np.ndarray  # W/(m2 K)
    T_amb: np.ndarray  # K

    @property
    def n_segments(self) -> int:
        return self.D.size

    @property
    def L(self) -> float:
        return float(self.x_break[-1])

    @classmethod
    def from_json(cls, d: dict) -> "PipeProfile":
        prof = d["profile"]
        if len(prof) < 2:
            raise ValueError("pipe.profile precisa de pelo menos dois pontos (início e fim)")
        xs = np.array([float(p["x"]) for p in prof])
        if xs[0] != 0.0 or np.any(np.diff(xs) <= 0):
            raise ValueError("pipe.profile: x deve começar em 0 e ser estritamente crescente")
        segs = prof[:-1]

        def col(key, default=None, scale=1.0):
            out = []
            last = default
            for p in segs:
                v = p.get(key, last)
                if v is None:
                    raise ValueError(f"pipe.profile: falta '{key}' no ponto x = {p['x']}")
                last = v
                out.append(float(v) * scale)
            return np.array(out)

        return cls(
            x_break=xs,
            D=col("D"),
            roughness=col("roughness", 0.0),
            theta=col("angle_deg", 0.0, np.pi / 180.0),
            U=col("U", 0.0),
            T_amb=col("T_amb", 288.15),
        )

    def to_json(self) -> dict:
        prof = []
        for i in range(self.n_segments):
            prof.append(
                dict(
                    x=float(self.x_break[i]),
                    D=float(self.D[i]),
                    roughness=float(self.roughness[i]),
                    angle_deg=float(self.theta[i] * 180.0 / np.pi),
                    U=float(self.U[i]),
                    T_amb=float(self.T_amb[i]),
                )
            )
        prof.append(dict(x=float(self.x_break[-1])))
        return {"profile": prof}

    def arrays(self):
        return dict(D=self.D, roughness=self.roughness, theta=self.theta, U=self.U, T_amb=self.T_amb)

    def segment_of(self, x):
        """Índice do troço que contém x (o último troço inclui x = L)."""
        return np.clip(np.searchsorted(self.x_break, np.asarray(x, float), side="right") - 1, 0, self.n_segments - 1)

    def elevation(self, x):
        """Cota z(x) a partir da entrada (integral de sin theta)."""
        x = np.asarray(x, float)
        dz = np.diff(self.x_break) * np.sin(self.theta)
        z_break = np.concatenate([[0.0], np.cumsum(dz)])
        i = self.segment_of(x)
        return z_break[i] + (x - self.x_break[i]) * np.sin(self.theta[i])

    def output_grid(self, dx_out: float):
        """Pontos de saída: múltiplos de dx_out, os pontos de quebra e L, ordenados e únicos."""
        g = np.arange(0.0, self.L + 0.5 * dx_out, dx_out)
        g = np.unique(np.concatenate([g, self.x_break]))
        return g[(g >= 0.0) & (g <= self.L + 1e-12)]

    def interval_segments(self, x_out):
        """Troço de cada intervalo [x_out[i], x_out[i+1]] (pelo ponto médio)."""
        mids = 0.5 * (np.asarray(x_out[:-1]) + np.asarray(x_out[1:]))
        return self.segment_of(mids)

    def uniform_mesh(self, n: int):
        """Malha com ~n nós distribuídos pelos troços proporcionalmente ao comprimento (mínimo 2 por
        troço), incluindo os pontos de quebra."""
        L = np.diff(self.x_break)
        counts = np.maximum(2, np.round(n * L / L.sum()).astype(int))
        nodes = [np.array([0.0])]
        for i in range(self.n_segments):
            nodes.append(np.linspace(self.x_break[i], self.x_break[i + 1], counts[i] + 1)[1:])
        return np.concatenate(nodes)
