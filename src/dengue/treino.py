"""Laço de treino comum aos modelos (LSTM agora, STGCN depois).

- Perda: MSE sobre o alvo em `log1p` (decisão 002).
- Otimizador: Adam. Lote = janelas inteiras (cada janela traz os 79 municípios).
- Early stopping pela perda na validação: o treino para quando a validação não melhora
  por `paciencia` épocas, e os pesos da melhor época são restaurados.

O teste nunca passa por aqui: `treinar` só recebe treino e validação.
"""

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from dengue import dados


def fixar_seed(seed: int) -> None:
    """Fixa as seeds de Python, NumPy e PyTorch para o resultado ser reproduzível."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class ConfigTreino:
    lr: float = 1e-3
    lote: int = 32
    max_epocas: int = 300
    paciencia: int = 15
    seed: int = 0


@dataclass
class Historico:
    perda_treino: list[float] = field(default_factory=list)
    perda_validacao: list[float] = field(default_factory=list)
    melhor_epoca: int = 0  # contada a partir de 1


def _tensores(p: dados.Particao) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(np.asarray(p.x, np.float32)), torch.from_numpy(
        np.asarray(p.y, np.float32)
    )


@torch.no_grad()
def perda(modelo: nn.Module, p: dados.Particao, lote: int = 256) -> float:
    """MSE médio em `log1p` numa partição inteira."""
    modelo.eval()
    x, y = _tensores(p)
    soma = 0.0
    for i in range(0, len(x), lote):
        soma += nn.functional.mse_loss(modelo(x[i : i + lote]), y[i : i + lote], reduction="sum")
    return float(soma) / y.numel()


def treinar(
    modelo: nn.Module,
    treino: dados.Particao,
    validacao: dados.Particao,
    config: ConfigTreino | None = None,
) -> Historico:
    """Treina `modelo` no lugar e devolve o histórico das perdas por época."""
    config = config or ConfigTreino()
    fixar_seed(config.seed)
    x, y = _tensores(treino)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=config.lr)
    gerador = torch.Generator().manual_seed(config.seed)  # ordem dos lotes reproduzível

    historico = Historico()
    melhor, melhor_estado, sem_melhora = float("inf"), None, 0
    for epoca in range(1, config.max_epocas + 1):
        modelo.train()
        ordem = torch.randperm(len(x), generator=gerador)
        soma = 0.0
        for i in range(0, len(x), config.lote):
            idx = ordem[i : i + config.lote]
            otimizador.zero_grad()
            loss = nn.functional.mse_loss(modelo(x[idx]), y[idx])
            loss.backward()
            otimizador.step()
            soma += loss.item() * len(idx)
        historico.perda_treino.append(soma / len(x))
        historico.perda_validacao.append(perda(modelo, validacao))

        if historico.perda_validacao[-1] < melhor:
            melhor, sem_melhora = historico.perda_validacao[-1], 0
            melhor_estado = copy.deepcopy(modelo.state_dict())
            historico.melhor_epoca = epoca
        else:
            sem_melhora += 1
            if sem_melhora >= config.paciencia:
                break

    modelo.load_state_dict(melhor_estado)
    return historico


@torch.no_grad()
def prever(modelo: nn.Module, p: dados.Particao, lote: int = 256) -> np.ndarray:
    """Previsão na escala **bruta** da taxa, [S, H, N], pronta para as métricas."""
    modelo.eval()
    x, _ = _tensores(p)
    saida = torch.cat([modelo(x[i : i + lote]) for i in range(0, len(x), lote)])
    return dados.inverter_alvo(saida.numpy())
