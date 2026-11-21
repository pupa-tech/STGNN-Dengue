"""Funções de perda comparadas na task 20.

Em todas, o modelo devolve `z` na escala `log1p` e a previsão na taxa é `expm1(z)`, como
antes. Só muda **como o erro é medido durante o treino**:

- `mse`: erro quadrático em `log1p` (a perda usada até aqui, decisão 002);
- `huber`: quadrática para erros menores que `delta` e linear acima disso. Em `log1p`,
  isso diminui o peso dos erros grandes (em geral, os surtos);
- `tweedie`: perda da família Tweedie (1 < p < 2), feita para dados com muitos zeros e
  cauda longa. Ela trabalha na escala da taxa. Para manter a mesma saída, é aplicada a
  `u = taxa + 1 = exp(y_log)` com média `mu = exp(z)`; o mínimo fica em `mu = E[u]`, ou
  seja, `expm1(z)` estima a taxa média.

A comparação entre as perdas é sempre feita no MAE e no RMSE na taxa bruta, nunca na
própria perda.
"""

import torch
from torch import nn

PERDAS = ("mse", "huber", "tweedie")


def tweedie(z: torch.Tensor, y_log: torch.Tensor, p: float = 1.5) -> torch.Tensor:
    """Perda Tweedie média (sem os termos que não dependem de `z`).

    L = -u · mu^(1-p) / (1-p) + mu^(2-p) / (2-p),  com u = exp(y_log) e mu = exp(z).
    """
    if not 1 < p < 2:
        raise ValueError("tweedie_p precisa estar entre 1 e 2")
    u = torch.exp(y_log)
    termo_u = -u * torch.exp((1 - p) * z) / (1 - p)
    termo_mu = torch.exp((2 - p) * z) / (2 - p)
    return (termo_u + termo_mu).mean()


def calcular(
    nome: str,
    z: torch.Tensor,
    y_log: torch.Tensor,
    huber_delta: float = 1.0,
    tweedie_p: float = 1.5,
    reducao: str = "mean",
) -> torch.Tensor:
    """Perda `nome` entre a saída `z` e o alvo `y_log` (ambos em `log1p`).

    `reducao="sum"` devolve a soma, usada para tirar a média numa partição inteira.
    """
    if nome == "mse":
        return nn.functional.mse_loss(z, y_log, reduction=reducao)
    if nome == "huber":
        return nn.functional.huber_loss(z, y_log, delta=huber_delta, reduction=reducao)
    if nome == "tweedie":
        media = tweedie(z, y_log, tweedie_p)
        return media * y_log.numel() if reducao == "sum" else media
    raise ValueError(f"perda deve ser uma de {PERDAS}, recebido {nome!r}")
