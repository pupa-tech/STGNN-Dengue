"""Modelos treináveis.

Formato comum a todos os modelos:
- entrada `x`: [B, W, N, F] (lote de janelas, semanas, municípios, variáveis), já normalizada;
- saída: [B, H, N], a previsão do alvo em `log1p` para as H semanas seguintes.
"""

import torch
from torch import nn

from dengue import dados


class LSTM(nn.Module):
    """LSTM sem grafo, com os mesmos pesos para todos os municípios (task 14).

    Cada município vira uma sequência independente de W semanas: o modelo só olha o
    passado do próprio município, nunca o dos vizinhos. Por isso é o baseline "sem grafo"
    do STGCN. Os pesos são compartilhados: um único LSTM aprende com os 79 municípios.

    O último estado oculto resume as W semanas, e uma camada linear prevê as H semanas
    de uma vez (previsão direta, sem realimentar a própria previsão).
    """

    def __init__(self, n_entradas: int = len(dados.ENTRADAS), oculto: int = 64, h: int = dados.H):
        super().__init__()
        self.lstm = nn.LSTM(n_entradas, oculto, batch_first=True)
        self.cabeca = nn.Linear(oculto, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, w, n, f = x.shape
        # [B, W, N, F] -> [B, N, W, F] -> [B·N, W, F]: uma sequência por município
        seq = x.permute(0, 2, 1, 3).reshape(b * n, w, f)
        _, (estado, _) = self.lstm(seq)  # estado: [1, B·N, oculto]
        saida = self.cabeca(estado[-1])  # [B·N, H]
        return saida.reshape(b, n, -1).permute(0, 2, 1)  # [B, H, N]
