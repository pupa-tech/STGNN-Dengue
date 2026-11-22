"""Modelos treináveis.

Formato comum a todos os modelos:
- entrada `x`: [B, W, N, F] (lote de janelas, semanas, municípios, variáveis), já normalizada;
- saída: [B, H, N], a previsão do alvo em `log1p` para as H semanas seguintes.
"""

import numpy as np
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


# --- STGCN (task 16) ---


class ConvTemporal(nn.Module):
    """Convolução no tempo com porta (GLU), causal: a saída em t só vê t, t-1, ..., t-kt+1.

    Metade dos canais é o sinal e a outra metade é uma porta entre 0 e 1 (sigmoide) que
    decide quanto do sinal passa. Entrada e saída: [B, C, N, T], com o mesmo T.
    """

    def __init__(self, c_entrada: int, c_saida: int, kt: int = 3):
        super().__init__()
        self.kt = kt
        self.conv = nn.Conv2d(c_entrada, 2 * c_saida, kernel_size=(1, kt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.kt - 1, 0))  # preenche só o passado (causal)
        sinal, porta = self.conv(x).chunk(2, dim=1)
        return sinal * torch.sigmoid(porta)


class ConvChebyshev(nn.Module):
    """Convolução no grafo com polinômios de Chebyshev de ordem K-1.

    `T_0 = I`, `T_1 = L~`, `T_k = 2 L~ T_{k-1} - T_{k-2}`. O termo `T_k` mistura cada nó
    com os vizinhos até k saltos, então K=3 olha até 2 saltos. `L~` é a Laplaciana
    escalada para [-1, 1] (veja `grafo.laplaciana_escalada`). Entrada/saída: [B, C, N, T].
    """

    def __init__(self, c_entrada: int, c_saida: int, lap_escalada, K: int = 3):
        super().__init__()
        lap = torch.as_tensor(np.asarray(lap_escalada), dtype=torch.float32)
        polinomios = [torch.eye(lap.shape[0])]
        if K > 1:
            polinomios.append(lap)
        for _ in range(2, K):
            polinomios.append(2 * lap @ polinomios[-1] - polinomios[-2])
        self.register_buffer("T", torch.stack(polinomios))  # [K, N, N], fixo (não treina)
        self.theta = nn.Parameter(torch.empty(K, c_entrada, c_saida))
        nn.init.xavier_uniform_(self.theta)
        self.vies = nn.Parameter(torch.zeros(c_saida))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vizinhanca = torch.einsum("knm,bcmt->kbcnt", self.T, x)  # x espalhado pelo grafo
        saida = torch.einsum("kbcnt,kco->bont", vizinhanca, self.theta)
        return saida + self.vies.view(1, -1, 1, 1)


class STGCN(nn.Module):
    """STGCN de 1 bloco: tempo (GLU) -> grafo (Chebyshev) -> tempo (GLU) -> saída.

    Mesmo formato do LSTM: entrada [B, W, N, F] normalizada, saída [B, H, N] em `log1p`.
    Os pesos são os mesmos para todos os municípios; o que diferencia os nós é o grafo.

    - Conexão residual: o bloco aprende uma correção sobre a entrada projetada.
    - LayerNorm só nos canais de cada nó (não mistura municípios, então com `L~ = I`
      cada município é de fato independente dos outros).
    - Cabeça: as W semanas de cada nó, achatadas, viram as H previsões.
    """

    def __init__(
        self,
        lap_escalada,
        n_entradas: int = len(dados.ENTRADAS),
        oculto: int = 32,
        K: int = 3,
        kt: int = 3,
        w: int = dados.W,
        h: int = dados.H,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.temp1 = ConvTemporal(n_entradas, oculto, kt)
        self.grafo = ConvChebyshev(oculto, oculto, lap_escalada, K)
        self.temp2 = ConvTemporal(oculto, oculto, kt)
        self.residual = nn.Conv2d(n_entradas, oculto, kernel_size=1)
        self.norma = nn.LayerNorm(oculto)
        self.dropout = nn.Dropout(dropout)
        self.cabeca = nn.Linear(oculto * w, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 2, 1)  # [B, W, N, F] -> [B, F, N, W]
        z = self.temp1(x)
        z = torch.relu(self.grafo(z))
        z = self.temp2(z) + self.residual(x)  # [B, C, N, W]
        z = self.norma(z.permute(0, 2, 3, 1))  # [B, N, W, C], normaliza só os canais
        z = self.dropout(z).flatten(2)  # [B, N, W·C]
        return self.cabeca(z).permute(0, 2, 1)  # [B, H, N]


# --- Transformer espacial (task 23) ---


class CamadaAtencao(nn.Module):
    """Uma camada do Transformer (pre-norm): atenção entre municípios e MLP, ambas residuais.

    Entrada e saída: [B, N, d], um token por município. Devolve também os pesos de
    atenção [B, N, N], com a média das cabeças: a linha i diz quanto o município i
    olhou para cada um dos outros.
    """

    def __init__(self, d: int, cabecas: int, dropout: float):
        super().__init__()
        self.norma1 = nn.LayerNorm(d)
        self.atencao = nn.MultiheadAttention(d, cabecas, dropout=dropout, batch_first=True)
        self.norma2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, mascara: torch.Tensor | None = None):
        a = self.norma1(z)
        a, pesos = self.atencao(a, a, a, attn_mask=mascara, need_weights=True)
        z = z + self.dropout(a)
        z = z + self.dropout(self.mlp(self.norma2(z)))
        return z, pesos


class TransformerEspacial(nn.Module):
    """Transformer com atenção entre os municípios, no lugar de um grafo fixo (task 23).

    Mesmo formato dos outros modelos: entrada [B, W, N, F] normalizada, saída [B, H, N]
    em `log1p`.

    - **Token:** cada município vira um vetor de dimensão `d`. As W semanas × F variáveis
      dele passam por uma camada linear, e somamos um embedding do município, aprendido,
      que funciona como a "identidade" dele (no STGCN, quem diferencia os nós é o grafo).
    - **Atenção:** em cada camada, cada município decide, pelos dados da janela, quanto
      olhar para cada um dos outros 79. É um grafo aprendido e que muda a cada semana.
    - **Cabeça:** o token final de cada município vira as H previsões.

    Com `atencao=False`, uma máscara deixa cada município olhar só para si mesmo. É a
    ablação equivalente ao STGCN com A = I: a arquitetura é a mesma, mas não há troca de
    informação entre municípios.
    """

    def __init__(
        self,
        n_municipios: int,
        n_entradas: int = len(dados.ENTRADAS),
        d: int = 32,
        cabecas: int = 4,
        camadas: int = 2,
        w: int = dados.W,
        h: int = dados.H,
        dropout: float = 0.2,
        atencao: bool = True,
    ):
        super().__init__()
        self.entrada = nn.Linear(w * n_entradas, d)
        self.municipio = nn.Parameter(torch.zeros(n_municipios, d))
        nn.init.normal_(self.municipio, std=0.02)
        self.camadas = nn.ModuleList(CamadaAtencao(d, cabecas, dropout) for _ in range(camadas))
        self.norma = nn.LayerNorm(d)
        self.cabeca = nn.Linear(d, h)
        # True = proibido. Sem atenção, só a diagonal fica liberada.
        mascara = None if atencao else ~torch.eye(n_municipios, dtype=torch.bool)
        self.register_buffer("mascara", mascara)

    def _codificar(self, x: torch.Tensor):
        b, w, n, f = x.shape
        tokens = x.permute(0, 2, 1, 3).reshape(b, n, w * f)  # [B, N, W·F]
        z = self.entrada(tokens) + self.municipio  # [B, N, d]
        pesos = []
        for camada in self.camadas:
            z, p = camada(z, self.mascara)
            pesos.append(p)
        return self.norma(z), pesos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self._codificar(x)
        return self.cabeca(z).permute(0, 2, 1)  # [B, H, N]

    @torch.no_grad()
    def mapas_atencao(self, x: torch.Tensor) -> torch.Tensor:
        """Pesos de atenção [camadas, B, N, N] (média das cabeças), em modo de avaliação."""
        estava_treinando = self.training
        self.eval()
        _, pesos = self._codificar(x)
        self.train(estava_treinando)
        return torch.stack(pesos)
