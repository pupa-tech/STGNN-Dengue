"""Grafo geográfico entre os municípios.

Ideia: municípios próximos tendem a ter epidemias parecidas (o mosquito e as pessoas
circulam entre eles). O grafo diz ao STGCN quem é vizinho de quem.

Passos:
1. distância em km entre cada par de municípios (haversine);
2. cada município se liga aos `k` mais próximos (k-NN), e a matriz é simetrizada;
3. o peso da aresta cai com a distância (kernel gaussiano);
4. a Laplaciana normalizada é o que a convolução de Chebyshev usa.
"""

import numpy as np

RAIO_TERRA_KM = 6371.0


def distancias_haversine(lat, lon) -> np.ndarray:
    """Matriz `[N, N]` de distâncias em km, levando em conta a curvatura da Terra."""
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * RAIO_TERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def grafo_knn(dist, k: int = 5) -> np.ndarray:
    """Adjacência ponderada `[N, N]` do k-NN com kernel gaussiano.

    - Cada nó liga aos `k` vizinhos mais próximos; se i escolhe j, j também liga a i
      (simetrização por "ou"), então alguns nós ficam com mais de `k` vizinhos.
    - Peso: `exp(-d² / σ²)`, com σ = mediana das distâncias das arestas. Assim o peso
      fica entre 0 e 1 e não depende da unidade escolhida.
    - Diagonal zero (sem laço do nó com ele mesmo).
    """
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[0]
    if not 0 < k < n:
        raise ValueError(f"k precisa estar entre 1 e {n - 1}, recebido {k}")

    d = dist.copy()
    np.fill_diagonal(d, np.inf)  # o próprio nó não conta como vizinho
    vizinhos = np.argsort(d, axis=1)[:, :k]

    aresta = np.zeros((n, n), dtype=bool)
    aresta[np.arange(n)[:, None], vizinhos] = True
    aresta = aresta | aresta.T

    sigma = np.median(dist[aresta])
    pesos = np.exp(-(dist**2) / sigma**2)
    return np.where(aresta, pesos, 0.0)


def laplaciana_normalizada(adj) -> np.ndarray:
    """`L = I - D^-1/2 A D^-1/2`. Seus autovalores ficam entre 0 e 2."""
    adj = np.asarray(adj, dtype=float)
    grau = adj.sum(axis=1)
    inv_raiz = np.zeros_like(grau)
    inv_raiz[grau > 0] = 1.0 / np.sqrt(grau[grau > 0])
    return np.eye(len(adj)) - inv_raiz[:, None] * adj * inv_raiz[None, :]


def laplaciana_escalada(adj) -> np.ndarray:
    """`L~ = 2L / λmax - I`, que leva os autovalores para [-1, 1].

    É a entrada dos polinômios de Chebyshev do STGCN, que só são estáveis nesse intervalo.
    """
    lap = laplaciana_normalizada(adj)
    lambda_max = np.linalg.eigvalsh(lap).max()
    return 2.0 * lap / lambda_max - np.eye(len(lap))


def _correlacao_media(series: np.ndarray, pares: np.ndarray) -> float:
    """Média da correlação de Pearson entre as séries de cada par `(i, j)`.

    Pares com alguma série constante (por exemplo, só zeros) são ignorados.
    """
    corr = np.corrcoef(series.T)  # [N, N]
    valores = corr[pares[:, 0], pares[:, 1]]
    return float(np.nanmean(valores))


def teste_vizinhos_aleatorios(series, adj, n_sorteios: int = 1000, seed: int = 0) -> dict:
    """Os vizinhos do grafo se parecem mais entre si do que pares quaisquer?

    `series` tem formato `[T, N]`. Use **só o período de treino**, para não espiar o futuro.

    Compara a correlação média entre pares ligados no grafo com a de `n_sorteios`
    conjuntos de pares aleatórios do mesmo tamanho. O `p_valor` é a fração de sorteios
    com correlação maior ou igual à dos vizinhos (quanto menor, mais o grafo faz sentido).
    """
    series = np.asarray(series, dtype=float)
    adj = np.asarray(adj)
    n = adj.shape[0]

    i, j = np.nonzero(np.triu(adj, k=1))
    pares_viz = np.stack([i, j], axis=1)
    corr_viz = _correlacao_media(series, pares_viz)

    rng = np.random.default_rng(seed)
    todos_i, todos_j = np.triu_indices(n, k=1)
    todos = np.stack([todos_i, todos_j], axis=1)
    corr_aleat = np.array(
        [
            _correlacao_media(series, todos[rng.choice(len(todos), len(pares_viz), replace=False)])
            for _ in range(n_sorteios)
        ]
    )
    return {
        "corr_vizinhos": corr_viz,
        "corr_aleatorios": float(corr_aleat.mean()),
        "corr_aleatorios_todos": corr_aleat,
        "p_valor": float(np.mean(corr_aleat >= corr_viz)),
    }


TIPOS_GRAFO = ("geo", "identidade")


def laplaciana_do_projeto(df, tipo: str = "geo", k: int = 5) -> np.ndarray:
    """`L~` pronta para o STGCN, a partir da tabela limpa (`dados.limpar`).

    - `"geo"`: k-NN geográfico com kernel gaussiano (task 15);
    - `"identidade"`: grafo sem arestas. Então `L = I` e `L~ = I`, e a convolução de
      Chebyshev só olha o próprio município (é o controle "sem grafo" da task 17).

    Os nós seguem a ordem de `dados.municipios` (códigos IBGE crescentes).
    """
    from dengue import dados  # import local: grafo.py não depende de pandas no topo

    if tipo not in TIPOS_GRAFO:
        raise ValueError(f"tipo deve ser um de {TIPOS_GRAFO}, recebido {tipo!r}")
    coord = df.groupby("municipio")[["latitude", "longitude"]].first()
    coord = coord.loc[dados.municipios(df)]
    n = len(coord)
    if tipo == "identidade":
        return np.eye(n)
    adj = grafo_knn(distancias_haversine(coord["latitude"], coord["longitude"]), k=k)
    return laplaciana_escalada(adj)
