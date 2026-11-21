"""Testes do grafo geográfico."""

import numpy as np
import pytest

from dengue import grafo


@pytest.fixture
def coordenadas():
    rng = np.random.default_rng(0)
    lat = rng.uniform(-23.3, -20.8, size=30)  # caixa parecida com o RJ
    lon = rng.uniform(-44.9, -41.0, size=30)
    return lat, lon


def test_haversine_rio_sao_paulo():
    # Rio (-22.91, -43.17) a São Paulo (-23.55, -46.63): cerca de 360 km
    d = grafo.distancias_haversine([-22.91, -23.55], [-43.17, -46.63])
    assert d[0, 1] == pytest.approx(360, abs=10)
    assert d[0, 0] == 0.0


def test_haversine_simetrica(coordenadas):
    d = grafo.distancias_haversine(*coordenadas)
    np.testing.assert_allclose(d, d.T)


def test_grafo_knn_propriedades(coordenadas):
    k = 5
    adj = grafo.grafo_knn(grafo.distancias_haversine(*coordenadas), k=k)
    np.testing.assert_allclose(adj, adj.T)  # simétrica
    assert np.all(np.diag(adj) == 0)  # sem laço
    assert np.all((adj >= 0) & (adj <= 1))  # pesos do kernel entre 0 e 1
    assert np.all((adj > 0).sum(axis=1) >= k)  # cada nó tem ao menos k vizinhos


def test_grafo_knn_liga_o_mais_proximo():
    # Três pontos em linha: 0 e 1 estão a 1 km, 2 está longe
    dist = np.array([[0.0, 1.0, 50.0], [1.0, 0.0, 49.0], [50.0, 49.0, 0.0]])
    adj = grafo.grafo_knn(dist, k=1)
    assert adj[0, 1] > 0
    assert adj[0, 2] == 0  # 0 não escolheu 2, e 2 escolheu 1


def test_grafo_knn_k_invalido():
    with pytest.raises(ValueError, match="k precisa"):
        grafo.grafo_knn(np.zeros((3, 3)), k=3)


def test_laplaciana_autovalores(coordenadas):
    adj = grafo.grafo_knn(grafo.distancias_haversine(*coordenadas), k=5)
    autovalores = np.linalg.eigvalsh(grafo.laplaciana_normalizada(adj))
    assert autovalores.min() == pytest.approx(0, abs=1e-8)
    assert autovalores.max() <= 2 + 1e-8

    escalados = np.linalg.eigvalsh(grafo.laplaciana_escalada(adj))
    assert escalados.min() >= -1 - 1e-8
    assert escalados.max() == pytest.approx(1.0)


def test_vizinhos_parecidos_sao_detectados():
    # Dois grupos de 5 nós: dentro de cada grupo as séries são quase iguais
    rng = np.random.default_rng(1)
    base = rng.normal(size=(200, 2))
    series = np.repeat(base, 5, axis=1) + 0.1 * rng.normal(size=(200, 10))
    adj = np.zeros((10, 10))
    adj[:5, :5] = 1
    adj[5:, 5:] = 1
    np.fill_diagonal(adj, 0)

    r = grafo.teste_vizinhos_aleatorios(series, adj, n_sorteios=200)
    assert r["corr_vizinhos"] > 0.9
    assert r["corr_vizinhos"] > r["corr_aleatorios"]
    assert r["p_valor"] < 0.05
