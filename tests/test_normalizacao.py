"""Testes da normalização: as estatísticas vêm só do treino (task 11)."""

import numpy as np
import pytest
from conftest import fazer_bruto

from dengue import dados

sem_csv = pytest.mark.skipif(not dados.CAMINHO_CSV.exists(), reason="CSV real ausente")


@pytest.fixture
def tabela_longa():
    return dados.limpar(fazer_bruto(n_municipios=3, n_semanas=1199))


def test_log1p_ida_e_volta():
    y = np.array([0.0, 0.5, 3.0, 215.0])
    np.testing.assert_allclose(dados.inverter_alvo(dados.transformar_alvo(y)), y, rtol=1e-12)
    assert dados.transformar_alvo(np.array([0.0]))[0] == 0.0
    # previsão negativa em log1p não vira taxa negativa
    assert dados.inverter_alvo(np.array([-0.5]))[0] == 0.0


def test_estatisticas_sao_as_do_treino():
    rng = np.random.default_rng(1)
    X_treino = rng.uniform(0, 10, size=(50, 4, 3))
    norm = dados.Normalizador.ajustar(X_treino, canal_alvo=0)
    esperado = X_treino.copy()
    esperado[..., 0] = np.log1p(esperado[..., 0])
    np.testing.assert_allclose(norm.media, esperado.mean(axis=(0, 1)), rtol=1e-5)
    np.testing.assert_allclose(norm.desvio, esperado.std(axis=(0, 1)), rtol=1e-5)
    # no próprio treino, cada variável fica com média 0 e desvio 1
    Z = norm.transformar(X_treino)
    np.testing.assert_allclose(Z.mean(axis=(0, 1)), 0, atol=1e-5)
    np.testing.assert_allclose(Z.std(axis=(0, 1)), 1, atol=1e-4)


def test_transformar_nao_altera_a_entrada():
    X = np.ones((5, 2, 2))
    dados.Normalizador.ajustar(X).transformar(X)
    assert (X == 1).all()


def test_variavel_constante_nao_divide_por_zero():
    norm = dados.Normalizador.ajustar(np.ones((5, 2, 2)))
    assert np.isfinite(norm.transformar(np.ones((5, 2, 2)))).all()


def test_mudar_validacao_e_teste_nao_muda_nada_do_treino(tabela_longa):
    """Teste central contra vazamento: o futuro não pode afetar o que o modelo vê no treino."""
    part_a, norm_a = dados.preparar(tabela_longa)

    alterada = tabela_longa.copy()
    futuro = alterada["data"] >= "2013-01-01"
    for coluna in dados.ENTRADAS:
        alterada.loc[futuro, coluna] = alterada.loc[futuro, coluna] * 1000 + 7
    part_b, norm_b = dados.preparar(alterada)

    np.testing.assert_array_equal(norm_a.media, norm_b.media)
    np.testing.assert_array_equal(norm_a.desvio, norm_b.desvio)
    np.testing.assert_array_equal(part_a["treino"].x, part_b["treino"].x)
    np.testing.assert_array_equal(part_a["treino"].y, part_b["treino"].y)
    # e a validação de fato mudou, ou seja, o teste não é trivial
    assert not np.array_equal(part_a["validacao"].x, part_b["validacao"].x)


def test_preparar_formatos_e_alvo_coerente(tabela_longa):
    particoes, _ = dados.preparar(tabela_longa)
    treino = particoes["treino"]
    S = len(treino.x)
    assert treino.x.shape == (S, dados.W, 3, len(dados.ENTRADAS))
    assert treino.y.shape == treino.y_bruto.shape == (S, dados.H, 3)
    np.testing.assert_allclose(dados.inverter_alvo(treino.y), treino.y_bruto, rtol=1e-5)
    assert treino.datas_alvo.max() < particoes["validacao"].datas_entrada.min()
    assert particoes["validacao"].datas_alvo.max() < particoes["teste"].datas_entrada.min()


@sem_csv
def test_preparar_dados_reais():
    particoes, norm = dados.preparar(dados.carregar_tabela())
    assert [len(p.x) for p in particoes.values()] == [724, 149, 305]
    x = particoes["treino"].x
    assert np.isfinite(x).all()
    # o treino normalizado fica centrado (as janelas repetem semanas, então é aproximado)
    np.testing.assert_allclose(x.mean(axis=(0, 1, 2)), 0, atol=0.05)
