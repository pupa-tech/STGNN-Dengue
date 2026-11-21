"""Testes do split temporal: o portão contra vazamento antes de qualquer modelo (task 10)."""

import numpy as np
import pandas as pd
import pytest
from conftest import fazer_bruto

from dengue import dados

sem_csv = pytest.mark.skipif(not dados.CAMINHO_CSV.exists(), reason="CSV real ausente")


@pytest.fixture
def tabela_longa():
    # 1999 a 2021 inteiro, com 3 municípios: cobre as três partições
    return dados.limpar(fazer_bruto(n_municipios=3, n_semanas=1199))


def checar_sem_vazamento(datas, X, y, indices):
    """Nenhuma semana usada (entrada ou alvo) por uma partição aparece em outra posterior."""
    usadas = {}
    for nome, idx in indices.items():
        _, _, idx_entrada, idx_alvo = dados.janelas(X, y, idx)
        usadas[nome] = datas[np.union1d(idx_entrada.ravel(), idx_alvo.ravel())]
    assert usadas["treino"].max() < usadas["validacao"].min()
    assert usadas["validacao"].max() < usadas["teste"].min()
    assert not set(usadas["teste"]) & set(usadas["treino"])
    assert not set(usadas["teste"]) & set(usadas["validacao"])
    return usadas


def test_montar_tensor_formatos(tabela_longa):
    X, y, datas = dados.montar_tensor(tabela_longa)
    assert X.shape == (1199, 3, len(dados.ENTRADAS))
    assert y.shape == (1199, 3)
    assert len(datas) == 1199 and datas.is_monotonic_increasing
    # o canal 0 de X é o próprio alvo
    np.testing.assert_array_equal(X[..., 0], y)


def test_particoes_respeitam_os_anos(tabela_longa):
    _, _, datas = dados.montar_tensor(tabela_longa)
    idx = dados.indices_particoes(datas)
    assert datas[idx["treino"]].year.min() == 1999 and datas[idx["treino"]].year.max() == 2012
    assert set(datas[idx["validacao"]].year) == {2013, 2014, 2015}
    assert datas[idx["teste"]].year.min() == 2016
    # cada semana está em exatamente uma partição
    todos = np.concatenate(list(idx.values()))
    assert len(todos) == len(np.unique(todos)) == len(datas)


def test_janelas_alinham_entrada_e_alvo():
    T, N = 20, 2
    y = np.arange(T * N, dtype=np.float32).reshape(T, N)
    X = y[..., None]
    x, alvo, idx_entrada, idx_alvo = dados.janelas(X, y, np.arange(T), w=4, h=4)
    assert x.shape == (T - 7, 4, N, 1) and alvo.shape == (T - 7, 4, N)
    # a primeira janela: entrada nas semanas 0..3, alvo nas semanas 4..7
    np.testing.assert_array_equal(idx_entrada[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(idx_alvo[0], [4, 5, 6, 7])
    np.testing.assert_array_equal(alvo[0], y[4:8])
    # o alvo começa logo depois da última semana de entrada, em todas as janelas
    assert (idx_alvo[:, 0] == idx_entrada[:, -1] + 1).all()


def test_janelas_rejeitam_particao_curta_ou_com_buraco():
    y = np.zeros((10, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="curta"):
        dados.janelas(y[..., None], y, np.arange(5), w=4, h=4)
    with pytest.raises(ValueError, match="consecutivas"):
        dados.janelas(y[..., None], y, np.array([0, 1, 2, 3, 5, 6, 7, 8, 9]))


def test_nenhuma_data_do_teste_aparece_no_treino_sintetico(tabela_longa):
    X, y, datas = dados.montar_tensor(tabela_longa)
    checar_sem_vazamento(datas, X, y, dados.indices_particoes(datas))


@sem_csv
def test_nenhuma_data_do_teste_aparece_no_treino_real():
    X, y, datas = dados.montar_tensor(dados.carregar_tabela())
    idx = dados.indices_particoes(datas)
    usadas = checar_sem_vazamento(datas, X, y, idx)
    assert usadas["treino"].max() <= pd.Timestamp("2012-12-31")
    assert usadas["teste"].min() >= pd.Timestamp("2016-01-01")
    # número de semanas por partição e de janelas (semanas - W - H + 1)
    assert [len(i) for i in idx.values()] == [731, 156, 312]
    n_janelas = [len(dados.janelas(X, y, i)[0]) for i in idx.values()]
    assert n_janelas == [724, 149, 305]
