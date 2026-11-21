"""Testes do LSTM sem grafo e do laço de treino (task 14)."""

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import fazer_bruto

from dengue import dados, lstm_validacao, modelos, treino

RAPIDO = treino.ConfigTreino(max_epocas=3, paciencia=2, lote=64)


def particao_falsa(s=40, n=5, seed=0):
    """Partição sintética em que o alvo é a média da taxa nas 4 semanas (fácil de aprender)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(s, dados.W, n, len(dados.ENTRADAS))).astype(np.float32)
    y = np.repeat(x[..., 0].mean(axis=1, keepdims=True), dados.H, axis=1).astype(np.float32)
    vazio = np.zeros((s, dados.W))
    return dados.Particao(x=x, y=y, y_bruto=np.expm1(y), datas_entrada=vazio, datas_alvo=vazio)


def test_formato_da_saida():
    modelo = modelos.LSTM(oculto=8)
    saida = modelo(torch.zeros(3, dados.W, 7, len(dados.ENTRADAS)))
    assert saida.shape == (3, dados.H, 7)


def test_pesos_compartilhados_entre_municipios():
    # sem grafo e com os mesmos pesos: trocar a ordem dos municípios só troca a ordem da saída
    torch.manual_seed(0)
    modelo = modelos.LSTM(oculto=8)
    x = torch.randn(2, dados.W, 6, len(dados.ENTRADAS))
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    torch.testing.assert_close(modelo(x[:, :, perm]), modelo(x)[:, :, perm])


def test_mesma_seed_mesmo_resultado():
    p = particao_falsa()
    saidas = []
    for _ in range(2):
        treino.fixar_seed(0)
        modelo = modelos.LSTM(oculto=8)
        treino.treinar(modelo, p, p, RAPIDO)
        saidas.append(treino.prever(modelo, p))
    np.testing.assert_array_equal(saidas[0], saidas[1])


def test_treino_reduz_a_perda():
    p = particao_falsa()
    treino.fixar_seed(0)
    modelo = modelos.LSTM(oculto=16)
    antes = treino.perda(modelo, p)
    hist = treino.treinar(modelo, p, p, treino.ConfigTreino(max_epocas=60, lr=1e-2, lote=8))
    assert treino.perda(modelo, p) < 0.3 * antes
    assert hist.melhor_epoca >= 1


def test_early_stopping_restaura_a_melhor_epoca():
    p_treino, p_val = particao_falsa(seed=0), particao_falsa(seed=1)
    treino.fixar_seed(0)
    modelo = modelos.LSTM(oculto=8)
    hist = treino.treinar(modelo, p_treino, p_val, treino.ConfigTreino(max_epocas=8, paciencia=3))
    melhor = min(hist.perda_validacao)
    assert hist.perda_validacao[hist.melhor_epoca - 1] == melhor
    assert treino.perda(modelo, p_val) == pytest.approx(melhor, rel=1e-5)


def test_previsao_na_escala_bruta_nunca_negativa():
    p = particao_falsa()
    modelo = modelos.LSTM(oculto=8)
    y_hat = treino.prever(modelo, p)
    assert y_hat.shape == p.y.shape and (y_hat >= 0).all()


def test_mudar_o_teste_nao_muda_o_resultado_da_validacao():
    # contra vazamento: o resultado da validação só pode depender do treino e da validação
    df = dados.limpar(fazer_bruto(n_municipios=3, n_semanas=1199))
    alterado = df.copy()
    no_teste = alterado["data"] >= dados.PARTICOES["teste"][0]
    alterado.loc[no_teste, dados.ENTRADAS] = 999.0
    a = lstm_validacao.avaliar(df, seeds=(0,), config=RAPIDO, oculto=8)
    b = lstm_validacao.avaliar(alterado, seeds=(0,), config=RAPIDO, oculto=8)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_media_desvio():
    r = pd.DataFrame({"mae": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
    tabela = lstm_validacao.media_desvio(r)
    assert tabela.loc["media", "mae"] == 2.0 and tabela.loc["desvio", "mae"] == 1.0


def test_resultado_registrado_tem_3_seeds():
    caminho = lstm_validacao.CAMINHO_RESULTADO
    tabela = pd.read_csv(caminho, index_col=0)
    assert {"0", "1", "2", "media", "desvio"} <= set(tabela.index.astype(str))
