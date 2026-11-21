"""Testes dos baselines ingênuos (task 13)."""

import numpy as np
import pandas as pd
import pytest
from conftest import fazer_bruto

from dengue import baselines, dados

sem_csv = pytest.mark.skipif(not dados.CAMINHO_CSV.exists(), reason="CSV real ausente")


@pytest.fixture
def tabela_longa():
    return dados.limpar(fazer_bruto(n_municipios=3, n_semanas=1199))


def test_persistencia_repete_a_ultima_semana():
    y_entrada = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    prev = baselines.persistencia(y_entrada, h=4)
    assert prev.shape == (2, 4, 3)
    for h in range(4):
        np.testing.assert_array_equal(prev[:, h], y_entrada[:, -1])


def test_persistencia_formato_invalido():
    with pytest.raises(ValueError, match="S, W, N"):
        baselines.persistencia(np.zeros((4, 3)))


def test_semana_do_ano_fica_entre_1_e_52():
    datas = pd.to_datetime(["2001-01-01", "2001-01-07", "2001-01-08", "2001-12-31", "2004-12-31"])
    np.testing.assert_array_equal(baselines.semana_do_ano(datas), [1, 1, 2, 52, 52])
    # mantém o formato [S, H]
    assert baselines.semana_do_ano(np.array(datas[:4]).reshape(2, 2)).shape == (2, 2)


def test_climatologia_a_mao():
    # dois anos, um município: semana 1 vale 2 e depois 4; o resto vale 0
    datas = pd.date_range("2001-01-01", "2002-12-31", freq="7D")
    y = np.zeros((len(datas), 1))
    semanas = baselines.semana_do_ano(datas)
    y[semanas == 1] = np.array([[2.0], [4.0]])
    tabela = baselines.ajustar_climatologia(y, datas, "media")
    assert tabela.shape == (52, 1)
    assert tabela[0, 0] == pytest.approx(3.0)
    assert tabela[1:].sum() == 0
    prev = baselines.climatologia(tabela, np.array(pd.to_datetime(["2003-01-02"])).reshape(1, 1))
    assert prev.shape == (1, 1, 1) and prev[0, 0, 0] == pytest.approx(3.0)


def test_climatologia_mediana_resiste_a_surto():
    datas = pd.date_range("2001-01-01", "2003-12-31", freq="7D")
    y = np.ones((len(datas), 1))
    y[0] = 100.0  # um surto na semana 1 de 2001
    media = baselines.ajustar_climatologia(y, datas, "media")
    mediana = baselines.ajustar_climatologia(y, datas, "mediana")
    assert media[0, 0] > 30 and mediana[0, 0] == pytest.approx(1.0)


def test_climatologia_exige_ano_completo_e_estatistica_valida():
    datas = pd.date_range("2001-01-01", periods=10, freq="7D")
    with pytest.raises(ValueError, match="não cobre"):
        baselines.ajustar_climatologia(np.zeros((10, 2)), datas)
    datas = pd.date_range("2001-01-01", "2001-12-31", freq="7D")
    with pytest.raises(ValueError, match="estatistica"):
        baselines.ajustar_climatologia(np.zeros((len(datas), 2)), datas, "moda")


def test_avaliar_usa_as_mesmas_janelas_de_preparar(tabela_longa):
    resultado = baselines.avaliar(tabela_longa, "validacao")
    assert list(resultado.index) == ["persistencia", "climatologia_media", "climatologia_mediana"]
    assert {"mae", "rmse", "mae_surto", "mae_h1", "rmse_h4"} <= set(resultado.columns)
    assert np.isfinite(resultado[["mae", "rmse"]].to_numpy()).all()

    # o alvo dos baselines é o mesmo y_bruto que o LSTM e o STGCN vão usar
    particoes, _ = dados.preparar(tabela_longa)
    _, y, datas = dados.montar_tensor(tabela_longa)
    idx = dados.indices_particoes(datas)["validacao"]
    _, y_alvo, _, _ = dados.janelas(y[..., None], y, idx)
    np.testing.assert_array_equal(y_alvo, particoes["validacao"].y_bruto)


def test_avaliar_nao_usa_o_futuro(tabela_longa):
    """Mudar a validação e o teste não muda a climatologia nem o limiar (vêm do treino)."""
    alterada = tabela_longa.copy()
    futuro = alterada["data"] >= "2013-01-01"
    # zera o futuro: se a climatologia viesse da validação, ela também iria a zero
    alterada.loc[futuro, "taxa"] = 0.0
    nova = baselines.avaliar(alterada)
    # com alvo zero, o MAE da climatologia é a média da própria previsão, que vem do treino
    _, y, datas = dados.montar_tensor(tabela_longa)
    treino = dados.indices_particoes(datas)["treino"]
    tabela = baselines.ajustar_climatologia(y[treino], datas[treino], "media")
    idx = dados.indices_particoes(datas)["validacao"]
    _, _, _, idx_alvo = dados.janelas(y[..., None], y, idx)
    prev = baselines.climatologia(tabela, datas.to_numpy()[idx_alvo])
    assert nova.loc["climatologia_media", "mae"] == pytest.approx(np.abs(prev).mean())


def test_avaliar_recusa_o_treino(tabela_longa):
    with pytest.raises(ValueError, match="particao"):
        baselines.avaliar(tabela_longa, "treino")


@sem_csv
def test_baselines_nos_dados_reais():
    resultado = baselines.avaliar(dados.carregar_tabela(), "validacao")
    # sanidade: a persistência acerta mais que a climatologia no curto prazo
    assert resultado.loc["persistencia", "mae_h1"] < resultado.loc["climatologia_media", "mae_h1"]
    # e o erro cresce com o horizonte
    assert resultado.loc["persistencia", "mae_h1"] < resultado.loc["persistencia", "mae_h4"]
