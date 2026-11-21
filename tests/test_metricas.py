"""Testes das métricas, com valores calculados à mão."""

import numpy as np
import pytest

from dengue import metricas


def test_previsao_perfeita_tem_erro_zero():
    y = np.random.default_rng(0).random((5, 4, 3))
    assert metricas.mae(y, y) == 0.0
    assert metricas.rmse(y, y) == 0.0


def test_mae_e_rmse_a_mao():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    y_hat = np.array([1.0, -1.0, 3.0, -3.0])
    assert metricas.mae(y, y_hat) == pytest.approx(2.0)  # (1+1+3+3)/4
    assert metricas.rmse(y, y_hat) == pytest.approx(np.sqrt(5.0))  # (1+1+9+9)/4 = 5


def test_rmse_nunca_menor_que_mae():
    rng = np.random.default_rng(1)
    y, y_hat = rng.random((10, 4, 79)), rng.random((10, 4, 79))
    assert metricas.rmse(y, y_hat) >= metricas.mae(y, y_hat)


def test_formatos_diferentes_dao_erro():
    with pytest.raises(ValueError, match="Formatos"):
        metricas.mae(np.zeros(3), np.zeros(4))


def test_por_horizonte_separa_cada_h():
    y = np.zeros((2, 4, 3))
    y_hat = np.zeros((2, 4, 3))
    y_hat[:, 2, :] = 2.0  # só o h=3 erra, sempre por 2
    r = metricas.por_horizonte(y, y_hat)
    assert r["mae"].shape == (4,)
    np.testing.assert_allclose(r["mae"], [0, 0, 2, 0])
    np.testing.assert_allclose(r["rmse"], [0, 0, 2, 0])


def test_por_horizonte_exige_tres_dimensoes():
    with pytest.raises(ValueError, match="S, H, N"):
        metricas.por_horizonte(np.zeros((4, 3)), np.zeros((4, 3)))


def test_limiar_surto_usa_media_estadual():
    # 10 semanas, 2 municípios: a média estadual vai de 0 a 9
    medias = np.arange(10, dtype=float)
    y_treino = np.stack([medias - 1, medias + 1], axis=1)
    assert metricas.limiar_surto(y_treino, q=0.9) == pytest.approx(np.quantile(medias, 0.9))


def test_limiar_surto_exige_duas_dimensoes():
    with pytest.raises(ValueError, match="T, N"):
        metricas.limiar_surto(np.zeros(10))


def test_mae_surto_so_olha_semanas_de_surto():
    y = np.zeros((2, 1, 2))
    y[1] = 10.0  # só a amostra 1 é surto (média 10 > limiar 5)
    y_hat = y.copy()
    y_hat[0] += 100.0  # erro enorme fora do surto: não deve contar
    y_hat[1] += 1.0  # erro de 1 dentro do surto
    assert metricas.mae_surto(y, y_hat, limiar=5.0) == pytest.approx(1.0)


def test_mae_surto_sem_surto_devolve_nan():
    y = np.zeros((2, 4, 3))
    assert np.isnan(metricas.mae_surto(y, y, limiar=1.0))


def test_resumo_tem_todas_as_chaves():
    y = np.ones((3, 4, 5))
    r = metricas.resumo(y, y * 2, limiar=0.5)
    por_h = {f"{m}_h{h}" for m in ("mae", "rmse") for h in range(1, 5)}
    esperadas = {"mae", "rmse", "mae_surto"} | por_h
    assert set(r) == esperadas
    assert r["mae"] == pytest.approx(1.0)
