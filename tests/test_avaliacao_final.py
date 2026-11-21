"""Testes da avaliação final no teste (task 21)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from conftest import fazer_bruto

from dengue import avaliacao_final, dados, experimento, treino

LSTM_RAPIDO = treino.ConfigTreino(max_epocas=2, paciencia=1)


@pytest.fixture(scope="module")
def df():
    # 1.199 semanas a partir de 1999 cobrem treino, validação e teste
    return dados.limpar(fazer_bruto(n_municipios=6, n_semanas=dados.N_SEMANAS))


def _final_rapida(**kw):
    return experimento.Config(
        id="t",
        otimizador="adamw",
        lr=1e-2,
        agenda="warmup_cos",
        perda="huber",
        passos=6,
        lote=64,
        log_cada=3,
        seeds=[0, 1],
        **kw,
    )


def test_config_final_vem_das_decisoes_da_validacao():
    cfg = experimento.carregar_config(avaliacao_final.CONFIG_FINAL)
    assert (cfg.modelo, cfg.grafo) == ("stgcn", "geo")
    assert (cfg.otimizador, cfg.lr, cfg.agenda) == ("adamw", 0.01, "warmup_cos")  # task 19
    assert cfg.perda == "huber"  # task 20
    assert cfg.passos == 920 and cfg.seeds == [0, 1, 2]


def test_variantes_do_stgcn_so_mudam_o_que_devem():
    final = experimento.carregar_config(avaliacao_final.CONFIG_FINAL)
    cfgs = avaliacao_final.configs_stgcn(final)
    assert cfgs["stgcn_final"] is final
    sem = dataclasses.asdict(cfgs["stgcn_sem_grafo"])
    assert {k for k, v in sem.items() if dataclasses.asdict(final)[k] != v} == {"id", "grafo"}
    base = cfgs["stgcn_base"]
    assert (base.otimizador, base.lr, base.agenda, base.perda) == ("adam", 1e-3, "const", "mse")
    assert base.passos == final.passos and base.grafo == "geo"


def test_avaliar_formato_das_saidas(df):
    resultado, series, por_municipio = avaliacao_final.avaliar(df, _final_rapida(), LSTM_RAPIDO)
    modelos = ["persistencia", "climatologia_mediana", *avaliacao_final.TREINADOS]
    assert set(resultado["modelo"]) == set(modelos)
    contagem = resultado.groupby(["modelo", "particao"]).size()
    assert contagem.loc["persistencia"].tolist() == [1, 1]
    assert (contagem.loc["stgcn_final"] == 2).all()  # 2 seeds em cada partição
    assert {"mae", "rmse", "mae_surto", "mae_calmaria", "vies_surto"} <= set(resultado.columns)

    n_janelas_teste = len(dados.preparar(df)[0]["teste"].y)
    assert len(series) == n_janelas_teste * len(avaliacao_final.HORIZONTES_SERIE)
    assert set(modelos) <= set(series.columns) and series["data"].min() >= pd.Timestamp("2016")
    assert len(por_municipio) == 6 and set(modelos) <= set(por_municipio.columns)


def test_validacao_nao_depende_do_teste(df):
    """Multiplicar o teste por 1000 não muda nada na validação: o teste não influencia
    nenhum treino, nenhuma escolha de época e nenhuma normalização."""
    final = _final_rapida()
    original, _, _ = avaliacao_final.avaliar(df, final, LSTM_RAPIDO)
    alterado = df.copy()
    no_teste = alterado["data"] >= pd.Timestamp(dados.PARTICOES["teste"][0])
    alterado.loc[no_teste, "taxa"] *= 1000
    mexido, _, _ = avaliacao_final.avaliar(alterado, final, LSTM_RAPIDO)

    def val(r):
        return r[r["particao"] == "validacao"].drop(columns="particao").reset_index(drop=True)

    pd.testing.assert_frame_equal(val(original), val(mexido))
    teste_o = original.loc[original["particao"] == "teste", "mae"].to_numpy()
    teste_m = mexido.loc[mexido["particao"] == "teste", "mae"].to_numpy()
    assert not np.allclose(teste_o, teste_m)  # o teste, sim, muda


def test_main_se_recusa_a_repetir(tmp_path):
    (tmp_path / "teste_final.csv").write_text("ja rodou\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="uma única"):
        avaliacao_final.main(["--resultados", str(tmp_path)])
