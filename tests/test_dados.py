"""Testes da carga dos dados.

Os testes com o CSV real são pulados quando o arquivo não existe (por exemplo, na CI).
"""

import pandas as pd
import pytest

from dengue import dados

sem_csv = pytest.mark.skipif(not dados.CAMINHO_CSV.exists(), reason="CSV real ausente")


def test_carregar_csv_inexistente_da_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        dados.carregar_csv(tmp_path / "nao_existe.csv")


def test_carregar_csv_converte_data(tmp_path):
    caminho = tmp_path / "mini.csv"
    caminho.write_text("CD_MUN,Data,x\n1,1999-01-01,0.5\n1,1999-01-08,1.0\n", encoding="utf-8")
    df = dados.carregar_csv(caminho)
    assert pd.api.types.is_datetime64_any_dtype(df["Data"])
    assert len(df) == 2


@sem_csv
def test_formato_do_csv_real():
    df = dados.carregar_csv()
    assert df["CD_MUN"].nunique() == dados.N_MUNICIPIOS
    assert df["Data"].nunique() == dados.N_SEMANAS
    assert len(df) == dados.N_MUNICIPIOS * dados.N_SEMANAS
    assert "Taxa de Internações por Dengue" in df.columns


# --- Limpeza (task 8) ---


def test_limpar_renomeia_descarta_uf_e_ordena(bruto):
    df = dados.limpar(bruto)
    assert list(df.columns) == list(dados.NOMES_COLUNAS.values())
    assert "UF" not in df.columns
    assert df["data"].is_monotonic_increasing
    # dentro de cada semana, os municípios vêm em ordem crescente de código
    primeira = df[df["data"] == df["data"].min()]["municipio"]
    assert primeira.is_monotonic_increasing


def test_limpar_fixa_tipos(bruto):
    df = dados.limpar(bruto)
    assert df["municipio"].dtype == "int64"
    assert pd.api.types.is_datetime64_any_dtype(df["data"])
    assert df["taxa"].dtype == "float64"


def test_limpar_reclama_de_coluna_ausente(bruto):
    with pytest.raises(ValueError, match="ausentes"):
        dados.limpar(bruto.drop(columns="Temperatura Máxima"))


def test_validar_detecta_na(bruto):
    df = dados.limpar(bruto)
    df.loc[5, "taxa"] = float("nan")
    with pytest.raises(ValueError, match="NAs"):
        dados.validar(df)


def test_validar_detecta_semana_faltando(bruto):
    df = dados.limpar(bruto)
    df = df[df["data"] != df["data"].unique()[4]].reset_index(drop=True)
    with pytest.raises(ValueError, match="consecutivas"):
        dados.validar(df)


def test_validar_detecta_municipio_faltando_em_uma_semana(bruto):
    df = dados.limpar(bruto).drop(index=7).reset_index(drop=True)
    with pytest.raises(ValueError, match="grade"):
        dados.validar(df)


def test_validar_detecta_alvo_negativo(bruto):
    df = dados.limpar(bruto)
    df.loc[0, "taxa"] = -1.0
    with pytest.raises(ValueError, match="negativos"):
        dados.validar(df)


def test_pivotar_tem_nos_em_ordem(bruto):
    df = dados.limpar(bruto)
    matriz = dados.pivotar(df, "taxa")
    assert matriz.shape == (10, 3)
    assert list(matriz.columns) == list(dados.municipios(df))
    # o valor na matriz é o mesmo da tabela longa
    linha = df.iloc[4]
    assert matriz.loc[linha["data"], linha["municipio"]] == linha["taxa"]


def test_salvar_e_carregar_tabela_ida_e_volta(bruto, tmp_path):
    df = dados.limpar(bruto)
    caminho = dados.salvar_tabela(df, tmp_path / "tabela.csv")
    lida = dados.carregar_tabela(caminho)
    pd.testing.assert_frame_equal(lida, df, check_dtype=False)


@sem_csv
def test_tabela_real_limpa_tem_formato_esperado():
    df = dados.limpar(dados.carregar_csv())
    assert df.shape == (dados.N_MUNICIPIOS * dados.N_SEMANAS, len(dados.NOMES_COLUNAS))
    assert dados.pivotar(df, dados.ALVO).shape == (dados.N_SEMANAS, dados.N_MUNICIPIOS)
