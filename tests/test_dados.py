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
