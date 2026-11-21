"""Dados sintéticos com o mesmo formato do CSV bruto, para os testes rodarem sem o arquivo real."""

import numpy as np
import pandas as pd
import pytest

from dengue import dados


def fazer_bruto(n_municipios: int = 3, n_semanas: int = 10, inicio: str = "1999-01-01"):
    """CSV bruto falso: municípios com códigos fora de ordem e linhas embaralhadas."""
    rng = np.random.default_rng(0)
    datas = pd.date_range(inicio, periods=n_semanas, freq="7D")
    codigos = 3300000 + np.arange(n_municipios, 0, -1)  # decrescentes de propósito
    linhas = []
    for i, cod in enumerate(codigos):
        for d in datas:
            linha = {col: rng.uniform(0, 10) for col in dados.NOMES_COLUNAS}
            linha.update(CD_MUN=cod, Data=d, latitude=-22.0 - i, longitude=-43.0 + i, UF=33)
            linhas.append(linha)
    bruto = pd.DataFrame(linhas)
    return bruto.sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.fixture
def bruto():
    return fazer_bruto()
