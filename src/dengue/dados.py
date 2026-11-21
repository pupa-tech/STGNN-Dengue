"""Carga dos dados de dengue do Rio de Janeiro.

Fonte: dataset *Brazil-Dengue-Arboviral-Diseases* (Zenodo). Veja `data/README.md`.
"""

from pathlib import Path

import pandas as pd

# Raiz do repositório (src/dengue/dados.py -> três níveis acima)
RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_CSV = RAIZ / "data" / "dengue_33_rio_de_janeiro.csv"

N_MUNICIPIOS = 79
N_SEMANAS = 1199


def carregar_csv(caminho: str | Path = CAMINHO_CSV) -> pd.DataFrame:
    """Lê o CSV bruto do RJ e devolve um DataFrame com a coluna `Data` como data.

    Cada linha é um município (`CD_MUN`) em uma semana (`Data`).
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"CSV não encontrado em {caminho}. Veja data/README.md para baixar os dados."
        )
    return pd.read_csv(caminho, parse_dates=["Data"], encoding="utf-8")
