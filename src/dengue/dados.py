"""Carga e limpeza dos dados de dengue do Rio de Janeiro.

Fonte: dataset *Brazil-Dengue-Arboviral-Diseases* (Zenodo). Veja `data/README.md`.

Fluxo: `carregar_csv()` (bruto) -> `limpar()` (tabela município × semana validada)
-> `salvar_tabela()` / `carregar_tabela()`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Raiz do repositório (src/dengue/dados.py -> três níveis acima)
RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_CSV = RAIZ / "data" / "dengue_33_rio_de_janeiro.csv"
CAMINHO_TABELA = RAIZ / "data" / "tabela_rj.csv"

N_MUNICIPIOS = 79
N_SEMANAS = 1199

# Nomes curtos, sem acento, para usar no código
NOMES_COLUNAS = {
    "CD_MUN": "municipio",
    "latitude": "latitude",
    "longitude": "longitude",
    "Data": "data",
    "Taxa de Internações por Dengue": "taxa",
    "Índice de Densidade Demográfica": "densidade",
    "Emissão de CH4": "ch4",
    "Emissão de CO2": "co2",
    "Emissão de NO2": "no2",
    "Índice de Pobreza": "pobreza",
    "Índice de Urbanização": "urbanizacao",
    "Temperatura Máxima": "temp_max",
    "Precipitação Média Mensal": "precipitacao",
    "Menor Umidade Relativa": "umidade_min",
}
ALVO = "taxa"
CLIMA = ["temp_max", "precipitacao", "umidade_min"]
SOCIOECONOMICAS = ["densidade", "ch4", "co2", "no2", "pobreza", "urbanizacao"]


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


def limpar(bruto: pd.DataFrame) -> pd.DataFrame:
    """Transforma o CSV bruto na tabela município × semana usada no projeto.

    - renomeia as colunas para nomes curtos (veja `NOMES_COLUNAS`);
    - descarta `UF` (é sempre 33);
    - fixa os tipos (`municipio` inteiro, `data` datetime, o resto float);
    - ordena por (data, municipio). A ordem dos nós do grafo é a dos códigos IBGE crescentes;
    - valida o resultado com `validar()`.
    """
    faltando = set(NOMES_COLUNAS) - set(bruto.columns)
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(faltando)}")

    df = bruto[list(NOMES_COLUNAS)].rename(columns=NOMES_COLUNAS)
    df["municipio"] = df["municipio"].astype("int64")
    df["data"] = pd.to_datetime(df["data"])
    numericas = [c for c in df.columns if c not in ("municipio", "data")]
    df[numericas] = df[numericas].astype("float64")
    df = df.sort_values(["data", "municipio"]).reset_index(drop=True)
    validar(df)
    return df


def validar(df: pd.DataFrame) -> None:
    """Confere a tabela limpa e levanta `ValueError` explicando o primeiro problema."""
    colunas = list(NOMES_COLUNAS.values())
    if list(df.columns) != colunas:
        raise ValueError(f"Colunas diferentes do esperado: {list(df.columns)}")
    if df.isna().any().any():
        raise ValueError(f"Há NAs nas colunas: {df.columns[df.isna().any()].tolist()}")
    if df.duplicated(["data", "municipio"]).any():
        raise ValueError("Há pares (data, municipio) repetidos")

    n_mun, n_sem = df["municipio"].nunique(), df["data"].nunique()
    if len(df) != n_mun * n_sem:
        raise ValueError("A grade não está completa: falta algum município em alguma semana")
    passos = np.diff(np.sort(df["data"].unique())).astype("timedelta64[D]")
    if len(passos) and not (passos == np.timedelta64(7, "D")).all():
        raise ValueError("As semanas não são consecutivas (intervalo diferente de 7 dias)")
    if not df.equals(df.sort_values(["data", "municipio"]).reset_index(drop=True)):
        raise ValueError("A tabela precisa estar ordenada por (data, municipio)")
    if (df.groupby("municipio")[["latitude", "longitude"]].nunique() > 1).any().any():
        raise ValueError("Coordenadas mudam ao longo do tempo para algum município")
    if (df[ALVO] < 0).any():
        raise ValueError("O alvo tem valores negativos")


def municipios(df: pd.DataFrame) -> np.ndarray:
    """Códigos IBGE na ordem dos nós (crescente). O índice i do tensor é o município i."""
    return np.sort(df["municipio"].unique())


def pivotar(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Matriz semanas × municípios de uma coluna (linhas = datas, colunas = nós em ordem)."""
    return df.pivot(index="data", columns="municipio", values=coluna).reindex(
        columns=municipios(df)
    )


def salvar_tabela(df: pd.DataFrame, caminho: str | Path = CAMINHO_TABELA) -> Path:
    """Salva a tabela limpa em CSV (fica em `data/`, fora do git)."""
    validar(df)
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)
    return caminho


def carregar_tabela(caminho: str | Path = CAMINHO_TABELA) -> pd.DataFrame:
    """Lê a tabela limpa. Se ela ainda não existir, gera a partir do CSV bruto e salva."""
    caminho = Path(caminho)
    if not caminho.exists():
        df = limpar(carregar_csv())
        salvar_tabela(df, caminho)
        return df
    df = pd.read_csv(caminho, parse_dates=["data"])
    validar(df)
    return df


def correlacao_defasada(df: pd.DataFrame, coluna: str, lags=range(27)) -> pd.Series:
    """Correlação de Pearson entre `coluna` na semana t - lag e o alvo na semana t.

    É calculada dentro de cada município e depois tirada a média entre eles. Municípios em
    que o alvo é constante (sem variância) no trecho são ignorados.
    """
    x = pivotar(df, coluna).to_numpy()
    y = pivotar(df, ALVO).to_numpy()
    resultado = {}
    for lag in lags:
        a, b = x[: len(x) - lag], y[lag:]
        a = (a - a.mean(0)) / a.std(0)
        desvio_b = b.std(0)
        b = (b - b.mean(0)) / np.where(desvio_b == 0, np.nan, desvio_b)
        resultado[lag] = np.nanmean((a * b).mean(0))
    return pd.Series(resultado, name=coluna)
