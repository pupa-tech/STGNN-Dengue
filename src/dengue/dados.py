"""Carga e limpeza dos dados de dengue do Rio de Janeiro.

Fonte: dataset *Brazil-Dengue-Arboviral-Diseases* (Zenodo). Veja `data/README.md`.

Fluxo: `carregar_csv()` (bruto) -> `limpar()` (tabela município × semana validada)
-> `salvar_tabela()` / `carregar_tabela()`.
"""

from dataclasses import dataclass
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


# --- Split temporal e janelas (task 10) ---

# Entradas dos modelos: a própria taxa e o clima (decisão G2, task 10)
ENTRADAS = [ALVO, *CLIMA]
W = 4  # semanas de entrada
H = 4  # semanas previstas (horizonte)

# Partições por ano, em ordem cronológica, sem sobreposição
PARTICOES = {
    "treino": ("1999-01-01", "2012-12-31"),
    "validacao": ("2013-01-01", "2015-12-31"),
    "teste": ("2016-01-01", "2021-12-31"),
}


def montar_tensor(df: pd.DataFrame, entradas=ENTRADAS):
    """Converte a tabela longa em tensores alinhados no tempo.

    Devolve `(X, y, datas)`:
    - `X`: array [T, N, F] com as colunas `entradas` (nós na ordem de `municipios`);
    - `y`: array [T, N] com o alvo bruto (taxa);
    - `datas`: as T semanas, em ordem.
    """
    X = np.stack([pivotar(df, c).to_numpy() for c in entradas], axis=-1)
    alvo = pivotar(df, ALVO)
    return X.astype(np.float32), alvo.to_numpy().astype(np.float32), pd.DatetimeIndex(alvo.index)


def indices_particoes(datas: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Índices (no eixo do tempo) das semanas de cada partição."""
    return {
        nome: np.flatnonzero((datas >= inicio) & (datas <= fim))
        for nome, (inicio, fim) in PARTICOES.items()
    }


def janelas(X: np.ndarray, y: np.ndarray, indices: np.ndarray, w: int = W, h: int = H):
    """Recorta janelas deslizantes que ficam **inteiras dentro** de uma partição.

    Para cada semana t da partição, a entrada são as semanas t-w+1..t e o alvo são as
    semanas t+1..t+h. Uma janela só é aceita se todas as w + h semanas estiverem na
    partição, então nenhuma semana de outra partição entra, nem como entrada.

    Devolve `(x, alvo, idx_entrada, idx_alvo)`:
    - `x`: [S, w, N, F]; `alvo`: [S, h, N];
    - `idx_entrada`: [S, w] e `idx_alvo`: [S, h], índices das semanas no tempo total.
    """
    indices = np.asarray(indices)
    if len(indices) and not (np.diff(indices) == 1).all():
        raise ValueError("Os índices da partição precisam ser semanas consecutivas")
    n_amostras = len(indices) - w - h + 1
    if n_amostras <= 0:
        raise ValueError(f"Partição curta demais para w={w} e h={h}")
    inicio = indices[0] + np.arange(n_amostras)
    idx_entrada = inicio[:, None] + np.arange(w)
    idx_alvo = inicio[:, None] + w + np.arange(h)
    return X[idx_entrada], y[idx_alvo], idx_entrada, idx_alvo


# --- Normalização ajustada só no treino (task 11) ---


def transformar_alvo(y: np.ndarray) -> np.ndarray:
    """log1p(y): comprime a cauda longa do alvo e mantém os zeros em zero."""
    return np.log1p(y)


def inverter_alvo(y_log: np.ndarray) -> np.ndarray:
    """Volta da escala log1p para a taxa bruta. Previsões negativas viram zero."""
    return np.clip(np.expm1(y_log), 0, None)


@dataclass
class Normalizador:
    """z-score por variável, com média e desvio calculados **só no treino**.

    O canal do alvo recebe `log1p` antes do z-score, igual ao alvo dos modelos.
    """

    media: np.ndarray  # [F]
    desvio: np.ndarray  # [F]
    canal_alvo: int = 0

    @classmethod
    def ajustar(cls, X_treino: np.ndarray, canal_alvo: int = 0) -> "Normalizador":
        """Calcula média e desvio de cada variável a partir de X_treino [T_treino, N, F]."""
        X_log = cls._log_no_alvo(X_treino, canal_alvo)
        media = X_log.mean(axis=(0, 1))
        desvio = X_log.std(axis=(0, 1))
        desvio = np.where(desvio == 0, 1.0, desvio)  # variável constante: não divide por zero
        return cls(media.astype(np.float32), desvio.astype(np.float32), canal_alvo)

    def transformar(self, X: np.ndarray) -> np.ndarray:
        """Aplica log1p no canal do alvo e o z-score do treino em X [..., F]."""
        return ((self._log_no_alvo(X, self.canal_alvo) - self.media) / self.desvio).astype(
            np.float32
        )

    @staticmethod
    def _log_no_alvo(X: np.ndarray, canal_alvo: int) -> np.ndarray:
        X = np.array(X, dtype=np.float64)  # cópia: nunca altera o array original
        X[..., canal_alvo] = transformar_alvo(X[..., canal_alvo])
        return X


@dataclass
class Particao:
    """Janelas prontas de uma partição."""

    x: np.ndarray  # [S, W, N, F] entradas normalizadas
    y: np.ndarray  # [S, H, N] alvo em log1p (o que os modelos aprendem)
    y_bruto: np.ndarray  # [S, H, N] alvo na escala original (para as métricas)
    datas_entrada: np.ndarray  # [S, W]
    datas_alvo: np.ndarray  # [S, H]


def preparar(df: pd.DataFrame, entradas=ENTRADAS, w: int = W, h: int = H):
    """Pipeline completo: tensor -> partições -> normalização do treino -> janelas.

    Devolve `(particoes, normalizador)`, em que `particoes` é um dict
    {"treino", "validacao", "teste"} -> `Particao`.
    """
    X, y, datas = montar_tensor(df, entradas)
    indices = indices_particoes(datas)
    normalizador = Normalizador.ajustar(X[indices["treino"]], canal_alvo=entradas.index(ALVO))
    X_norm = normalizador.transformar(X)
    y_log = transformar_alvo(y).astype(np.float32)

    particoes = {}
    for nome, idx in indices.items():
        x, alvo_log, idx_entrada, idx_alvo = janelas(X_norm, y_log, idx, w, h)
        particoes[nome] = Particao(
            x=x,
            y=alvo_log,
            y_bruto=y[idx_alvo],
            datas_entrada=datas.to_numpy()[idx_entrada],
            datas_alvo=datas.to_numpy()[idx_alvo],
        )
    return particoes, normalizador
