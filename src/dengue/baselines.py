"""Baselines ingênuos: persistência e climatologia (task 13).

São os modelos triviais que qualquer modelo treinado precisa superar. Nenhum deles aprende
pesos: a persistência repete a última semana observada, e a climatologia prevê a média (ou a
mediana) histórica daquela semana do ano, calculada **só no treino**.

Tudo aqui trabalha na escala original da taxa, igual às métricas. As janelas são as mesmas de
`dados.janelas` (W=4, H=4), então a comparação com o LSTM e o STGCN é justa.

Para gerar a tabela da validação:

    python -m dengue.baselines
"""

from pathlib import Path

import numpy as np
import pandas as pd

from dengue import dados, metricas

CAMINHO_RESULTADO = dados.RAIZ / "results" / "baselines_validacao.csv"


def persistencia(y_entrada: np.ndarray, h: int = dados.H) -> np.ndarray:
    """Repete a última semana da entrada nos h passos à frente.

    `y_entrada` tem formato [S, W, N] (taxa bruta). Devolve [S, h, N].
    """
    y_entrada = np.asarray(y_entrada)
    if y_entrada.ndim != 3:
        raise ValueError(f"Esperado formato [S, W, N], recebido {y_entrada.shape}")
    ultima = y_entrada[:, -1:, :]  # [S, 1, N]
    return np.repeat(ultima, h, axis=1)


def semana_do_ano(datas) -> np.ndarray:
    """Semana do ano de 1 a 52, contada a partir de 1º de janeiro.

    As semanas do dataset começam em dias variados; dias 365 e 366 (que dariam uma
    "semana 53" com um ou dois dias) entram na semana 52.
    """
    dia = pd.DatetimeIndex(np.ravel(datas)).day_of_year.to_numpy()
    semana = np.minimum((dia - 1) // 7 + 1, 52)
    return semana.reshape(np.shape(datas))


def ajustar_climatologia(y_treino: np.ndarray, datas_treino, estatistica: str = "media"):
    """Tabela [52, N] com a média ou a mediana de cada semana do ano, por município.

    Só passe semanas de treino: usar a validação aqui seria vazamento.
    """
    y_treino = np.asarray(y_treino, dtype=float)
    if y_treino.ndim != 2:
        raise ValueError(f"Esperado formato [T, N], recebido {y_treino.shape}")
    funcoes = {"media": np.mean, "mediana": np.median}
    if estatistica not in funcoes:
        raise ValueError(f"estatistica deve ser uma de {sorted(funcoes)}")
    semanas = semana_do_ano(datas_treino)
    faltando = set(range(1, 53)) - set(semanas.tolist())
    if faltando:
        raise ValueError(f"O treino não cobre as semanas do ano {sorted(faltando)}")
    return np.stack([funcoes[estatistica](y_treino[semanas == s], axis=0) for s in range(1, 53)])


def climatologia(tabela: np.ndarray, datas_alvo) -> np.ndarray:
    """Previsão pela tabela [52, N] para as datas-alvo [S, H]. Devolve [S, H, N]."""
    return tabela[semana_do_ano(datas_alvo) - 1]


def avaliar(df: pd.DataFrame, particao: str = "validacao") -> pd.DataFrame:
    """Roda os baselines numa partição e devolve uma linha de métricas por baseline.

    A climatologia e o limiar de surto vêm só do treino. O teste só deve ser avaliado uma
    vez, na task 21, então `particao="teste"` precisa ser pedido de propósito.
    """
    _, y, datas = dados.montar_tensor(df)
    indices = dados.indices_particoes(datas)
    if particao not in indices or particao == "treino":
        raise ValueError("particao deve ser 'validacao' ou 'teste'")

    idx_treino = indices["treino"]
    limiar = metricas.limiar_surto(y[idx_treino])

    # mesmas janelas de dados.preparar, mas com o alvo bruto também na entrada
    y_entrada, y_alvo, _, idx_alvo = dados.janelas(y[..., None], y, indices[particao])
    y_entrada = y_entrada[..., 0]
    datas_alvo = datas.to_numpy()[idx_alvo]

    previsoes = {"persistencia": persistencia(y_entrada)}
    for estatistica in ("media", "mediana"):
        tabela = ajustar_climatologia(y[idx_treino], datas[idx_treino], estatistica)
        previsoes[f"climatologia_{estatistica}"] = climatologia(tabela, datas_alvo)

    linhas = [
        {"modelo": nome, **metricas.resumo(y_alvo, y_hat, limiar)}
        for nome, y_hat in previsoes.items()
    ]
    return pd.DataFrame(linhas).set_index("modelo")


def main(caminho: str | Path = CAMINHO_RESULTADO) -> pd.DataFrame:
    """Gera `results/baselines_validacao.csv` a partir da tabela limpa."""
    resultado = avaliar(dados.carregar_tabela(), "validacao")
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    resultado.round(4).to_csv(caminho)
    return resultado


if __name__ == "__main__":
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(main())
