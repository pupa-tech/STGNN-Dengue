"""Resultado do STGCN na validação, com 3 seeds (tasks 16 e 17).

    python -m dengue.stgcn_validacao   # grafo geográfico e sem grafo (identidade)
    python -m dengue.stgcn_validacao

Gera `results/stgcn_validacao.csv`: uma linha por (grafo, seed), mais a média e o desvio
de cada grafo. As colunas são as mesmas de `results/lstm_validacao.csv`.
"""

from pathlib import Path

import pandas as pd

from dengue import dados, grafo, metricas, modelos, treino
from dengue.lstm_validacao import media_desvio

CAMINHO_RESULTADO = dados.RAIZ / "results" / "stgcn_validacao.csv"
SEEDS = (0, 1, 2)


def avaliar(
    df: pd.DataFrame,
    tipo_grafo: str = "geo",
    seeds=SEEDS,
    config: treino.ConfigTreino | None = None,
    **kw,
) -> pd.DataFrame:
    """Treina um STGCN por seed e devolve as métricas na validação (uma linha por seed).

    `kw` vai para o construtor do `modelos.STGCN` (por exemplo, `oculto=8` nos testes).
    O teste de 2016–2021 não é usado.
    """
    config = config or treino.ConfigTreino()
    particoes, _ = dados.preparar(df)
    p_treino, p_val = particoes["treino"], particoes["validacao"]
    lap = grafo.laplaciana_do_projeto(df, tipo_grafo)

    _, y, datas = dados.montar_tensor(df)
    limiar = metricas.limiar_surto(y[dados.indices_particoes(datas)["treino"]])

    linhas = []
    for seed in seeds:
        treino.fixar_seed(seed)  # a inicialização dos pesos também depende da seed
        modelo = modelos.STGCN(lap, **kw)
        cfg = treino.ConfigTreino(**{**config.__dict__, "seed": seed})
        hist = treino.treinar(modelo, p_treino, p_val, cfg)
        y_hat = treino.prever(modelo, p_val)
        linhas.append(
            {
                "seed": seed,
                "melhor_epoca": hist.melhor_epoca,
                "perda_val_log1p": min(hist.perda_validacao),
                **metricas.resumo(p_val.y_bruto, y_hat, limiar),
            }
        )
    return pd.DataFrame(linhas).set_index("seed")


def main(caminho: str | Path = CAMINHO_RESULTADO, tipos=grafo.TIPOS_GRAFO) -> pd.DataFrame:
def main(caminho: str | Path = CAMINHO_RESULTADO, tipos=("geo",)) -> pd.DataFrame:
    df = dados.carregar_tabela()
    partes = []
    for tipo in tipos:
        resultado = media_desvio(avaliar(df, tipo))
        resultado.index = resultado.index.astype(str)
        resultado.insert(0, "grafo", tipo)
        partes.append(resultado)
    resultado = pd.concat(partes)
    resultado.index.name = "seed"
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    resultado.round(4).to_csv(caminho)
    return resultado


if __name__ == "__main__":
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(main())
