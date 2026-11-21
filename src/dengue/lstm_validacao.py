"""Resultado do LSTM sem grafo na validação, com 3 seeds (task 14).

    python -m dengue.lstm_validacao

Gera `results/lstm_validacao.csv`: uma linha por seed, mais a média e o desvio. As colunas
são as mesmas de `results/baselines_validacao.csv`, para comparar lado a lado.
"""

from pathlib import Path

import pandas as pd

from dengue import dados, metricas, modelos, treino

CAMINHO_RESULTADO = dados.RAIZ / "results" / "lstm_validacao.csv"
SEEDS = (0, 1, 2)


def avaliar(df: pd.DataFrame, seeds=SEEDS, config: treino.ConfigTreino | None = None, **kw):
    """Treina um LSTM por seed e devolve as métricas na validação (uma linha por seed).

    `kw` vai para o construtor do `modelos.LSTM` (por exemplo, `oculto=16` nos testes).
    O teste de 2016–2021 não é usado: só o treino e a validação entram aqui.
    """
    config = config or treino.ConfigTreino()
    particoes, _ = dados.preparar(df)
    p_treino, p_val = particoes["treino"], particoes["validacao"]

    _, y, datas = dados.montar_tensor(df)
    limiar = metricas.limiar_surto(y[dados.indices_particoes(datas)["treino"]])

    linhas = []
    for seed in seeds:
        treino.fixar_seed(seed)  # a inicialização dos pesos também depende da seed
        modelo = modelos.LSTM(**kw)
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


def media_desvio(resultado: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta as linhas `media` e `desvio` (desvio amostral entre as seeds)."""
    resumo = pd.DataFrame({"media": resultado.mean(), "desvio": resultado.std(ddof=1)}).T
    return pd.concat([resultado, resumo])


def main(caminho: str | Path = CAMINHO_RESULTADO) -> pd.DataFrame:
    resultado = media_desvio(avaliar(dados.carregar_tabela()))
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    resultado.index.name = "seed"
    resultado.round(4).to_csv(caminho)
    return resultado


if __name__ == "__main__":
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(main())
