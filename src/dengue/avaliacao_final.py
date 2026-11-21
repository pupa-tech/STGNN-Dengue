"""Avaliação final no teste, 2016 a 2021 (task 21). Roda **uma única vez**.

    python -m dengue.avaliacao_final

Todos os modelos são treinados só com o treino, e todas as escolhas (época, η, perda,
agenda) vieram da validação. Aqui o teste é aberto pela primeira e única vez. Se os
resultados já existem, o script se recusa a rodar de novo: repetir seria escolher pelo teste.

Modelos comparados:
- `persistencia` e `climatologia_mediana`: baselines sem treino (task 13);
- `lstm`: LSTM sem grafo, Adam com early stopping (task 14);
- `stgcn_base`: STGCN geo com Adam 1e-3, agenda constante e MSE (baseline do desafio 3);
- `stgcn_sem_grafo`: a configuração final com A = I (ablação da task 17);
- `stgcn_final`: a configuração congelada em `configs/final.yaml`.

Saídas em `results/`:
- `teste_final.csv`: uma linha por (modelo, seed, partição), com as métricas na validação
  (para conferir com as tasks anteriores) e no teste;
- `teste_final_series.csv`: a taxa média estadual real e prevista (média das seeds) por
  data-alvo do teste, em h = 1 e h = 4;
- `teste_final_municipios.csv`: o MAE de cada município no teste, por modelo (média das seeds).
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from dengue import baselines, dados, experimento, metricas, modelos, treino

DIR_RESULTADOS = dados.RAIZ / "results"
CONFIG_FINAL = dados.RAIZ / "configs" / "final.yaml"
TREINADOS = ("lstm", "stgcn_base", "stgcn_sem_grafo", "stgcn_final")
PARTICOES = ("validacao", "teste")
HORIZONTES_SERIE = (1, 4)


def configs_stgcn(final: experimento.Config) -> dict[str, experimento.Config]:
    """As três variantes do STGCN, todas com o mesmo orçamento de passos da final."""
    base = dataclasses.replace(
        final, id="final_stgcn_base", otimizador="adam", lr=1e-3, agenda="const", perda="mse"
    )
    sem_grafo = dataclasses.replace(final, id="final_stgcn_sem_grafo", grafo="identidade")
    return {"stgcn_base": base, "stgcn_sem_grafo": sem_grafo, "stgcn_final": final}


def prever_baselines(y: np.ndarray, datas: pd.DatetimeIndex, particao: str) -> dict:
    """Previsões [S, H, N] da persistência e da climatologia (tabela ajustada só no treino)."""
    indices = dados.indices_particoes(datas)
    y_entrada, _, _, idx_alvo = dados.janelas(y[..., None], y, indices[particao])
    tabela = baselines.ajustar_climatologia(
        y[indices["treino"]], datas[indices["treino"]], "mediana"
    )
    return {
        "persistencia": baselines.persistencia(y_entrada[..., 0]),
        "climatologia_mediana": baselines.climatologia(tabela, datas.to_numpy()[idx_alvo]),
    }


def treinar_e_prever(
    nome: str,
    seed: int,
    df: pd.DataFrame,
    particoes: dict,
    cfgs: dict[str, experimento.Config],
    config_lstm: treino.ConfigTreino,
) -> dict[str, np.ndarray]:
    """Treina um modelo (só treino + validação) e devolve as previsões das duas partições."""
    treino.fixar_seed(seed)
    p_treino, p_val = particoes["treino"], particoes["validacao"]
    if nome == "lstm":
        modelo = modelos.LSTM()
        treino.treinar(modelo, p_treino, p_val, dataclasses.replace(config_lstm, seed=seed))
    else:
        modelo = experimento.criar_modelo(cfgs[nome], df)
        experimento.treinar_por_passos(modelo, p_treino, p_val, cfgs[nome], seed)
    return {p: treino.prever(modelo, particoes[p]) for p in PARTICOES}


def avaliar(
    df: pd.DataFrame,
    final: experimento.Config,
    config_lstm: treino.ConfigTreino | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Roda tudo e devolve (métricas, séries estaduais, MAE por município)."""
    config_lstm = config_lstm or treino.ConfigTreino()
    cfgs = configs_stgcn(final)
    particoes, _ = dados.preparar(df)
    _, y, datas = dados.montar_tensor(df)
    limiar = metricas.limiar_surto(y[dados.indices_particoes(datas)["treino"]])

    # previsões[modelo][particao] = lista de [S, H, N], uma por seed
    previsoes = {}
    for p in PARTICOES:
        for nome, y_hat in prever_baselines(y, datas, p).items():
            previsoes.setdefault(nome, {})[p] = [y_hat]
    for nome in TREINADOS:
        for seed in final.seeds:
            print(f"[treino] {nome} seed {seed}", flush=True)
            for p, y_hat in treinar_e_prever(nome, seed, df, particoes, cfgs, config_lstm).items():
                previsoes.setdefault(nome, {}).setdefault(p, []).append(y_hat)

    linhas = []
    for nome, por_particao in previsoes.items():
        for p, lista in por_particao.items():
            y_real = particoes[p].y_bruto
            for i, y_hat in enumerate(lista):
                linhas.append(
                    {
                        "modelo": nome,
                        "particao": p,
                        "seed": final.seeds[i] if nome in TREINADOS else np.nan,
                        **metricas.resumo(y_real, y_hat, limiar),
                        **metricas.por_regime(y_real, y_hat, limiar),
                    }
                )
    resultado = pd.DataFrame(linhas)

    teste = particoes["teste"]
    medias = {nome: np.mean(por_p["teste"], axis=0) for nome, por_p in previsoes.items()}
    series = []
    for h in HORIZONTES_SERIE:
        bloco = {
            "data": teste.datas_alvo[:, h - 1],
            "h": h,
            "real": teste.y_bruto[:, h - 1].mean(-1),
        }
        bloco.update({nome: y_hat[:, h - 1].mean(-1) for nome, y_hat in medias.items()})
        series.append(pd.DataFrame(bloco))

    coords = df.groupby("municipio")[["latitude", "longitude"]].first()
    coords = coords.loc[dados.municipios(df)]
    por_municipio = coords.assign(
        **{
            nome: np.mean(
                [np.abs(teste.y_bruto - y_hat).mean(axis=(0, 1)) for y_hat in lista], axis=0
            )
            for nome, lista in ((n, p["teste"]) for n, p in previsoes.items())
        },
        taxa_media=teste.y_bruto.mean(axis=(0, 1)),
    )
    return resultado, pd.concat(series, ignore_index=True), por_municipio


def main(argv=None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Avaliação final no teste (uma única vez).")
    parser.add_argument("--resultados", default=DIR_RESULTADOS, help="pasta de saída")
    args = parser.parse_args(argv)
    pasta = Path(args.resultados)
    caminhos = {
        "metricas": pasta / "teste_final.csv",
        "series": pasta / "teste_final_series.csv",
        "municipios": pasta / "teste_final_municipios.csv",
    }
    if any(c.exists() for c in caminhos.values()):
        raise SystemExit(
            "O teste já foi avaliado (results/teste_final*.csv existe). Ele é usado uma única "
            "vez: rodar de novo seria escolher pelo teste. Use os arquivos que já existem."
        )
    final = experimento.carregar_config(CONFIG_FINAL)
    resultado, series, por_municipio = avaliar(dados.carregar_tabela(), final)
    pasta.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(caminhos["metricas"], index=False)
    series.round(5).to_csv(caminhos["series"], index=False)
    por_municipio.round(5).to_csv(caminhos["municipios"])
    return resultado


if __name__ == "__main__":
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        r = main()
        print(r.groupby(["particao", "modelo"], sort=False)[["mae", "rmse", "mae_surto"]].mean())
