"""Figuras finais do README (task 22).

    python -m dengue.figuras

Só lê os CSV que já estão em `results/`, sem treinar nada e sem abrir o teste de novo. Salva
três PNGs em `results/figuras/`:

- `resultado_teste.png`: MAE, RMSE e MAE em surto de cada modelo no teste (task 21);
- `serie_estadual.png`: taxa média estadual real e prevista ao longo do teste, em h = 4;
- `otimizadores_eta.png`: perda na validação × η para cada otimizador (task 19, etapa 2).
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")  # salva em arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dengue import dados  # noqa: E402

DIR_RESULTADOS = dados.RAIZ / "results"

NOMES = {
    "persistencia": "Persistência",
    "climatologia_mediana": "Climatologia (mediana)",
    "lstm": "LSTM sem grafo",
    "stgcn_base": "STGCN, protocolo antigo",
    "stgcn_sem_grafo": "STGCN final sem grafo",
    "stgcn_final": "STGCN final",
}
CORES = {
    "persistencia": "tab:gray",
    "climatologia_mediana": "silver",
    "lstm": "tab:orange",
    "stgcn_base": "lightsteelblue",
    "stgcn_sem_grafo": "tab:cyan",
    "stgcn_final": "tab:blue",
}
OTIMIZADORES = {
    "sgd": "SGD",
    "sgd_nesterov": "SGD Nesterov",
    "adam": "Adam",
    "adamw": "AdamW",
    "muon": "Muon",
}


def resultado_teste(tabela: pd.DataFrame, caminho: Path) -> Path:
    """Barras horizontais com média ± desvio das 3 seeds, um painel por métrica."""
    tabela = tabela.set_index("modelo").loc[list(NOMES)]
    metricas = {"mae": "MAE geral", "rmse": "RMSE geral", "mae_surto": "MAE nas semanas de surto"}
    fig, eixos = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (m, titulo) in zip(eixos, metricas.items(), strict=True):
        ax.barh(
            [NOMES[i] for i in tabela.index],
            tabela[f"{m}_media"],
            xerr=tabela[f"{m}_desvio"],
            capsize=3,
            color=[CORES[i] for i in tabela.index],
        )
        ax.set_title(titulo)
    eixos[0].invert_yaxis()
    fig.suptitle("Teste 2016 a 2021 (menor é melhor; barra de erro = desvio entre 3 seeds)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def serie_estadual(series: pd.DataFrame, caminho: Path, h: int = 4) -> Path:
    """Taxa média estadual real × prevista por data-alvo do teste."""
    s = series[series["h"] == h].sort_values("data")
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(s["data"], s["real"], color="k", lw=2, label="real")
    for m in ("persistencia", "lstm", "stgcn_final"):
        ax.plot(s["data"], s[m], color=CORES[m], lw=1.2, label=NOMES[m])
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_yticks([0, 0.25, 0.5, 1, 2, 4, 8])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.set(ylabel="taxa média estadual (symlog)", title=f"Teste, previsão {h} semanas à frente")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def otimizadores_eta(experimentos: pd.DataFrame, caminho: Path) -> Path:
    """Curva perda na validação × η, uma linha por otimizador (warmup + cosseno, seed 0)."""
    e = experimentos[
        (experimentos["etapa"] == "etapa2")
        & (experimentos["agenda"] == "warmup_cos")
        & (experimentos["seed"] == 0)
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    for otm, nome in OTIMIZADORES.items():
        c = e[e["otimizador"] == otm].sort_values("lr")
        if not c.empty:
            ax.plot(c["lr"], c["melhor_val_loss"], marker="o", label=nome)
    ax.set_xscale("log")
    ax.set(
        xlabel="taxa de aprendizado η (escala log)",
        ylabel="melhor perda na validação (MSE em log1p)",
        title="Task 19: 920 passos, warmup + cosseno, seed 0",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def main(resultados: str | Path = DIR_RESULTADOS) -> list[Path]:
    resultados = Path(resultados)
    saida = resultados / "figuras"
    saida.mkdir(parents=True, exist_ok=True)
    tabela = pd.read_csv(resultados / "teste_final_tabela.csv")
    series = pd.read_csv(resultados / "teste_final_series.csv", parse_dates=["data"])
    experimentos = pd.read_csv(resultados / "experimentos.csv")
    return [
        resultado_teste(tabela, saida / "resultado_teste.png"),
        serie_estadual(series, saida / "serie_estadual.png"),
        otimizadores_eta(experimentos, saida / "otimizadores_eta.png"),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera as figuras finais do README.")
    parser.add_argument("--resultados", default=DIR_RESULTADOS, help="pasta com os CSV")
    for p in main(parser.parse_args().resultados):
        print(p)
