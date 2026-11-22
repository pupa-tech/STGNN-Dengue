"""As figuras do README são geradas só a partir dos CSV de results/."""

import pandas as pd

from dengue import figuras


def _csvs_falsos(pasta):
    linhas = []
    for i, m in enumerate(figuras.NOMES):
        linha = {"modelo": m}
        for c in ("mae", "rmse", "mae_surto"):
            linha[f"{c}_media"] = 0.1 * (i + 1)
            linha[f"{c}_desvio"] = 0.01
        linhas.append(linha)
    pd.DataFrame(linhas).to_csv(pasta / "teste_final_tabela.csv", index=False)

    datas = pd.date_range("2016-01-01", periods=10, freq="7D")
    series = pd.DataFrame({"data": list(datas) * 2, "h": [1] * 10 + [4] * 10, "real": 1.0})
    for m in figuras.NOMES:
        series[m] = 0.9
    series.to_csv(pasta / "teste_final_series.csv", index=False)

    exp = pd.DataFrame(
        [
            {"etapa": "etapa2", "agenda": "warmup_cos", "seed": 0, "otimizador": o, "lr": lr,
             "melhor_val_loss": 0.1 + lr}
            for o in ("sgd", "adam", "adamw") for lr in (0.001, 0.01, 0.1)
        ]
    )
    exp.to_csv(pasta / "experimentos.csv", index=False)


def test_gera_as_tres_figuras(tmp_path):
    _csvs_falsos(tmp_path)
    caminhos = figuras.main(tmp_path)
    assert [p.name for p in caminhos] == [
        "resultado_teste.png",
        "serie_estadual.png",
        "otimizadores_eta.png",
    ]
    for p in caminhos:
        assert p.parent == tmp_path / "figuras"
        assert p.stat().st_size > 1000
