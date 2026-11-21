"""Testes das funções de perda (task 20)."""

import pandas as pd
import pytest
import torch
from conftest import fazer_bruto

from dengue import dados, experimento, metricas, perdas


def _par(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(16, 4, 5, generator=g), torch.rand(16, 4, 5, generator=g) * 3


def test_mse_igual_ao_de_antes():
    z, y = _par()
    torch.testing.assert_close(perdas.calcular("mse", z, y), torch.nn.functional.mse_loss(z, y))


def test_huber_com_delta_grande_vira_meio_mse():
    z, y = _par()
    obtido = perdas.calcular("huber", z, y, huber_delta=100.0)
    torch.testing.assert_close(obtido, torch.nn.functional.mse_loss(z, y) / 2)


def test_huber_cresce_linear_nos_erros_grandes():
    y = torch.zeros(1)
    l10 = perdas.calcular("huber", torch.tensor([10.0]), y)
    l20 = perdas.calcular("huber", torch.tensor([20.0]), y)
    assert float(l20 - l10) == pytest.approx(10.0)  # inclinação = delta = 1


def test_tweedie_finita_com_taxa_zero():
    z = torch.randn(10, requires_grad=True)
    perda = perdas.calcular("tweedie", z, torch.zeros(10))
    perda.backward()
    assert torch.isfinite(perda) and torch.isfinite(z.grad).all()


def test_tweedie_tem_minimo_na_media():
    # Para vários valores de taxa, o z ótimo (constante) é log(média de taxa + 1).
    taxas = torch.tensor([0.0, 0.0, 1.0, 5.0, 20.0])
    y_log = torch.log1p(taxas)
    z = torch.zeros(1, requires_grad=True)
    otimizador = torch.optim.LBFGS([z], lr=0.5, max_iter=200)

    def passo():
        otimizador.zero_grad()
        perda = perdas.calcular("tweedie", z.expand_as(y_log), y_log)
        perda.backward()
        return perda

    otimizador.step(passo)
    assert float(torch.expm1(z.detach())) == pytest.approx(float(taxas.mean()), rel=1e-3)


def test_reducao_soma_e_media_batem():
    z, y = _par()
    for nome in perdas.PERDAS:
        soma = perdas.calcular(nome, z, y, reducao="sum")
        torch.testing.assert_close(soma / y.numel(), perdas.calcular(nome, z, y))


def test_perda_invalida():
    z, y = _par()
    with pytest.raises(ValueError, match="perda"):
        perdas.calcular("mae", z, y)
    with pytest.raises(ValueError, match="perda"):
        experimento.Config(id="x", perda="mae")
    with pytest.raises(ValueError, match="tweedie_p"):
        perdas.tweedie(z, y, p=2.5)


def test_por_regime_separa_surto_e_calmaria():
    y = torch.zeros(2, 1, 3).numpy()
    y[0] = 10.0  # primeira janela é surto
    y_hat = y + 1.0  # superestima tudo em 1
    r = metricas.por_regime(y, y_hat, limiar=5.0)
    assert r["vies_surto"] == pytest.approx(1.0) and r["vies_calmaria"] == pytest.approx(1.0)
    assert r["mae_calmaria"] == pytest.approx(1.0)


def test_configs_do_repositorio_so_mudam_a_perda():
    cfgs = {
        p: experimento.carregar_config(dados.RAIZ / "configs" / "perdas" / f"{p}.yaml")
        for p in perdas.PERDAS
    }
    assert {p: c.perda for p, c in cfgs.items()} == {p: p for p in perdas.PERDAS}
    comuns = [{k: v for k, v in vars(c).items() if k not in ("id", "perda")} for c in cfgs.values()]
    assert all(c == comuns[0] for c in comuns)
    assert comuns[0]["otimizador"] == "adamw" and comuns[0]["lr"] == 0.01
    assert comuns[0]["agenda"] == "warmup_cos"


@pytest.mark.parametrize("perda", perdas.PERDAS)
def test_executar_com_cada_perda(tmp_path, perda):
    df = dados.limpar(fazer_bruto(n_municipios=8, n_semanas=900))
    cfg = experimento.Config(
        id="teste",
        otimizador="adamw",
        lr=1e-2,
        agenda="warmup_cos",
        perda=perda,
        passos=12,
        lote=64,
        log_cada=4,
        seeds=[0],
    )
    linhas = experimento.executar(cfg, df=df, dir_resultados=tmp_path)
    assert not linhas.loc[0, "divergiu"]
    exp = pd.read_csv(tmp_path / "experimentos.csv")
    assert exp.loc[0, "perda"] == perda
    assert {"mae_calmaria", "vies_calmaria", "vies_surto"} <= set(exp.columns)


def test_resumir_media_e_desvio():
    exp = pd.DataFrame({"perda": ["a", "a", "b", "b"], "mae": [1.0, 3.0, 2.0, 2.0]})
    t = experimento.resumir(exp, "perda", ["mae"])
    assert list(t.index) == ["a", "b"]
    assert t.loc["a", "mae_media"] == 2.0 and t.loc["a", "mae_desvio"] == pytest.approx(2**0.5)
    assert t.loc["b", "mae_desvio"] == 0.0 and t.loc["a", "n_seeds"] == 2
