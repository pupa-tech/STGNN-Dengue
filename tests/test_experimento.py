"""Testes do executor de experimentos (task 18)."""

import math

import pandas as pd
import pytest
import torch
from conftest import fazer_bruto

from dengue import dados, experimento


def test_carregar_config_do_repositorio():
    cfg = experimento.carregar_config(dados.RAIZ / "configs" / "base.yaml")
    assert cfg.otimizador == "adam" and cfg.agenda == "const"
    assert cfg.passos == 920 and cfg.seeds == [0, 1, 2]


def test_config_rejeita_chave_desconhecida(tmp_path):
    caminho = tmp_path / "x.yaml"
    caminho.write_text("id: x\notimizadr: adam\n", encoding="utf-8")
    with pytest.raises(ValueError, match="desconhecidas"):
        experimento.carregar_config(caminho)


def test_config_rejeita_valor_invalido():
    with pytest.raises(ValueError, match="otimizador"):
        experimento.Config(id="x", otimizador="rmsprop")


@pytest.mark.parametrize(
    ("agenda", "esperado"),
    [
        ("const", [1.0, 1.0, 1.0, 1.0]),
        ("cos", [1.0, 0.5, 0.0, 0.5 * (1 + math.cos(math.pi * 0.25))]),
        ("multistep", [1.0, 0.1, 0.01, 1.0]),
    ],
)
def test_fator_lr(agenda, esperado):
    passos = [0, 50, 100, 25]
    obtido = [experimento.fator_lr(p, 100, agenda) for p in passos]
    assert obtido == pytest.approx(esperado)


def test_warmup_cos_sobe_e_depois_cai():
    fatores = [experimento.fator_lr(p, 100, "warmup_cos", 0.1) for p in range(101)]
    assert fatores[0] == 0.0
    assert fatores[10] == pytest.approx(1.0)  # fim do warmup: η cheio
    assert fatores[5] == pytest.approx(0.5)
    assert fatores[100] == pytest.approx(0.0)


def test_muon_ortogonaliza():
    g = torch.randn(8, 5)
    s = torch.linalg.svdvals(experimento._ortogonalizar(g))
    assert torch.all((s > 0.5) & (s < 1.5))  # Newton–Schulz deixa os valores singulares ≈ 1


@pytest.mark.parametrize("forma", [(4, 3, 1, 3), (3, 5, 7), (6, 2)])
def test_muon_ida_e_volta_da_matriz(forma):
    t = torch.randn(forma)
    m = experimento._como_matriz(t)
    assert m.ndim == 2
    torch.testing.assert_close(experimento._de_matriz(m, t.shape), t)


@pytest.mark.parametrize("otimizador", experimento.OTIMIZADORES)
def test_executar_ponta_a_ponta(tmp_path, otimizador):
    df = dados.limpar(fazer_bruto(n_municipios=8, n_semanas=900))
    cfg = experimento.Config(
        id="teste", otimizador=otimizador, agenda="warmup_cos", passos=12, lote=64,
        log_cada=4, seeds=[0],
    )
    linhas = experimento.executar(cfg, df=df, dir_resultados=tmp_path)
    assert len(linhas) == 1 and linhas.loc[0, "n_passos_exec"] == 12

    log = pd.read_csv(tmp_path / "logs" / "teste_s0.csv")
    assert list(log["passo"]) == [4, 8, 12]
    assert {"perda", "grad", "theta", "lr"} <= set(log.columns)
    assert any(c.startswith("rho/") for c in log.columns)
    assert (tmp_path / "logs" / "teste_s0_val.csv").exists()

    exp = pd.read_csv(tmp_path / "experimentos.csv")
    assert {"otimizador", "seed", "mae", "melhor_val_loss"} <= set(exp.columns)

    # rodar de novo pula o que já existe (dá para retomar)
    assert experimento.executar(cfg, df=df, dir_resultados=tmp_path).empty
