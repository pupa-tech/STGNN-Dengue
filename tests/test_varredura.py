"""Testes da varredura de experimentos (task 19)."""

import pytest

from dengue import dados, varredura


def test_expandir_produto_cartesiano():
    spec = {
        "base": {"agenda": "cos", "seeds": [0]},
        "grade": [
            {"otimizador": ["sgd", "adam"], "lr": [0.1, 1.0]},
            {"otimizador": "muon", "lr": 0.01, "agenda": ["const", "multistep"]},
        ],
    }
    cfgs = varredura.expandir(spec)
    assert [(c.otimizador, c.lr, c.agenda) for c in cfgs] == [
        ("sgd", 0.1, "cos"),
        ("sgd", 1.0, "cos"),
        ("adam", 0.1, "cos"),
        ("adam", 1.0, "cos"),
        ("muon", 0.01, "const"),
        ("muon", 0.01, "multistep"),
    ]
    assert all(c.seeds == [0] for c in cfgs)
    assert cfgs[0].id == "e19_sgd_lr0.1_cos"


def test_expandir_tira_repetidos_e_id_nao_depende_da_etapa():
    a = varredura.expandir({"base": {"etapa": "etapa2"}, "grade": [{"lr": [0.01, 0.01]}]})
    b = varredura.expandir({"base": {"etapa": "etapa3"}, "grade": [{"lr": 0.01}]})
    assert len(a) == 1 and a[0].id == b[0].id  # o mesmo ponto é reaproveitado entre etapas


def test_expandir_rejeita_erros():
    with pytest.raises(ValueError, match="vazia"):
        varredura.expandir({"base": {}})
    with pytest.raises(ValueError, match="id"):
        varredura.expandir({"base": {"id": "x"}, "grade": [{"lr": 0.1}]})
    with pytest.raises(ValueError, match="otimizador"):
        varredura.expandir({"grade": [{"otimizador": "rmsprop"}]})
    with pytest.raises(TypeError):
        varredura.expandir({"grade": [{"taxa": 0.1}]})  # chave desconhecida


def test_configs_do_repositorio_carregam():
    for caminho in sorted((dados.RAIZ / "configs" / "otimizacao").glob("*.yaml")):
        cfgs = varredura.carregar(caminho)
        assert cfgs and all(c.modelo == "stgcn" and c.passos == 920 for c in cfgs)
