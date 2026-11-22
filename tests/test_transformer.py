"""Testes do Transformer espacial (task 23)."""

import torch
from test_lstm import particao_falsa

from dengue import dados, modelos, treino

N = 6


def entrada(b=2, n=N, seed=0):
    gerador = torch.Generator().manual_seed(seed)
    return torch.randn(b, dados.W, n, len(dados.ENTRADAS), generator=gerador)


def _modelo(**kw):
    torch.manual_seed(0)
    return modelos.TransformerEspacial(N, d=8, cabecas=2, **kw).eval()


def _efeito_de_mudar_no_zero(modelo):
    """Quanto a previsão dos outros municípios muda quando só o município 0 muda."""
    x = entrada(b=1)
    x2 = x.clone()
    x2[:, :, 0] += 5.0
    with torch.no_grad():
        return (modelo(x2) - modelo(x))[:, :, 1:].abs().amax(dim=(0, 1))


def test_formato_da_saida():
    assert _modelo()(entrada()).shape == (2, dados.H, N)


def test_sem_atencao_municipios_sao_independentes():
    assert torch.all(_efeito_de_mudar_no_zero(_modelo(atencao=False)) == 0)


def test_com_atencao_todos_os_municipios_se_influenciam():
    assert torch.all(_efeito_de_mudar_no_zero(_modelo()) > 0)


def test_mapas_de_atencao():
    modelo = _modelo()
    mapas = modelo.mapas_atencao(entrada(b=3))
    assert mapas.shape == (2, 3, N, N)  # [camadas, B, N, N]
    torch.testing.assert_close(mapas.sum(-1), torch.ones(2, 3, N))  # cada linha soma 1
    sem = _modelo(atencao=False).mapas_atencao(entrada(b=3))
    torch.testing.assert_close(sem, torch.eye(N).expand(2, 3, N, N))  # só olha para si


def test_mapas_nao_mudam_o_modo_de_treino():
    modelo = _modelo().train()
    modelo.mapas_atencao(entrada())
    assert modelo.training


def test_treino_reduz_a_perda():
    p = particao_falsa(n=N)
    treino.fixar_seed(0)
    modelo = modelos.TransformerEspacial(N, d=16, cabecas=2, dropout=0.0)
    antes = treino.perda(modelo, p)
    treino.treinar(modelo, p, p, treino.ConfigTreino(max_epocas=60, lr=1e-2, lote=8))
    assert treino.perda(modelo, p) < 0.3 * antes
