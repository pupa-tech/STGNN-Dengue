"""Testes do STGCN (task 16)."""

import numpy as np
import torch
from conftest import fazer_bruto
from test_lstm import particao_falsa

from dengue import dados, grafo, modelos, stgcn_validacao, treino

N = 6


def lap_em_linha(n=N):
    """Grafo em linha 0-1-2-...: cada nó só é vizinho do anterior e do seguinte."""
    adj = np.zeros((n, n))
    idx = np.arange(n - 1)
    adj[idx, idx + 1] = adj[idx + 1, idx] = 1.0
    return grafo.laplaciana_escalada(adj)


def entrada(b=2, n=N, seed=0):
    gerador = torch.Generator().manual_seed(seed)
    return torch.randn(b, dados.W, n, len(dados.ENTRADAS), generator=gerador)


def test_formato_da_saida():
    modelo = modelos.STGCN(lap_em_linha(), oculto=8)
    assert modelo(entrada()).shape == (2, dados.H, N)


def test_polinomios_de_chebyshev():
    lap = torch.as_tensor(lap_em_linha(), dtype=torch.float32)
    conv = modelos.ConvChebyshev(1, 1, lap.numpy(), K=3)
    torch.testing.assert_close(conv.T[0], torch.eye(N))
    torch.testing.assert_close(conv.T[1], lap)
    torch.testing.assert_close(conv.T[2], 2 * lap @ lap - torch.eye(N))


def test_conv_temporal_e_causal():
    # mudar a última semana não pode mudar as saídas das semanas anteriores
    torch.manual_seed(0)
    conv = modelos.ConvTemporal(3, 4, kt=3)
    x = torch.randn(1, 3, N, dados.W)
    x2 = x.clone()
    x2[..., -1] += 10.0
    torch.testing.assert_close(conv(x)[..., :-1], conv(x2)[..., :-1])


def _efeito_de_mudar_no_zero(lap):
    """Quanto a previsão dos outros nós muda quando só o nó 0 muda."""
    torch.manual_seed(0)
    modelo = modelos.STGCN(lap, oculto=8).eval()
    x = entrada(b=1)
    x2 = x.clone()
    x2[:, :, 0] += 5.0
    with torch.no_grad():
        return (modelo(x2) - modelo(x))[:, :, 1:].abs().amax(dim=(0, 1))  # por nó 1..N-1


def test_sem_grafo_municipios_sao_independentes():
    efeito = _efeito_de_mudar_no_zero(np.eye(N))
    assert torch.all(efeito == 0)


def test_com_grafo_so_vizinhos_ate_dois_saltos_sao_afetados():
    efeito = _efeito_de_mudar_no_zero(lap_em_linha())  # K=3: até 2 saltos
    assert efeito[0] > 0 and efeito[1] > 0  # nós 1 e 2
    assert torch.all(efeito[2:] == 0)  # nós 3, 4, 5 estão a 3 ou mais saltos


def test_treino_reduz_a_perda():
    p = particao_falsa(n=N)
    treino.fixar_seed(0)
    modelo = modelos.STGCN(lap_em_linha(), oculto=16, dropout=0.0)
    antes = treino.perda(modelo, p)
    treino.treinar(modelo, p, p, treino.ConfigTreino(max_epocas=60, lr=1e-2, lote=8))
    assert treino.perda(modelo, p) < 0.3 * antes


def test_laplaciana_do_projeto():
    df = dados.limpar(fazer_bruto(n_municipios=8, n_semanas=5))
    np.testing.assert_array_equal(grafo.laplaciana_do_projeto(df, "identidade"), np.eye(8))
    lap = grafo.laplaciana_do_projeto(df, "geo", k=2)
    assert lap.shape == (8, 8)
    np.testing.assert_allclose(lap, lap.T, atol=1e-12)


def test_avaliar_ponta_a_ponta():
    # dados sintéticos cobrindo treino e validação (1999 a 2015)
    df = dados.limpar(fazer_bruto(n_municipios=6, n_semanas=900))
    cfg = treino.ConfigTreino(max_epocas=2, paciencia=1, lote=64)
    r = stgcn_validacao.avaliar(df, "geo", seeds=(0,), config=cfg, oculto=4)
    assert len(r) == 1
    assert {"mae", "rmse", "mae_surto", "mae_h4"} <= set(r.columns)
