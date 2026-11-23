"""Para quem o Transformer olha? Mapa de atenção na validação, comparado com o grafo geo (task 23).

    python -m dengue.atencao

Treina o Transformer congelado em `configs/transformer.yaml` (seed 0, só treino e validação),
calcula a atenção média nas janelas da validação e salva
`results/transformer/atencao_validacao.csv`: uma matriz 79 × 79 em que a linha i diz quanto o
município i olhou, em média, para cada município j (cada linha soma 1). O teste não é usado.
"""

import numpy as np
import pandas as pd
import torch

from dengue import dados, experimento, grafo, modelos

CAMINHO_MAPA = dados.RAIZ / "results" / "transformer" / "atencao_validacao.csv"
CONFIG_TRANSFORMER = dados.RAIZ / "configs" / "transformer.yaml"


def mapa_medio(modelo: modelos.TransformerEspacial, p: dados.Particao, lote: int = 256):
    """Atenção [N, N] média sobre as janelas de `p` e sobre as camadas."""
    x = torch.as_tensor(p.x, dtype=torch.float32)
    soma, total = 0.0, 0
    for i in range(0, len(x), lote):
        mapas = modelo.mapas_atencao(x[i : i + lote])  # [camadas, B, N, N]
        soma = soma + mapas.mean(0).sum(0)
        total += mapas.shape[1]
    return (soma / total).numpy()


def comparar_com_grafo(mapa: np.ndarray, dist: np.ndarray, k: int = 5) -> dict:
    """Resume o quanto a atenção aprendida concorda com o grafo k-NN geográfico.

    - `atencao_propria`: peso médio que cada município dá a si mesmo;
    - `fracao_vizinhos`: da atenção que vai para os **outros** municípios, que fração cai
      nos vizinhos do k-NN;
    - `fracao_esperada`: a mesma fração se a atenção fosse uniforme entre os outros (o
      "acaso"); `fracao_vizinhos` muito acima dela quer dizer que o modelo redescobriu a
      vizinhança geográfica;
    - `corr_distancia`: correlação de Spearman entre atenção e distância, fora da
      diagonal (negativa = olha mais para quem está perto).
    """
    mapa = np.asarray(mapa, dtype=float)
    n = mapa.shape[0]
    aresta = grafo.grafo_knn(dist, k=k) > 0
    fora = ~np.eye(n, dtype=bool)
    para_fora = (mapa * fora).sum(1)
    fracao = (mapa * aresta).sum(1) / np.where(para_fora > 0, para_fora, np.nan)
    # Spearman = Pearson dos postos (sem depender do scipy)
    corr = pd.Series(mapa[fora]).rank().corr(pd.Series(np.asarray(dist)[fora]).rank())
    return {
        "atencao_propria": float(np.diag(mapa).mean()),
        "fracao_vizinhos": float(np.nanmean(fracao)),
        "fracao_esperada": float((aresta.sum(1) / (n - 1)).mean()),
        "corr_distancia": float(corr),
    }


def distancias_do_projeto(df: pd.DataFrame) -> np.ndarray:
    """Distâncias haversine [N, N] na ordem de `dados.municipios`."""
    coord = df.groupby("municipio")[["latitude", "longitude"]].first()
    coord = coord.loc[dados.municipios(df)]
    return grafo.distancias_haversine(coord["latitude"], coord["longitude"])


def main(caminho=CAMINHO_MAPA, seed: int = 0) -> pd.DataFrame:
    from dengue import treino

    df = dados.carregar_tabela()
    cfg = experimento.carregar_config(CONFIG_TRANSFORMER)
    particoes, _ = dados.preparar(df)
    treino.fixar_seed(seed)
    modelo = experimento.criar_modelo(cfg, df)
    experimento.treinar_por_passos(modelo, particoes["treino"], particoes["validacao"], cfg, seed)
    codigos = dados.municipios(df)
    mapa = pd.DataFrame(
        mapa_medio(modelo, particoes["validacao"]), index=codigos, columns=codigos
    )
    mapa.index.name = "municipio"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    mapa.round(6).to_csv(caminho)
    return mapa


if __name__ == "__main__":
    m = main()
    print(comparar_com_grafo(m.to_numpy(), distancias_do_projeto(dados.carregar_tabela())))
