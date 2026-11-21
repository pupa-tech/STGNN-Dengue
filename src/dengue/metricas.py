"""Métricas comuns a todos os modelos.

Todas as métricas são calculadas na **escala original** da taxa de internações.
Se o modelo prevê em `log1p`, desfaça com `np.expm1` antes de chamar estas funções.

Formato padrão das previsões: `[S, H, N]`
    S = número de amostras (janelas), H = horizonte (semanas à frente), N = municípios.
"""

import numpy as np


def _como_array(y, y_hat) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError(f"Formatos diferentes: y {y.shape} e y_hat {y_hat.shape}")
    return y, y_hat


def mae(y, y_hat) -> float:
    """Erro absoluto médio: em média, quanto a previsão erra (na unidade do alvo)."""
    y, y_hat = _como_array(y, y_hat)
    return float(np.mean(np.abs(y - y_hat)))


def rmse(y, y_hat) -> float:
    """Raiz do erro quadrático médio: pune mais os erros grandes (como os dos surtos)."""
    y, y_hat = _como_array(y, y_hat)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def por_horizonte(y, y_hat) -> dict[str, np.ndarray]:
    """MAE e RMSE separados por horizonte, para entradas `[S, H, N]`.

    Devolve `{"mae": [H], "rmse": [H]}`. A posição 0 é h=1 (uma semana à frente).
    """
    y, y_hat = _como_array(y, y_hat)
    if y.ndim != 3:
        raise ValueError(f"Esperado formato [S, H, N], recebido {y.shape}")
    erro = y - y_hat
    return {
        "mae": np.mean(np.abs(erro), axis=(0, 2)),
        "rmse": np.sqrt(np.mean(erro**2, axis=(0, 2))),
    }


def limiar_surto(y_treino, q: float = 0.9) -> float:
    """Limiar de surto: percentil `q` da média estadual semanal **no treino**.

    `y_treino` tem formato `[T, N]` (semanas × municípios). Só passe dados de treino:
    usar validação ou teste aqui seria vazamento.
    """
    y_treino = np.asarray(y_treino, dtype=float)
    if y_treino.ndim != 2:
        raise ValueError(f"Esperado formato [T, N], recebido {y_treino.shape}")
    media_estadual = y_treino.mean(axis=1)
    return float(np.quantile(media_estadual, q))


def mascara_surto(y, limiar: float) -> np.ndarray:
    """Marca as semanas-alvo de surto em `y` `[S, H, N]`. Devolve booleano `[S, H]`.

    Uma semana é de surto quando a média estadual **real** passa do limiar.
    """
    y = np.asarray(y, dtype=float)
    return y.mean(axis=-1) > limiar


def mae_surto(y, y_hat, limiar: float) -> float:
    """MAE só nas semanas de surto. Devolve `nan` se não houver nenhuma."""
    y, y_hat = _como_array(y, y_hat)
    mascara = mascara_surto(y, limiar)
    if not mascara.any():
        return float("nan")
    return float(np.mean(np.abs(y[mascara] - y_hat[mascara])))


def resumo(y, y_hat, limiar: float) -> dict[str, float]:
    """Todas as métricas de uma vez, no formato usado nas tabelas de resultado."""
    resultado = {
        "mae": mae(y, y_hat),
        "rmse": rmse(y, y_hat),
        "mae_surto": mae_surto(y, y_hat, limiar),
    }
    horizontes = por_horizonte(y, y_hat)
    for h, (m, r) in enumerate(zip(horizontes["mae"], horizontes["rmse"], strict=True), start=1):
        resultado[f"mae_h{h}"] = float(m)
        resultado[f"rmse_h{h}"] = float(r)
    return resultado
