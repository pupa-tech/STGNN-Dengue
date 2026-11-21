"""Varredura de experimentos (task 19): expande uma grade descrita em YAML e roda cada ponto.

    python -m dengue.varredura configs/otimizacao/etapa2.yaml

Formato do YAML:

    base:            # campos comuns a todos os pontos (os mesmos da `experimento.Config`)
      agenda: warmup_cos
      seeds: [0]
    grade:           # cada item vira o produto cartesiano das listas dele
      - {otimizador: sgd, lr: [0.01, 0.1, 1.0]}
      - {otimizador: [adam, adamw], lr: [0.001, 0.01]}

O `id` de cada ponto é gerado a partir da config (`e19_<otimizador>_lr<η>_<agenda>`), sem a
etapa. Assim, um ponto que já rodou em outra etapa é reaproveitado (o executor pula o
que já tem log), e a etapa seguinte só roda o que falta.
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd
import yaml

from dengue import experimento


def gerar_id(cfg: dict) -> str:
    return f"e19_{cfg['otimizador']}_lr{cfg['lr']:g}_{cfg['agenda']}"


def expandir(especificacao: dict) -> list[experimento.Config]:
    """Lista de configs da grade, na ordem do YAML, sem repetições."""
    base = dict(especificacao.get("base") or {})
    if "id" in base:
        raise ValueError("o id é gerado automaticamente; tire-o da base")
    configs, vistos = [], set()
    for item in especificacao.get("grade") or []:
        listas = {k: v if isinstance(v, list) else [v] for k, v in item.items()}
        for valores in itertools.product(*listas.values()):
            campos = {"agenda": "const", **base, **dict(zip(listas, valores, strict=True))}
            padrao = experimento.Config(id="_")
            campos["id"] = gerar_id({"otimizador": padrao.otimizador, "lr": padrao.lr, **campos})
            if campos["id"] in vistos:
                continue
            vistos.add(campos["id"])
            configs.append(experimento.Config(**campos))
    if not configs:
        raise ValueError("a grade está vazia")
    return configs


def carregar(caminho: str | Path) -> list[experimento.Config]:
    return expandir(yaml.safe_load(Path(caminho).read_text(encoding="utf-8")) or {})


def main(argv=None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Roda uma grade de experimentos.")
    parser.add_argument("config", help="YAML com `base` e `grade`")
    parser.add_argument("--resultados", default=experimento.DIR_RESULTADOS, help="pasta de saída")
    args = parser.parse_args(argv)
    configs = carregar(args.config)
    print(f"{len(configs)} pontos na grade")
    partes = [experimento.executar(c, dir_resultados=args.resultados) for c in configs]
    return pd.concat(partes, ignore_index=True)


if __name__ == "__main__":
    main()
