# STGNN-Dengue

Previsão semanal de internações por dengue nos municípios do Rio de Janeiro com redes neurais espaço-temporais em grafos (STGCN), comparadas a baselines, e um estudo de otimizadores e agendas de taxa de aprendizado.

## Como rodar

Requer Python 3.11 ou mais novo.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                           # testes
ruff check .                     # estilo
```

## Estrutura

```
src/dengue/      código reutilizável (dados, grafo, modelos, treino, métricas)
notebooks/       a narrativa do estudo; só importam de src/
configs/         configurações (YAML) dos experimentos
results/         tabelas e figuras finais
tests/           testes automáticos
docs/decisoes/   registro das decisões do projeto
data/            dados brutos (fora do git)
```
