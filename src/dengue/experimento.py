"""Executor de experimentos por YAML (task 18), com o protocolo do desafio 3.

    python -m dengue.experimento configs/base.yaml
    python -m dengue.experimento configs/base.yaml --seeds 0     # só uma seed

Protocolo:
- **orçamento fixo de passos** (lote 32, 920 passos por padrão), sem early stopping, para
  que todos os otimizadores e agendas tenham a mesma quantidade de atualizações;
- a perda na validação é medida a cada época, e no fim valem os pesos da **melhor época**;
- a cada `log_cada` passos o log guarda: perda, ‖g‖ (antes do clip), ‖θ‖, η_t e, por
  camada, ‖g‖, ‖θ‖ e ρ = ‖θ_depois − θ_antes‖ / ‖θ_antes‖ (o tamanho relativo do passo).

Saídas (o teste nunca é usado aqui):
- `results/logs/<id>_s<seed>.csv`: o log por passo;
- `results/logs/<id>_s<seed>_val.csv`: a perda na validação por época;
- `results/experimentos.csv`: uma linha por (experimento, seed), com config e métricas.
Se o log de um (id, seed) já existe, a execução é pulada (dá para retomar uma varredura).
"""

import argparse
import copy
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from dengue import dados, grafo, metricas, modelos, perdas, treino

DIR_RESULTADOS = dados.RAIZ / "results"
OTIMIZADORES = ("sgd", "sgd_nesterov", "adam", "adamw", "muon")
AGENDAS = ("const", "cos", "warmup_cos", "multistep")
MODELOS = ("stgcn", "lstm", "transformer")


@dataclass
class Config:
    id: str
    modelo: str = "stgcn"
    grafo: str = "geo"  # só vale para o stgcn
    atencao: bool = True  # só vale para o transformer: False = cada município só vê a si mesmo
    otimizador: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_adamw_muon: float = 3e-4  # η do AdamW que treina, no Muon, os parâmetros que não são matriz
    agenda: str = "const"
    perda: str = "mse"  # mse | huber | tweedie (task 20)
    huber_delta: float = 1.0
    tweedie_p: float = 1.5
    frac_warmup: float = 0.05
    passos: int = 920
    lote: int = 32
    clip: float = 5.0
    log_cada: int = 20
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    etapa: str = ""

    def __post_init__(self):
        for nome, valor, validos in [
            ("modelo", self.modelo, MODELOS),
            ("grafo", self.grafo, grafo.TIPOS_GRAFO),
            ("otimizador", self.otimizador, OTIMIZADORES),
            ("agenda", self.agenda, AGENDAS),
            ("perda", self.perda, perdas.PERDAS),
        ]:
            if valor not in validos:
                raise ValueError(f"{nome} deve ser um de {validos}, recebido {valor!r}")
        if self.passos <= 0 or self.lote <= 0 or self.lr <= 0:
            raise ValueError("passos, lote e lr precisam ser positivos")


def carregar_config(caminho: str | Path) -> Config:
    """Lê o YAML e valida. Chaves desconhecidas dão erro (pega erro de digitação)."""
    conteudo = yaml.safe_load(Path(caminho).read_text(encoding="utf-8")) or {}
    desconhecidas = set(conteudo) - {f.name for f in fields(Config)}
    if desconhecidas:
        raise ValueError(f"Chaves desconhecidas no YAML: {sorted(desconhecidas)}")
    return Config(**conteudo)


# --- Agendas de η ---


def fator_lr(passo: int, total: int, agenda: str, frac_warmup: float = 0.05) -> float:
    """Fator que multiplica o η base no passo `passo` (0..total). 1.0 = η cheio.

    - const: sempre 1;
    - cos: cai de 1 a 0 em meio cosseno;
    - warmup_cos: sobe linearmente de 0 a 1 nos primeiros `frac_warmup` do treino, depois cos;
    - multistep: 1, depois 0,1 a partir de 50% e 0,01 a partir de 75%.
    """
    if agenda == "const":
        return 1.0
    if agenda == "cos":
        return 0.5 * (1 + math.cos(math.pi * passo / total))
    if agenda == "warmup_cos":
        warmup = int(frac_warmup * total)
        if passo < warmup:
            return passo / warmup
        return 0.5 * (1 + math.cos(math.pi * (passo - warmup) / max(1, total - warmup)))
    if agenda == "multistep":
        if passo >= 0.75 * total:
            return 0.01
        if passo >= 0.5 * total:
            return 0.1
        return 1.0
    raise ValueError(f"agenda desconhecida: {agenda}")


# --- Muon ---


def _como_matriz(t: torch.Tensor) -> torch.Tensor:
    """Vê um tensor de pesos como matriz [saída, resto]. Conv [o, i, 1, k] -> [o, i·k];
    Chebyshev θ [K, i, o] -> [o, K·i]."""
    if t.ndim == 3:
        return t.permute(2, 0, 1).reshape(t.shape[2], -1)
    return t.reshape(t.shape[0], -1)


def _de_matriz(m: torch.Tensor, forma: torch.Size) -> torch.Tensor:
    if len(forma) == 3:
        k, i, o = forma
        return m.reshape(o, k, i).permute(1, 2, 0)
    return m.reshape(forma)


def _ortogonalizar(g: torch.Tensor, passos: int = 5) -> torch.Tensor:
    """Newton–Schulz: aproxima a matriz ortogonal mais próxima de g (todos os valores
    singulares ≈ 1). É o que faz o Muon atualizar em todas as direções, não só na dominante."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + 1e-7)
    transposta = x.shape[0] > x.shape[1]
    if transposta:
        x = x.T
    for _ in range(passos):
        aa = x @ x.T
        x = a * x + (b * aa + c * aa @ aa) @ x
    return x.T if transposta else x


class Muon(torch.optim.Optimizer):
    """Muon (momentum + ortogonalização) para pesos com 2 ou mais dimensões.

    Os pesos são vistos como matriz [saída, resto], o momentum é ortogonalizado e o passo
    é escalado por sqrt(max(1, linhas/colunas)).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for grupo in self.param_groups:
            for p in grupo["params"]:
                if p.grad is None:
                    continue
                estado = self.state[p]
                g = _como_matriz(p.grad)
                if "momentum" not in estado:
                    estado["momentum"] = torch.zeros_like(g)
                buf = estado["momentum"].mul_(grupo["momentum"]).add_(g)
                passo = _ortogonalizar(buf)
                escala = math.sqrt(max(1.0, passo.shape[0] / passo.shape[1]))
                p.mul_(1 - grupo["lr"] * grupo["weight_decay"])
                p.add_(_de_matriz(passo, p.shape), alpha=-grupo["lr"] * escala)


def criar_otimizadores(modelo: nn.Module, cfg: Config) -> list[torch.optim.Optimizer]:
    """Lista de otimizadores (o Muon vem junto com um AdamW para vieses, normas e cabeça)."""
    params = modelo.parameters()
    wd = cfg.weight_decay
    if cfg.otimizador == "sgd":
        return [torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=wd)]
    if cfg.otimizador == "sgd_nesterov":
        return [torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, nesterov=True, weight_decay=wd)]
    if cfg.otimizador == "adam":
        return [torch.optim.Adam(params, lr=cfg.lr, weight_decay=wd)]
    if cfg.otimizador == "adamw":
        return [torch.optim.AdamW(params, lr=cfg.lr, weight_decay=wd or 0.01)]
    # muon: matrizes internas no Muon; o resto (vieses, LayerNorm, cabeça) no AdamW
    matrizes, resto = [], []
    for nome, p in modelo.named_parameters():
        (matrizes if p.ndim >= 2 and not nome.startswith("cabeca") else resto).append(p)
    return [
        Muon(matrizes, lr=cfg.lr, weight_decay=wd),
        torch.optim.AdamW(resto, lr=cfg.lr_adamw_muon, weight_decay=wd or 0.01),
    ]


# --- Treino com orçamento fixo de passos ---


def criar_modelo(cfg: Config, df: pd.DataFrame) -> nn.Module:
    if cfg.modelo == "lstm":
        return modelos.LSTM()
    if cfg.modelo == "transformer":
        return modelos.TransformerEspacial(len(dados.municipios(df)), atencao=cfg.atencao)
    return modelos.STGCN(grafo.laplaciana_do_projeto(df, cfg.grafo))


def _norma(tensores) -> float:
    return float(torch.sqrt(sum((t.detach() ** 2).sum() for t in tensores)))


def _perda(cfg: Config, z: torch.Tensor, y: torch.Tensor, reducao: str = "mean") -> torch.Tensor:
    return perdas.calcular(cfg.perda, z, y, cfg.huber_delta, cfg.tweedie_p, reducao)


@torch.no_grad()
def perda_particao(modelo: nn.Module, p: dados.Particao, cfg: Config, lote: int = 256) -> float:
    """Perda média do experimento (a mesma do treino) numa partição inteira.

    A melhor época é escolhida por ela: escolher pelo MSE favoreceria a perda MSE.
    """
    modelo.eval()
    x, y = treino._tensores(p)
    soma = sum(
        float(_perda(cfg, modelo(x[i : i + lote]), y[i : i + lote], "sum"))
        for i in range(0, len(x), lote)
    )
    return soma / y.numel()


def treinar_por_passos(
    modelo: nn.Module,
    p_treino: dados.Particao,
    p_val: dados.Particao,
    cfg: Config,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Treina por exatamente `cfg.passos` passos. Devolve (log por passo, log por época, info).

    Ao final, `modelo` fica com os pesos da época de menor perda na validação.
    """
    treino.fixar_seed(seed)
    x = torch.from_numpy(np.asarray(p_treino.x, np.float32))
    y = torch.from_numpy(np.asarray(p_treino.y, np.float32))
    otimizadores = criar_otimizadores(modelo, cfg)
    agendas = [
        torch.optim.lr_scheduler.LambdaLR(
            o, lambda t: fator_lr(t, cfg.passos, cfg.agenda, cfg.frac_warmup)
        )
        for o in otimizadores
    ]
    gerador = torch.Generator().manual_seed(seed)
    nomes = [n for n, p in modelo.named_parameters() if p.ndim >= 2]

    log, log_val = [], []
    melhor, melhor_estado, divergiu, clip_hits = float("inf"), None, False, 0
    passo, epoca = 0, 0
    while passo < cfg.passos and not divergiu:
        epoca += 1
        ordem = torch.randperm(len(x), generator=gerador)
        for i in range(0, len(x), cfg.lote):
            if passo >= cfg.passos:
                break
            modelo.train()
            idx = ordem[i : i + cfg.lote]
            for o in otimizadores:
                o.zero_grad()
            perda = _perda(cfg, modelo(x[idx]), y[idx])
            if not torch.isfinite(perda):
                divergiu = True
                break
            perda.backward()

            registrar = (passo + 1) % cfg.log_cada == 0
            params = dict(modelo.named_parameters())
            if registrar:
                antes = {n: params[n].detach().clone() for n in nomes}
                grads = {n: float(params[n].grad.norm()) for n in nomes}
            norma_g = float(nn.utils.clip_grad_norm_(modelo.parameters(), cfg.clip))
            clip_hits += norma_g > cfg.clip
            for o in otimizadores:
                o.step()
            eta = otimizadores[0].param_groups[0]["lr"]  # η_t usado neste passo
            for a in agendas:
                a.step()
            passo += 1

            if registrar:
                linha = {
                    "passo": passo,
                    "epoca": epoca,
                    "perda": perda.item(),
                    "grad": norma_g,
                    "theta": _norma(modelo.parameters()),
                    "lr": eta,
                }
                for n in nomes:
                    p = params[n].detach()
                    linha[f"rho/{n}"] = float((p - antes[n]).norm() / (antes[n].norm() + 1e-12))
                    linha[f"gn/{n}"] = grads[n]
                    linha[f"thn/{n}"] = float(p.norm())
                log.append(linha)

        if divergiu:
            break
        perda_val = perda_particao(modelo, p_val, cfg)
        log_val.append({"epoca": epoca, "passo": passo, "perda_val": perda_val})
        if not math.isfinite(perda_val):
            divergiu = True
        elif perda_val < melhor:
            melhor, melhor_estado = perda_val, copy.deepcopy(modelo.state_dict())

    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)
    info = {
        "melhor_val_loss": melhor,
        "val_loss_final": log_val[-1]["perda_val"] if log_val else float("nan"),
        "divergiu": divergiu,
        "clip_hits": int(clip_hits),
        "n_passos_exec": passo,
    }
    return pd.DataFrame(log), pd.DataFrame(log_val), info


def executar(
    cfg: Config,
    df: pd.DataFrame | None = None,
    seeds=None,
    dir_resultados: str | Path = DIR_RESULTADOS,
) -> pd.DataFrame:
    """Roda o experimento para cada seed e devolve as linhas adicionadas ao `experimentos.csv`."""
    df = dados.carregar_tabela() if df is None else df
    dir_resultados = Path(dir_resultados)
    dir_logs = dir_resultados / "logs"
    dir_logs.mkdir(parents=True, exist_ok=True)
    caminho_exp = dir_resultados / "experimentos.csv"

    particoes, _ = dados.preparar(df)
    p_treino, p_val = particoes["treino"], particoes["validacao"]
    _, y, datas = dados.montar_tensor(df)
    limiar = metricas.limiar_surto(y[dados.indices_particoes(datas)["treino"]])

    linhas = []
    for seed in cfg.seeds if seeds is None else seeds:
        nome = f"{cfg.id}_s{seed}"
        if (dir_logs / f"{nome}.csv").exists():
            print(f"[pula] {nome} já tem log")
            continue
        inicio = time.time()
        treino.fixar_seed(seed)
        modelo = criar_modelo(cfg, df)
        log, log_val, info = treinar_por_passos(modelo, p_treino, p_val, cfg, seed)
        y_hat = treino.prever(modelo, p_val)
        config = {k: v for k, v in asdict(cfg).items() if k != "seeds"}
        linha = {
            **config,
            "seed": seed,
            **info,
            **metricas.resumo(p_val.y_bruto, y_hat, limiar),
            **metricas.por_regime(p_val.y_bruto, y_hat, limiar),
            "tempo_s": round(time.time() - inicio, 1),
        }
        log.to_csv(dir_logs / f"{nome}.csv", index=False)
        log_val.to_csv(dir_logs / f"{nome}_val.csv", index=False)
        novo = pd.DataFrame([linha])
        if caminho_exp.exists():
            novo = pd.concat([pd.read_csv(caminho_exp), novo], ignore_index=True)
        novo.to_csv(caminho_exp, index=False)
        linhas.append(linha)
        print(f"[ok] {nome}: val MAE {linha['mae']:.4f} | val loss {info['melhor_val_loss']:.4f}")
    return pd.DataFrame(linhas)


def resumir(exp: pd.DataFrame, por: str | list[str], colunas: list[str]) -> pd.DataFrame:
    """Tabela média ± desvio entre seeds: uma linha por grupo, colunas `<m>_media`, `<m>_desvio`
    e `n_seeds`. O desvio é o amostral (ddof=1), como nas outras tabelas do projeto."""
    grupos = exp.groupby(por, sort=False)
    media = grupos[colunas].mean().add_suffix("_media")
    desvio = grupos[colunas].std().add_suffix("_desvio")
    tabela = pd.concat([media, desvio], axis=1)
    ordem = [f"{c}_{s}" for c in colunas for s in ("media", "desvio")]
    return tabela[ordem].assign(n_seeds=grupos.size())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Roda um experimento descrito num YAML.")
    parser.add_argument("config", help="caminho do YAML (veja configs/base.yaml)")
    parser.add_argument("--seeds", type=int, nargs="*", help="sobrescreve as seeds do YAML")
    parser.add_argument(
        "--resultados", default=DIR_RESULTADOS, help="pasta de saída (padrão: results/)"
    )
    args = parser.parse_args(argv)
    executar(carregar_config(args.config), seeds=args.seeds, dir_resultados=args.resultados)


if __name__ == "__main__":
    main()
