# 001: Organização do repositório

- **Data:** 2026-11-21
- **Status:** aceita
- **Responsável:** G1
- **Issue/PR:** task 4

## Contexto

A primeira versão do projeto tinha seis notebooks longos, com lógica, experimentos e conclusões misturados. Era difícil reproduzir um resultado ou explicar um trecho de código isolado.

## Opções consideradas

1. **Só notebooks:** rápido para explorar, mas o código se repete entre notebooks e não pode ser testado.
2. **Só scripts:** reproduzível e testável, mas perde a narrativa que a banca e o portfólio pedem.
3. **Pacote em `src/` + notebooks finos:** o código fica em módulos testados; os notebooks só importam e contam a história.

## Decisão

Opção 3: pacote `dengue` em `src/dengue/` com testes em `tests/`, e notebooks numerados em `notebooks/` que apenas importam do pacote.

## Motivo

O código testado no CI garante que o que aparece no notebook é o mesmo que foi verificado. O layout `src/` evita importar por acidente a pasta local no lugar do pacote instalado.

## Consequências

- Toda lógica nova entra primeiro em `src/` com teste; o notebook vem depois.
- Configurações de experimentos ficam em `configs/` (YAML), e resultados finais pequenos em `results/`.
- Dados brutos nunca entram no git (`data/` está no `.gitignore`).
