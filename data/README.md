# Dados

Os dados brutos ficam nesta pasta, **fora do git** (só este README é versionado).

## Fonte

Dataset *Brazil-Dengue-Arboviral-Diseases*, publicado no **Zenodo**. Ele traz séries semanais da taxa de internações por dengue por município brasileiro, já integradas a variáveis climáticas, demográficas e socioeconômicas. O projeto usa só o estado do Rio de Janeiro (código IBGE 33).

## Arquivo esperado

```
data/dengue_33_rio_de_janeiro.csv
```

Baixe o dataset do Zenodo e copie para esta pasta o arquivo `dengue_33_rio_de_janeiro.csv` (fica na pasta `data/` do dataset). O código o carrega com `dengue.dados.carregar_csv()`.

## Formato

- **79 municípios × 1.199 semanas** (1999-01-01 a 2021-12-17), ou seja, 94.721 linhas, sem NAs nem semanas faltando.
- Uma linha por município e semana.

| Coluna | Descrição |
| --- | --- |
| `CD_MUN` | código IBGE do município (7 dígitos) |
| `latitude`, `longitude` | coordenadas do município (fixas) |
| `Data` | início da semana (intervalo de 7 dias) |
| `Taxa de Internações por Dengue` | **alvo**: taxa semanal de internações |
| `Índice de Densidade Demográfica` | interpolado, varia no tempo |
| `Emissão de CH4`, `Emissão de CO2`, `Emissão de NO2` | emissões de gases |
| `Índice de Pobreza`, `Índice de Urbanização` | interpolados, variam no tempo |
| `Temperatura Máxima` | clima |
| `Precipitação Média Mensal` | clima |
| `Menor Umidade Relativa` | clima |
| `UF` | código do estado (sempre 33) |
