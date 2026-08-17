# Tech Challenge 2 — Sistema de Recomendação MovieLens

Projeto da Fase 2 da Pós Tech FIAP (Machine Learning Engineering). Implementa um
pipeline de MLOps ponta a ponta para um sistema de recomendação: pré-processamento
versionado com DVC, treino de um modelo de embeddings (PyTorch) comparado a um
baseline (Scikit-Learn), rastreamento de experimentos e Model Registry no MLflow,
containerização em Docker e uma API de serving (FastAPI) publicada na nuvem.

O enunciado original propõe um dataset de e-commerce; usamos o **MovieLens 32M**
como alternativa permitida pelo desafio ("qualquer dataset com ≥ 10.000 interações
user-item"), adaptando o problema para prever a nota que um usuário daria a um filme
e recomendar os filmes com maior nota prevista.

## Sumário

- [Arquitetura](#arquitetura)
- [Stack técnico](#stack-técnico)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Como rodar](#como-rodar)
- [Pipeline (DVC)](#pipeline-dvc)
- [MLflow](#mlflow)
- [Docker](#docker)
- [API de recomendação](#api-de-recomendação)
- [Testes](#testes)
- [Resultados](#resultados)
- [Deploy em nuvem (bônus)](#deploy-em-nuvem-bônus)
- [Limitações conhecidas](#limitações-conhecidas)
- [Equipe](#equipe)

## Arquitetura

```
data/raw/ratings.csv (DVC)
        │
        ▼
   [preprocess]  MovieLensPreprocessor (Strategy) — gera user_idx/movie_idx
        │
        ▼
data/processed/clean.csv (DVC)
        │
        ├──► [train_baseline]     RandomForestRegressor (Scikit-Learn)
        │           │
        └──► [train_neural_net]   MLPEmbedding (PyTorch)
                    │
                    ▼
        MLflow: params + métricas + artefatos de cada run
                    │
                    ▼
        MLflow Model Registry: Staging → Production
                    │
                    ▼
        models/model.pt + mappings + interactions (artefatos p/ API)
                    │
                    ▼
        FastAPI (src/api) ──► Docker ──► Cloud Run (URL pública)
```

Os modelos são instanciados por uma `ModelFactory` (Factory pattern) e o
pré-processamento é encapsulado numa `PreprocessStrategy` (Strategy pattern),
permitindo trocar de modelo/dataset sem alterar o pipeline de treino.

## Stack técnico

| Camada | Ferramenta |
|---|---|
| Modelo neural | PyTorch (`MLPEmbedding`: embeddings de usuário/item + MLP) |
| Baseline | Scikit-Learn (`RandomForestRegressor`) |
| Tracking + Registry | MLflow |
| Versionamento de dados/pipeline | DVC |
| API de serving | FastAPI + Uvicorn |
| Dependências | Poetry |
| Config | Pydantic Settings (`.env`) |
| Containerização | Docker (multi-stage) + Docker Compose |
| Deploy (bônus) | Google Cloud Run |
| Qualidade | Ruff (lint + format), pre-commit, Pytest |

## Estrutura do projeto

```
src/
├── api/           FastAPI: main, predictor, schemas, middleware, logging
├── data/          Estratégias de pré-processamento (Strategy pattern)
├── models/        Factory de modelos + arquitetura da rede neural
├── tracking/      Integração com MLflow (log + promoção no Registry)
├── training/      Pipeline de treino e cálculo de métricas
└── config.py      Configurações centralizadas (Pydantic Settings)
scripts/           Script de validação de ambiente
tests/             Testes automatizados (pytest)
docs/              Model Card (docs/model_card.md)
configs/           Reservado — configuração já é centralizada em .env/config.py
dvc.yaml           Definição do pipeline (3 stages)
Dockerfile         Imagem multi-stage para treino
Dockerfile.api      Imagem multi-stage para servir a API
docker-compose.yml  Orquestra treino + MLflow server
cloudbuild.yaml, .gcloudignore   Build remoto para o deploy no Cloud Run
```

## Como rodar

### Pré-requisitos

- Python 3.11 ou 3.12
- [Poetry](https://python-poetry.org/)
- Docker + Docker Compose (opcional, para rodar containerizado)

### Instalação

```bash
poetry install
cp .env.example .env
poetry run python -m scripts.validate_env
```

### Dataset

O dataset (MovieLens 32M, `ratings.csv`, ~830MB) **não está no repositório** — o
remote do DVC configurado (`localstorage`, ver `.dvc/config`) é uma pasta local, não
um storage compartilhado, então `dvc pull` não funciona em outra máquina. Para
reproduzir:

1. Baixe `ratings.csv` em <https://www.kaggle.com/datasets/justsahil/movielens-32m>
2. Salve em `data/raw/ratings.csv`
3. Verifique a integridade (opcional): MD5 esperado `cf12b74f9ad4b94a011f079e26d4270a`,
   tamanho 877.076.222 bytes
4. Rode `poetry run dvc add data/raw/ratings.csv` se quiser recriar o `.dvc` local

## Pipeline (DVC)

O pipeline tem 3 stages (mínimo exigido: 3), definidos em `dvc.yaml`:

| Stage | O que faz |
|---|---|
| `preprocess` | Codifica `userId`/`movieId` em índices contíguos (`user_idx`/`movie_idx`) |
| `train_baseline` | Treina e avalia o `RandomForestRegressor`, loga no MLflow |
| `train_neural_net` | Treina o `MLPEmbedding` com early stopping, avalia, loga e promove no MLflow Registry |

```bash
poetry run dvc repro     # roda o pipeline completo, respeitando o cache de dependências
poetry run dvc status    # confere o que está desatualizado
```

Também é possível rodar cada modelo isoladamente:

```bash
poetry run python -m src.training.train --model rf_baseline
poetry run python -m src.training.train --model mlp
```

## MLflow

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre em `http://localhost:5000`. Cada run de treino loga parâmetros, as 4 métricas
de avaliação e o artefato do modelo. O modelo MLP é automaticamente registrado no
Model Registry (`movielens-mlp-recommender`) e promovido de Staging para Production
a cada novo treino bem-sucedido.

## Docker

**Treino containerizado + MLflow server:**

```bash
docker compose up --build
```

Sobe o `mlflow` (server na porta 5000, com healthcheck) e, quando ele estiver
saudável, o `train` roda o pipeline de treino do MLP dentro do container, lendo
`data/` via bind mount.

**Build manual da imagem de treino:**

```bash
docker build -t movielens-train .
```

## API de recomendação

A API expõe o modelo MLP promovido a Production para previsão de nota e
recomendação de filmes.

| Endpoint | Método | Descrição |
|---|---|---|
| `/health` | GET | Healthcheck + status de carregamento do modelo |
| `/predict` | POST | Prevê a nota de um usuário para um filme (`user_id`, `movie_id`) |
| `/recommend` | POST | Retorna os `top_n` filmes com maior nota prevista para um usuário, excluindo os já avaliados |
| `/docs` | GET | Swagger UI (documentação interativa, gerada pelo FastAPI) |

**Rodar localmente:**

```bash
poetry run python -m src.training.train --model mlp   # gera models/model.pt e mappings
poetry run uvicorn src.api.main:app --reload
```

**Build da imagem de serving:**

```bash
docker build -f Dockerfile.api -t movielens-api .
```

## Testes

```bash
poetry run pytest tests/ -v
```

Cobre os design patterns (Factory/Strategy), o cálculo das métricas, a arquitetura
da rede neural e a lógica do `Predictor` da API — incluindo um teste que garante que
a geração de recomendações em lote produz exatamente os mesmos resultados que a
previsão individual.

## Resultados

Avaliação no conjunto de teste, 4 métricas (RMSE, MAE, R², MedAE):

| Modelo | RMSE ↓ | MAE ↓ | R² ↑ | MedAE ↓ |
|---|---|---|---|---|
| Baseline (Random Forest) | 1.111 | 0.909 | -0.078 | 0.800 |
| MLP (PyTorch, embeddings) | 2.947 | 2.751 | -6.587 | 2.924 |

O MLP **não superou o baseline** nesta configuração — ver
[Model Card](docs/model_card.md) para a análise detalhada da causa (amostra de
treino pequena frente ao tamanho do catálogo) e o que já foi investigado.

## Deploy em nuvem (bônus)

A API está publicada no Google Cloud Run:

**`https://movielens-api-894098705993.us-central1.run.app`**

Build feito remotamente via Cloud Build (`cloudbuild.yaml`), a partir do
`Dockerfile.api`, com os artefatos do modelo já embutidos na imagem. O serviço
escala a zero quando ocioso (`min-instances=0`), então a primeira requisição após um
período parado pode levar alguns segundos a mais (cold start).

## Limitações conhecidas

- O MLP foi treinado com uma amostra pequena (`SAMPLE_SIZE=1000`) do dataset
  completo por restrição de tempo/recursos de uma disciplina — isso prejudica
  diretamente sua performance frente ao baseline (detalhes no Model Card).
- Não há testes de carga/stress na API além da validação funcional dos endpoints.
- `_load_interactions` na API depende de um artefato pré-computado no treino
  (`interactions.pkl`); se o modelo for retreinado, a API precisa ser reiniciada
  para carregar os novos artefatos.

## Equipe

Adriano Carvalho · Charles Albano · Nickolas Ferraz · Renata Dias Santana
