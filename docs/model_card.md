# Model Card — MovieLens MLP Recommender

## Detalhes do modelo

| | |
|---|---|
| Nome registrado | `movielens-mlp-recommender` (MLflow Model Registry) |
| Arquitetura | `MLPEmbedding` — embeddings de usuário e item + MLP (2 camadas, ReLU) |
| Framework | PyTorch |
| Dimensão do embedding | 32 |
| Otimizador | Adam (lr = 0.001) |
| Critério de parada | Early stopping por paciência (3 épocas sem melhora na validação), máx. 20 épocas |
| Seed | 42 (fixado via `torch.manual_seed`, determinismo verificado em múltiplos runs) |
| Hardware de treino | CPU (PyTorch CPU-only) |
| Versão em Production | v8, promovida via `MlflowClient.transition_model_version_stage` |

## Uso pretendido

**Uso primário**: prever a nota (1–5) que um usuário daria a um filme não avaliado,
e a partir disso gerar recomendações ordenadas por nota prevista (`/recommend` na
API). Construído como exercício acadêmico (Tech Challenge FIAP) para demonstrar um
pipeline de MLOps completo — não foi validado para uso em produção real.

**Fora do escopo**: recomendação para usuários ou filmes que não existem no
conjunto de treino (cold-start explícito — a API retorna 404 nesses casos, não
tenta extrapolar); qualquer decisão automatizada sem supervisão humana; qualquer
domínio fora de recomendação de filmes (o `MovieLensPreprocessor` é específico ao
schema do MovieLens).

## Dados de treino

- **Fonte**: [MovieLens 32M](https://www.kaggle.com/datasets/justsahil/movielens-32m)
  (`ratings.csv`), ~32 milhões de interações usuário-filme.
- **Pré-processamento**: `MovieLensPreprocessor` codifica `userId`/`movieId` em
  índices contíguos (`user_idx`/`movie_idx`) via `LabelEncoder`, sem filtrar linhas.
- **Amostragem**: o treino usa uma amostra de `SAMPLE_SIZE=1000` linhas
  (`df.sample(..., random_state=42)`) do dataset completo, por restrição de
  tempo/recursos computacionais do projeto — **essa é a decisão de design com
  maior impacto na performance do modelo** (ver Limitações).
- **Split**: 80/20 treino/teste (`test_size=0.2`, `random_state=42`).
- As dimensões dos embeddings (`num_users`, `num_items`) são calculadas a partir do
  dataset **completo**, não da amostra — o modelo suporta qualquer usuário/filme do
  catálogo, mesmo que não tenha aparecido na amostra de treino (mas nesse caso os
  embeddings correspondentes nunca recebem gradiente, permanecendo com a
  inicialização aleatória).

## Baseline de comparação

`RandomForestRegressor` (Scikit-Learn, `n_estimators=50`), treinado nas mesmas
features (`user_idx`, `movie_idx`) e no mesmo split.

## Avaliação

4 métricas de regressão, calculadas no conjunto de teste (`src/training/metrics.py`):

| Métrica | Baseline (Random Forest) | MLP (PyTorch) |
|---|---|---|
| RMSE ↓ | 1.111 | 2.947 |
| MAE ↓ | 0.909 | 2.751 |
| R² ↑ | -0.078 | -6.587 |
| MedAE ↓ | 0.800 | 2.924 |

Ambos os modelos têm R² negativo, ou seja, nenhum supera uma previsão trivial
(média constante) nesta configuração de dados — mas o MLP fica **substancialmente
pior que o baseline** em todas as métricas.

Rastreamento completo: 8 runs registrados no MLflow (`sqlite:///mlflow.db`),
métricas e artefatos versionados por run.

## Limitações e causa raiz da performance do MLP

O principal fator identificado é o **tamanho da amostra de treino frente ao
tamanho do catálogo**: 1.000 interações de treino para ~200 mil usuários e ~87 mil
filmes distintos no dataset completo. Isso significa que a esmagadora maioria dos
embeddings de usuário/filme nunca recebe atualização de gradiente durante o
treino — ficam com a inicialização aleatória, gerando previsões próximas de ruído
para a maior parte do catálogo. Esse é um efeito clássico de cold-start/esparsidade
em sistemas de recomendação baseados em embedding, agravado pela amostra reduzida.

**Testado e não resolvido no escopo deste projeto**: aumentar `SAMPLE_SIZE` para
20.000 (20x maior) não trouxe melhora relevante nas métricas, e custou ~107s de
treino adicional — indício de que o problema não se resolve apenas com mais dados
na mesma proporção; precisaria de uma fração muito maior do dataset completo (ou
técnicas de regularização/inicialização específicas para embeddings esparsos) para
o MLP se tornar competitivo. Dado o escopo e prazo da disciplina, essa investigação
foi documentada aqui em vez de aprofundada.

O baseline (Random Forest) não sofre do mesmo problema na mesma intensidade porque
árvores de decisão não dependem de um vetor de parâmetros por usuário/filme — cada
árvore consegue generalizar a partir de poucos exemplos usando os índices
diretamente como splits.

## Vieses

- **Viés de popularidade**: como o modelo aprende um embedding por usuário/filme,
  usuários e filmes com mais interações no treino tendem a ter representações mais
  informativas; usuários/filmes raros ficam sub-representados.
- **Viés da base de usuários do MovieLens**: o dataset reflete o comportamento de
  quem opta por avaliar filmes na plataforma GroupLens/MovieLens, não é uma amostra
  representativa da população geral de espectadores.
- **Amplificação por amostragem**: a amostra de treino reduzida (item anterior)
  agrava esses vieses, já que a already-pequena diversidade de interações do
  dataset é reduzida ainda mais.

## Considerações éticas

Sistemas de recomendação podem reforçar bolhas de filtro (mostrar sempre conteúdo
similar ao já consumido). Este modelo é um artefato educacional, não está exposto a
usuários reais além da API de demonstração pública (Cloud Run), e não processa
nenhum dado pessoal além de IDs anonimizados já anonimizados pelo dataset MovieLens.

## Recomendações para uso futuro

- Treinar com uma fração muito maior (ou o dataset completo) antes de considerar o
  MLP para qualquer uso além de demonstração.
- Investigar inicialização/regularização específica para embeddings esparsos
  (ex.: `weight_decay`, embeddings pré-treinados, negative sampling).
- Adicionar um teste automatizado que compare a performance do MLP contra o
  baseline e alerte caso a distância aumente ainda mais em treinos futuros.
