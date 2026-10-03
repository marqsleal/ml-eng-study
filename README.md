# Revisão aplicada de Data Science com scikit-learn

Este repositório reúne uma revisão prática e assertiva dos principais conceitos de ciência de dados usando a biblioteca [scikit-learn](https://scikit-learn.org/). Cada notebook é um módulo independente: ele apresenta o problema, prepara os dados, treina modelos candidatos, avalia resultados e discute os critérios de escolha.

## API de inferência

O pipeline treinado no notebook 005 está persistido em `src/model.pickle`. A aplicação FastAPI em `src/main.py` carrega o arquivo ao iniciar e disponibiliza inferências HTTP.

> Esta API é um exemplo didático baseado no dataset `breast_cancer` do scikit-learn. Ela não substitui avaliação, diagnóstico ou decisão clínica profissional.

### Executar localmente

Instale as dependências específicas da API:

```bash
pip install -r src/requirements.txt
```

Em seguida, inicie o servidor com Uvicorn:

```bash
uvicorn src.main:app --reload
```

O servidor fica disponível em `http://127.0.0.1:8000`. A documentação interativa está em `http://127.0.0.1:8000/docs`.

### Endpoints

| Método | Rota | Descrição |
| --- | --- | --- |
| `GET` | `/health` | Confirma que a API e o modelo foram carregados. |
| `POST` | `/predict` | Recebe uma observação e retorna a classificação e a probabilidade de malignidade. |

O corpo de `POST /predict` deve ser um objeto JSON cujas chaves sejam exatamente as 31 colunas usadas no treinamento, incluindo `faixa_raio`. Entradas com campos ausentes ou desconhecidos retornam `HTTP 422`.

Exemplo de resposta:

```json
{
  "classe": 0,
  "diagnostico": "benigno",
  "probabilidade_maligno": 0.04
}
```

### Executar com Docker

Na raiz do projeto:

```bash
docker build -t cancer-api src
docker run --rm -p 8000:8000 cancer-api
```

### Testes manuais

Com a API em execução, execute os casos em `scripts/`:

```bash
bash scripts/001__health.sh
bash scripts/002__predict_benign.sh
bash scripts/003__predict_malignant.sh
```

Cada script aceita opcionalmente a URL-base da API como primeiro argumento, por exemplo: `bash scripts/001__health.sh http://localhost:8000`.

O objetivo não é apenas executar estimadores, mas entender **quando usá-los**, **como os dados influenciam seu desempenho** e **quais métricas sustentam uma decisão**.

O conteúdo usa como base o livro *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, de Aurélien Géron (2ª edição), complementando seus conceitos com demonstrações e anotações próprias deste repositório.

## Como estudar os módulos

Em cada notebook, siga a mesma sequência:

1. Defina o tipo de problema e a variável-alvo (quando houver).
2. Faça uma análise exploratória breve e identifique a natureza das features.
3. Separe treino e teste antes de qualquer ajuste aprendido nos dados.
4. Aplique o pré-processamento necessário usando apenas os dados de treino para ajustar transformadores.
5. Compare uma baseline simples com modelos mais flexíveis.
6. Avalie com métricas adequadas ao objetivo do problema e validação cruzada.
7. Interprete erros, limitações e o impacto das features nos resultados.

## Módulos

| Módulo | Notebook | Foco | Documentação complementar |
| --- | --- | --- | --- |
| 000 | [Processamento de dados](notebooks/000__SCIKIT__DATA_PROCESSING.ipynb) | Divisão de dados, imputação, escalonamento, codificação e features | [Guia de processamento](docs/000__DATA_PROCESSING.md) |
| 001 | [Regressão](notebooks/001__SCIKIT__REGRESSION.ipynb) | Predição de valores numéricos contínuos | [Guia de regressão](docs/001__REGRESSION.md) |
| 002 | [Classificação](notebooks/002__SCIKIT__CLASSIFICATION.ipynb) | Predição de classes, probabilidades e decisão | [Guia de classificação](docs/002__CLASSIFICATION.md) |
| 003 | [Clusterização](notebooks/003__SCIKIT__CLUSTERING.ipynb) | Agrupamento não supervisionado e segmentação | [Guia de clusterização](docs/003__CLUSTERING.md) |
| 004 | [Seleção de modelos](notebooks/004__SCIKIT__MODEL_SELECTION.ipynb) | Validação, busca de hiperparâmetros e comparação justa | [Guia de seleção](docs/004__MODEL_SELECTION.md) |
| 005 | [Treinamento ponta a ponta](notebooks/005__SCIKIT__END_TO_END_PIPELINE.ipynb) | Imputação, codificação, escala, busca de hiperparâmetros e persistência do pipeline | — |
| 006 | Docker e API | Imagem, container, Dockerfile e execução da API de classificação | [Guia de Docker](docs/006__DOCKER.md) |

## Paradigmas, modelos e métricas

### Regressão

Use regressão quando o alvo é um número contínuo, como preço, demanda, duração ou temperatura.

| Famílias de modelos | Quando começar por elas | Métricas principais |
| --- | --- | --- |
| `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` | Relações aproximadamente lineares; necessidade de baseline interpretável; muitas features correlacionadas (`Ridge`) ou desejo de seleção esparsa (`Lasso`) | MAE, RMSE, R² |
| `KNeighborsRegressor` | Relações locais e não lineares, com dados bem escalonados e volume moderado | MAE, RMSE |
| `DecisionTreeRegressor` | Regras e interações não lineares; útil como modelo interpretável inicial | MAE, RMSE, R² |
| `RandomForestRegressor`, `ExtraTreesRegressor` | Relações não lineares, interações e mistura de escalas; baseline robusta para dados tabulares | MAE, RMSE, R² |
| `HistGradientBoostingRegressor`, `GradientBoostingRegressor` | Dados tabulares com relações complexas e busca por desempenho | MAE, RMSE, R² |

- **MAE**: erro absoluto médio; leitura direta na unidade do alvo e menor sensibilidade a outliers.
- **RMSE**: penaliza erros grandes; prefira quando grandes desvios são especialmente custosos.
- **R²**: proporção de variação explicada em relação a uma baseline que prevê a média; não use isoladamente.

### Classificação

Use classificação quando o alvo representa categorias, como fraude/não fraude, espécie, diagnóstico ou faixa de risco.

| Famílias de modelos | Quando começar por elas | Métricas principais |
| --- | --- | --- |
| `LogisticRegression` | Baseline probabilística, interpretável e eficiente para relações aproximadamente lineares | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC |
| `KNeighborsClassifier` | Fronteiras locais não lineares; exige escala comparável nas features | Accuracy, F1 |
| `SVC` / `LinearSVC` | Muitas features; `LinearSVC` para grandes conjuntos esparsos, `SVC` com kernel para fronteiras complexas em dados moderados | F1, ROC-AUC |
| `DecisionTreeClassifier` | Regras interpretáveis, relações não lineares e interações | F1, recall, matriz de confusão |
| `RandomForestClassifier`, `ExtraTreesClassifier` | Dados tabulares heterogêneos e relações não lineares | F1, ROC-AUC, PR-AUC |
| `HistGradientBoostingClassifier`, `GradientBoostingClassifier` | Melhor desempenho em tabulares após ajuste cuidadoso | F1, ROC-AUC, PR-AUC |

- **Accuracy**: proporção de acertos; só é confiável quando as classes têm distribuição e custo de erro semelhantes.
- **Precision**: entre os positivos previstos, quantos realmente são positivos; priorize quando falso positivo é caro.
- **Recall**: entre os positivos reais, quantos foram encontrados; priorize quando falso negativo é caro.
- **F1**: equilíbrio entre precision e recall.
- **ROC-AUC**: capacidade de ordenar as classes em diversos limiares; pode parecer otimista com classes muito desbalanceadas.
- **PR-AUC**: mais informativa para classe positiva rara. Sempre complemente com matriz de confusão e escolha explícita do limiar.

### Clusterização

Use clusterização quando não há alvo e o objetivo é descobrir segmentos, padrões ou grupos semelhantes. Resultados não são "verdades": devem ser validados pela métrica, pela estabilidade e pelo contexto de negócio.

| Modelos | Melhor adequação | Avaliação |
| --- | --- | --- |
| `KNeighborsClassifier` (KNN) | Referência baseada em vizinhança para comparação quando existirem rótulos; não é um algoritmo de clusterização | Accuracy, F1 e matriz de confusão |
| `KMeans` | Grupos aproximadamente esféricos, compactos e de tamanho semelhante; exige número de clusters e escala comparável | Cotovelo (inércia) e silhouette |
| `DBSCAN` | Grupos de formas arbitrárias, ruído e ausência de número de clusters pré-definido; sensível a `eps` e `min_samples` | Proporção de ruído, silhouette dos pontos agrupados e inspeção visual |

- **Método do cotovelo (inércia)**: execute o `KMeans` para vários valores de `k` e observe a soma das distâncias quadradas entre os pontos e seus centróides. Como a inércia sempre diminui quando `k` aumenta, escolha o ponto em que a redução passa a ser marginal — o “cotovelo”. É uma análise visual e não fornece uma regra automática; use-a para reduzir os candidatos a `k` plausíveis.
- **Silhouette**: mede, para cada observação, a proximidade com o próprio grupo em comparação ao grupo vizinho mais próximo; seu valor varia aproximadamente de -1 a 1. Quanto maior a média, mais coesos e separados estão os clusters. Diferentemente da inércia, ela permite comparar diretamente valores candidatos de `k`; valores negativos sugerem alocações inadequadas.

Em resumo: o **cotovelo** mostra o ganho marginal de adicionar clusters e é específico do `KMeans`; a **silhouette** mede qualidade relativa da separação e pode avaliar tanto `KMeans` quanto `DBSCAN` (desconsiderando os pontos de ruído, rotulados como `-1`). A escolha final deve considerar as duas análises, o tamanho útil dos grupos e a interpretação no domínio.

## Features e pré-processamento

| Tipo de feature/situação | Preparação recomendada | Modelos mais sensíveis |
| --- | --- | --- |
| Numérica com escalas distintas | `StandardScaler` ou `MinMaxScaler` | KNN, SVM, K-Means, regressões regularizadas e modelos baseados em distância |
| Numérica com assimetria forte | `log1p`, `PowerTransformer` ou `QuantileTransformer`, após avaliar domínio e zeros | Lineares, KNN, SVM e K-Means; árvores normalmente toleram melhor |
| Categórica nominal | `OneHotEncoder(handle_unknown="ignore")` | Modelos lineares, KNN e SVM; árvores também podem usar, com atenção à alta cardinalidade |
| Categórica ordinal | Codificação ordinal somente se a ordem for real; caso contrário, one-hot | Modelos lineares e de distância são especialmente sensíveis a uma ordem artificial |
| Texto | `TfidfVectorizer`/representação esparsa | `LogisticRegression`, `LinearSVC`, `MultinomialNB` |
| Alta cardinalidade | Agrupar categorias raras, avaliar frequência/target encoding com validação rigorosa | One-hot pode expandir muito o espaço; modelos lineares e árvores podem ser afetados de maneiras distintas |
| Outliers | Investigar a causa; transformações, tratamento justificado ou modelos mais tolerantes | KNN, SVM, K-Means e regressão linear são sensíveis; árvores e ensembles de árvores costumam ser mais tolerantes |
| Dados ausentes | `SimpleImputer` ou `KNNImputer`, ajustados no treino antes de transformar treino e teste | Todos; a imputação deve ser aprendida apenas no treino |

Em geral, modelos lineares são bons para relações lineares e interpretação; árvores e ensembles capturam não linearidades e interações sem exigir escala; métodos baseados em distância e margem dependem fortemente de escala e representação; e modelos de *boosting* frequentemente se destacam em dados tabulares, mas pedem validação e ajuste cuidadosos.

## Seleção e comparação justa

- Use `train_test_split(..., stratify=y)` em classificação, quando aplicável; reserve o teste final para a avaliação definitiva.
- Aplique `cross_val_score`/`cross_validate` ou `GridSearchCV`/`RandomizedSearchCV` no conjunto de treino.
- Para classificação desbalanceada, avalie `class_weight="balanced"`, reamostragem **dentro** do fluxo de validação e métricas além de accuracy.
- Defina uma baseline explícita: `DummyRegressor` ou `DummyClassifier` ajuda a confirmar que o modelo aprende algo relevante.
- Compare modelos sob as mesmas divisões, métricas e pré-processamento. Reporte média e dispersão das dobras de validação.
- Ajuste hiperparâmetros após definir a métrica-alvo; para uma estimativa menos enviesada após muitas tentativas, use validação cruzada aninhada quando o custo permitir.
- Priorize o modelo mais simples que atenda ao objetivo de negócio, incluindo latência, interpretabilidade e estabilidade, não apenas a maior métrica.

## Baseline e interpretação do modelo

Uma **baseline** é a referência mínima que um modelo precisa superar para ser útil. Ela estabelece se a complexidade do modelo está realmente adicionando valor.

- Em regressão, compare com `DummyRegressor(strategy="mean")`, que sempre prevê a média do alvo. Um modelo que não supera esse resultado não está capturando sinal útil.
- Em classificação, use `DummyClassifier(strategy="most_frequent")` como referência de prever sempre a classe mais comum. Em dados desbalanceados, ela pode alcançar alta accuracy e ainda assim ser inútil para a classe de interesse; por isso avalie também precision, recall e F1.
- Uma baseline de modelo simples, como regressão linear ou logística, também é valiosa: ela permite quantificar se modelos mais complexos justificam seu custo e menor interpretabilidade.

**Feature importance** descreve quanto uma feature contribuiu para o modelo, mas o significado depende do método:

- Em modelos lineares, os coeficientes indicam direção e intensidade da relação. Compare magnitudes somente após padronizar as variáveis numéricas; correlação entre features pode tornar essa leitura instável.
- Em árvores e ensembles, `feature_importances_` mede a redução acumulada de impureza. É rápida, mas pode favorecer variáveis contínuas ou com muitas categorias e não informa se o efeito é positivo ou negativo.
- A importância por permutação (`permutation_importance`) mede a queda da métrica quando os valores de uma feature são embaralhados no conjunto de validação/teste. É mais agnóstica ao modelo e ligada à métrica escolhida, mas features muito correlacionadas podem aparentar baixa importância individual.

**SHAP** (*SHapley Additive exPlanations*) é uma técnica de explicabilidade baseada em valores de Shapley. Ela decompõe uma predição individual em contribuições de cada feature, mostrando quanto cada uma deslocou a previsão em relação ao valor-base do modelo. Ao agregar os valores absolutos de SHAP, obtém-se uma visão global de importância; ao inspecionar uma observação, obtém-se uma explicação local.

SHAP é complementar, não uma prova de causalidade: suas explicações descrevem o comportamento do modelo e podem ser afetadas por features correlacionadas, qualidade dos dados e escolhas de pré-processamento. Para esta revisão, use SHAP depois de validar o modelo e interprete-o junto com métricas, gráficos de distribuição e conhecimento de domínio.

## Estrutura do projeto

```text
notebooks/  # demonstrações executáveis por paradigma
docs/       # explicações, referências e decisões complementares de cada módulo
```

As referências conceituais, detalhes de implementação e anotações adicionais de cada assunto devem ser mantidos em `docs/`; os notebooks devem privilegiar demonstrações reproduzíveis e a discussão dos resultados observados.
