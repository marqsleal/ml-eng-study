# Seleção de modelos com scikit-learn

Este módulo transforma os exemplos isolados dos módulos anteriores em um fluxo de seleção reproduzível. O objetivo é escolher representação, algoritmo e hiperparâmetros sem deixar que o conjunto de teste influencie as decisões. A referência principal é o capítulo 2 de *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Aurélien Géron, 2ª edição), especialmente validação cruzada, ajuste e teste final.

## Estimators, transformers e pipelines

Um **estimator** é um objeto com `fit()`: ele aprende algo dos dados. Um **transformer** também implementa `transform()`, como `SimpleImputer`, `StandardScaler`, `PCA` e `OneHotEncoder`. Um preditor, como `Ridge` ou `LogisticRegression`, normalmente expõe `predict()` após o ajuste.

`Pipeline` encadeia estimators. Ao ser usado dentro de validação cruzada, cada transformer é ajustado exclusivamente na parcela de treino de cada dobra. Isso impede que mediana, escala, componentes PCA ou categorias do conjunto de validação vazem para o ajuste.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', Ridge()),
])
```

Parâmetros de uma etapa usam `etapa__parametro`: `model__alpha`, `model__C` ou `pca__n_components`. Para dados mistos, `ColumnTransformer` aplica pipelines diferentes às colunas numéricas e categóricas.

## Separação correta dos dados

| Parte dos dados | Finalidade |
| --- | --- |
| treino/desenvolvimento | Ajustar pipelines e comparar alternativas por validação cruzada. |
| dobras de validação | Estimar desempenho de candidatos sem usar a dobra avaliada no ajuste. |
| teste final | Medir a estimativa final de generalização depois que todas as escolhas forem congeladas. |

Reserve o teste antes de procurar modelos. Depois de consultar sua métrica, não altere modelo, features, limiar ou hiperparâmetros; caso contrário, o teste vira mais uma validação e a estimativa fica otimista.

## Estratégias de validação cruzada

| Estratégia | Quando usar | Cuidado |
| --- | --- | --- |
| `KFold` | Regressão e observações i.i.d. | Use `shuffle=True` e `random_state` quando a ordem não tiver significado. |
| `StratifiedKFold` | Classificação binária ou multiclasse | Preserva aproximadamente a proporção das classes em cada dobra. |
| `RepeatedKFold` / `RepeatedStratifiedKFold` | Poucos dados ou necessidade de estimativa mais estável | Aumenta bastante o custo. |
| `GroupKFold` | Várias linhas por paciente, empresa, usuário ou família | Um grupo nunca pode aparecer simultaneamente em treino e validação. |
| `TimeSeriesSplit` | Dados temporais | Nunca embaralhe futuro para prever passado. |
| CV aninhada | Muitas escolhas de modelo/hiperparâmetro e necessidade de estimativa menos enviesada | A busca ocorre na CV interna; a externa mede o processo inteiro. |

## Métricas e comparação

Defina a métrica antes de iniciar a busca. Em regressão, use RMSE quando erros grandes forem mais custosos e MAE quando a leitura direta na unidade do alvo for prioritária; R² complementa, mas não substitui erros absolutos. Em classificação, `roc_auc` mede ordenação geral, enquanto F1, precision e recall refletem custos de decisão; em dados muito desbalanceados, inclua PR-AUC.

`cross_validate` permite registrar várias métricas. Compare média e desvio padrão das dobras, tempo de ajuste e simplicidade do modelo. A maior média isolada não é necessariamente uma escolha robusta.

## Estratégias de busca de hiperparâmetros

Grid, Random, Bayesiana e Optuna **não** são tipos de validação cruzada: são maneiras de propor candidatos, normalmente avaliados por uma CV escolhida antes.

| Estratégia | Como propõe candidatos | Quando usar | Limitação |
| --- | --- | --- |
| `GridSearchCV` | Todas as combinações de uma grade finita | Poucos valores plausíveis e espaço pequeno | O número de ajustes cresce multiplicativamente. |
| `RandomizedSearchCV` | `n_iter` amostras de listas ou distribuições | Espaço amplo, parâmetros contínuos ou orçamento limitado | Distribuições e orçamento mal escolhidos desperdiçam testes. |
| `BayesSearchCV` | Usa avaliações anteriores para escolher candidatos promissores | Ajustes caros com muitos parâmetros | Requer `scikit-optimize`; ainda pode sobreajustar a CV. |
| Optuna | Um estudo executa *trials* sugeridos por um sampler; objetivo pode ser customizado | Espaços condicionais, orçamento maior e pruning | Requer `optuna`; a função objetivo deve encapsular a CV corretamente. |

`GridSearchCV` e `RandomizedSearchCV` vêm do scikit-learn. `BayesSearchCV` vem de `scikit-optimize`. Optuna pode usar `OptunaSearchCV` (via `optuna-integration`) ou, de forma mais flexível, uma função `objective(trial)` que instancia o pipeline e retorna a média da validação cruzada.

## Regressão: grade pequena

O notebook usa `load_diabetes` para comparar `DummyRegressor`, `Ridge` e `RandomForestRegressor`. A busca principal é:

```text
SimpleImputer → StandardScaler → Ridge
```

Use `KFold(n_splits=5, shuffle=True, random_state=42)`, RMSE como métrica de seleção e uma grade pequena para `model__alpha`. `GridSearchCV` é apropriado porque os valores candidatos são explícitos e poucos. O teste final só é usado depois da escolha.

## Classificação: espaço amplo

O notebook usa `load_breast_cancer` e separa treino/teste de forma estratificada. O fluxo é:

```text
SimpleImputer → StandardScaler → SVC(probability=True)
```

Use `StratifiedKFold`, `roc_auc` como métrica principal e F1 como métrica complementar. `C` e `gamma` variam em várias ordens de grandeza: por isso `RandomizedSearchCV` com distribuições logarítmicas é mais adequado que uma grade extensa. O mesmo pipeline e orçamento de candidatos ilustram `BayesSearchCV` e Optuna como alternativas avançadas.

## Clusterização: seleção de configuração, não CV supervisionada

Clusterização não possui, em geral, um alvo nem um teste final equivalente. K-Means pode prever a associação de pontos novos; DBSCAN não fornece `predict()` padrão. Portanto, não aplique `GridSearchCV` ou `RandomizedSearchCV` genericamente como se a tarefa fosse supervisionada.

No `load_wine`, compare `KMeans` com e sem PCA por:

- silhouette, em uma representação fixa;
- tamanho dos clusters;
- estabilidade sob reamostragem;
- interpretação dos perfis nas features originais.

Para DBSCAN, inclua também proporção de ruído e descarte soluções triviais: um único cluster, todos os pontos como ruído ou clusters muito pequenos. Uma busca Bayesiana ou Optuna só é válida se a função objetivo combinar esses critérios de forma explícita; não basta maximizar silhouette cegamente.

## Leitura de resultados e refit

Todas as buscas expõem `best_params_`, `best_score_`, `best_estimator_` e `cv_results_`. Por padrão, `refit=True` ajusta o melhor pipeline novamente usando todo o conjunto de desenvolvimento. Inspecione as primeiras posições de `cv_results_`, o desvio padrão e os tempos antes de assumir que o melhor valor é uma melhoria real.

Após escolher, aplique `best_estimator_` no teste final uma única vez. Para uma medida de incerteza, o notebook calcula um intervalo de confiança simples sobre os erros absolutos da regressão; a interpretação deve considerar tamanho e representatividade do teste.

## Dependências opcionais

Além da base do projeto, as demonstrações avançadas exigem `scikit-optimize`, `optuna` e `optuna-integration`, listados em `requirements.txt`. O notebook continua executável sem elas: as células avançadas exibem uma instrução de instalação e são ignoradas.
