# Processamento de dados com scikit-learn

Este módulo reúne as transformações que preparam os dados antes do treinamento: divisão treino/teste, tratamento de ausências, escalonamento, codificação de categorias e criação/seleção de features. O foco é entender **o que cada ferramenta aprende**, **quando usá-la** e **como evitar data leakage**.

O notebook usa `sklearn.datasets.load_diabetes` como fonte de dados numéricos e cria exemplos categóricos e ausentes apenas para demonstrar a API. Ele não é um projeto ponta a ponta.

## Glossário

| Termo | Significado |
| --- | --- |
| **Feature** / `X` | Variáveis de entrada fornecidas ao modelo. |
| **Target** / `y` | Variável que o modelo tenta prever. |
| **Fit** | Etapa em que um transformador aprende parâmetros dos dados: média, mínimo, categorias, mediana etc. |
| **Transform** | Etapa em que os parâmetros já aprendidos são aplicados a um conjunto de dados. |
| **Data leakage** | Vazamento de informação do teste para o treino, produzindo avaliação otimista e inválida. |
| **Feature numérica** | Valor mensurável, como idade, renda e temperatura. |
| **Feature categórica** | Rótulo de uma categoria, como cidade, cor ou tipo de plano. |
| **Cardinalidade** | Quantidade de categorias distintas em uma feature categórica. |
| **Imputação** | Preenchimento de valores ausentes com uma regra aprendida ou definida. |
| **Escalonamento** | Transformação da faixa ou distribuição de valores numéricos. |

## Ordem segura de processamento

1. Separe `X` e `y`.
2. Divida os dados com `train_test_split`.
3. Ajuste cada transformador apenas em `X_train` com `fit` ou `fit_transform`.
4. Aplique o mesmo transformador aprendido em `X_test` com `transform`.
5. Treine o modelo com os dados transformados de treino e avalie nos dados transformados de teste.

```python
transformador.fit(X_train)
X_train_processado = transformador.transform(X_train)
X_test_processado = transformador.transform(X_test)
```

Nesta etapa de estudo, os exemplos fazem esse fluxo explicitamente e não usam `Pipeline`.

## Divisão treino/teste — `train_test_split`

`train_test_split` separa uma parte dos dados para verificar se o modelo generaliza para observações não vistas. Os argumentos mais importantes são:

| Argumento | Padrão | Uso |
| --- | --- | --- |
| `test_size` | `None` | Fração ou quantidade reservada para teste; exemplos: `0.2`, `0.25`. |
| `train_size` | `None` | Fração ou quantidade reservada para treino; normalmente basta definir `test_size`. |
| `random_state` | `None` | Semente para reproduzir a mesma divisão; exemplo: `42`. |
| `shuffle` | `True` | Embaralha antes da divisão. Em séries temporais, normalmente use `False` e preserve a ordem. |
| `stratify` | `None` | Mantém a proporção de classes; use `stratify=y` em classificação, especialmente com desbalanceamento. |

Para regressão não há estratificação automática padrão; a divisão aleatória é suficiente em muitos casos. Em dados temporais, geográficos ou com grupos, use uma estratégia de divisão compatível com a dependência dos dados.

## Valores ausentes — imputers

O scikit-learn não aceita `NaN` em vários estimadores. Antes de imputar, investigue o motivo da ausência: erro de coleta, valor não aplicável ou um padrão com significado próprio.

| Ferramenta | Quando usar | Variações principais |
| --- | --- | --- |
| `SimpleImputer` | Primeira opção para dados numéricos ou categóricos; simples e rápida | `strategy='mean'` (numéricos sem outliers importantes), `'median'` (numéricos com assimetria/outliers), `'most_frequent'` (categorias) e `'constant'` (marcador como `'desconhecido'`) |
| `KNNImputer` | Quando observações semelhantes tendem a ter valores semelhantes | `n_neighbors` controla vizinhos; requer escala comparável e pode ser lento em bases grandes |
| `MissingIndicator` | Quando a própria ausência pode ser informativa | Cria uma coluna booleana por feature com valor ausente; pode complementar uma imputação |

Exemplo seguro com mediana:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

## Escalonamento — scalers

Escalonar é importante quando o modelo usa distância, margem ou regularização. KNN, SVM, K-Means e Ridge/Lasso/Elastic Net são sensíveis à escala. Árvores e Random Forest dividem valores por limiares e geralmente não precisam de escala.

| Ferramenta | Transformação | Quando usar | Limitação |
| --- | --- | --- | --- |
| `MinMaxScaler` | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$; por padrão, leva para `[0, 1]` | Escolha padrão deste repositório para modelos sensíveis à escala; preserva a ordem e limites relativos | Um outlier altera mínimo/máximo e pode comprimir os demais valores |
| `StandardScaler` | $x' = \frac{x - \mu}{\sigma}$; média 0 e desvio padrão 1 | Útil quando a hipótese do modelo ou o algoritmo se beneficia de features centradas | Média e desvio padrão são afetados por outliers |
| `MaxAbsScaler` | Divide cada feature pelo maior valor absoluto | Matrizes esparsas, pois preserva zeros | Não centraliza dados e também é afetado por extremos |
| `Normalizer` | Normaliza cada **linha** para norma L1, L2 ou máxima | Comparar perfis/vetores, como documentos em texto | Não é um scaler por coluna; raramente é a escolha para dados tabulares comuns |

Para `MinMaxScaler`, `feature_range=(0, 1)` é o padrão; exemplos são `feature_range=(-1, 1)` ou `(0, 100)`. O notebook do módulo 001 usa `MinMaxScaler` antes dos modelos regularizados.

## Features categóricas — encoders

| Ferramenta | Quando usar | Variações e cuidados |
| --- | --- | --- |
| `OneHotEncoder` | Categorias nominais, sem ordem real: cidade, cor, plano | Cria uma coluna 0/1 por categoria. Use `handle_unknown='ignore'` para categorias novas no teste; `drop='first'` pode reduzir colinearidade em modelos lineares; alta cardinalidade cria muitas colunas |
| `OrdinalEncoder` | Categorias de entrada com ordem real: baixo, médio, alto | Informe a ordem das categorias quando ela for conhecida. Não use em categorias nominais: números artificiais induzem uma ordem inexistente |
| `LabelEncoder` | Apenas no target `y` de classificação | Não use para features `X`; para essas, prefira OneHot ou Ordinal Encoder |

Para categorias ausentes, é comum imputar primeiro com `SimpleImputer(strategy='most_frequent')` ou `strategy='constant'` e depois codificar.

## Outras ferramentas úteis

| Ferramenta | Para que serve | Quando aplicar |
| --- | --- | --- |
| `PolynomialFeatures` | Cria potências e interações entre features | Quando um modelo linear precisa representar curvaturas/interações; cresce rapidamente com muitas features |
| `FunctionTransformer` | Aplica uma função customizada, como `np.log1p` | Para transformações determinísticas justificadas pelo domínio; `log1p` exige valores maiores ou iguais a -1 |
| `KBinsDiscretizer` | Converte valores numéricos em faixas | Para regras, visualização ou relações não lineares por intervalos; pode perder informação contínua |
| `VarianceThreshold` | Remove features com variância abaixo de um limite | Para descartar colunas constantes ou quase constantes |
| `SelectKBest` | Seleciona as `k` features mais associadas ao target segundo um teste estatístico | Apenas no treino, e dentro da validação ao comparar modelos; em regressão, use funções como `f_regression` ou `mutual_info_regression` |
| `ColumnTransformer` | Aplica transformações diferentes por conjunto de colunas | Em dados mistos: imputer/scaler para numéricas e imputer/encoder para categóricas. Será abordado quando o curso introduzir composição de transformações |

## Referências

- Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2ª edição: capítulo 2, especialmente preparação de dados, limpeza, atributos categóricos e escalonamento de features.
