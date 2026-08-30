# Regressão com scikit-learn

Este módulo apresenta modelos de **regressão supervisionada**: a variável-alvo `y` é numérica e contínua. Exemplos: estimar preço de um imóvel, consumo de energia, tempo de entrega ou progressão de uma doença.

O notebook correspondente usa `sklearn.datasets.load_diabetes`, dataset já disponível na biblioteca, exclusivamente para demonstrar a API e comparar famílias de modelos. Não é uma solução de negócio nem um projeto ponta a ponta.

## Glossário e notação

| Termo ou símbolo | Legenda |
| --- | --- |
| **Feature** / `X` | Variável de entrada usada pelo modelo. `X` representa a matriz de todas as features. |
| **Target** / `y` | Valor real contínuo que se deseja prever. |
| `x` | Vetor de features de uma única observação. |
| `yᵢ` | Valor real do target na observação `i`. |
| `ŷᵢ` | Predição do modelo para a observação `i`. O acento circunflexo (“chapéu”) indica valor estimado. |
| `eᵢ` | Resíduo ou erro assinado: `yᵢ − ŷᵢ`. |
| `n` | Quantidade de observações avaliadas. |
| `p` | Quantidade de features. |
| `β₀` | Intercepto: previsão-base quando as features assumem valor zero. |
| `βⱼ` | Coeficiente associado à feature `xⱼ` em modelos lineares. |
| `α` | Intensidade da regularização em Ridge, Lasso e Elastic Net. |
| `ρ` | Proporção da penalidade L1 no Elastic Net; corresponde a `l1_ratio`. |
| `Rₘ` | Região do espaço de features criada por uma folha de árvore. |
| `𝟙(·)` | Função indicadora: vale 1 se a condição é verdadeira e 0 caso contrário. |
| `B` | Número de árvores em uma Random Forest. |
| **Regularização** | Penalidade acrescentada ao ajuste para reduzir complexidade e overfitting. |
| **Outlier** | Observação muito distante do padrão geral; pode ser erro, caso raro ou informação relevante. |

## Dataset demonstrativo

O módulo utiliza o dataset **Diabetes** incluído no scikit-learn. Ele contém 442 observações, 10 features numéricas e um alvo contínuo que representa uma medida quantitativa de progressão da doença após um ano. Carregue-o sem downloads externos:

```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
X = diabetes.data
y = diabetes.target
```

Como as features já foram transformadas na versão distribuída pela biblioteca, o dataset serve para aprender a API e comparar modelos. O notebook ainda aplica `MinMaxScaler` explicitamente para demonstrar o procedimento exigido por modelos sensíveis à escala.

## Métricas

| Métrica | O que mede | Melhor valor | Quando priorizar |
| --- | --- | --- | --- |
| MAE — `mean_absolute_error` | Média dos erros absolutos | Menor | Quando todos os erros devem ter peso proporcional e a métrica precisa permanecer na unidade de `y` |
| RMSE — raiz do MSE | Dá peso maior a erros grandes | Menor | Quando desvios grandes são particularmente indesejáveis |
| R² — `r2_score` | Variação explicada em relação a prever a média | Maior; máximo 1 | Para comunicar ganho relativo, sempre junto de MAE ou RMSE |

MAE e RMSE estão na mesma unidade do alvo. R² pode ser negativo no conjunto de teste: isso significa que o modelo foi pior do que a baseline que prevê a média.

### Como a metrificação é calculada

O modelo aprende somente com `X_train` e `y_train`. Depois, ele recebe as features que não viu, `X_test`, e produz uma previsão para cada linha: `y_pred = modelo.predict(X_test)`. A avaliação compara essa previsão com os valores reais preservados em `y_test`.

Para cada observação `i`, o resíduo (erro assinado) é:

$$e_i = y_i - \hat{y}_i$$

em que `yᵢ` é o valor real e `ŷᵢ` é a previsão. Para `n` observações de teste:

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 \qquad RMSE = \sqrt{MSE}$$

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y}_{test})^2}$$

MAE preserva o peso proporcional de cada erro. Ao elevar os resíduos ao quadrado, MSE e RMSE penalizam mais fortemente grandes desvios. R² compara a soma dos erros quadrados do modelo com prever a média de `y_test`.

## Baseline

Antes de comparar modelos, use `DummyRegressor(strategy="mean")`. Ele ignora as features e sempre prevê a média do alvo observada no treino. Um regressor só é útil se superar essa referência no conjunto de teste.

```python
from sklearn.dummy import DummyRegressor

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
previsoes = baseline.predict(X_test)
```

A previsão da baseline é $\hat{y} = \bar{y}_{train}$ para todas as linhas de teste. A média é calculada a partir do treino, nunca do teste.

## Equações dos modelos

Considere uma observação com features $x = (x_1, \ldots, x_p)$.

### Regressão linear — `LinearRegression`

$$
\hat{y} = \beta_0 + \sum_{j=1}^{p}\beta_jx_j
$$

O ajuste por mínimos quadrados ordinários (*Ordinary Least Squares*, OLS) escolhe os coeficientes que minimizam a soma dos erros quadrados:

$$
\underset{\beta_0,\ldots,\beta_p}{\operatorname{minimizar}}
\quad
\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

### Ridge — `Ridge`

$$
\underset{\beta_0,\ldots,\beta_p}{\operatorname{minimizar}}
\quad
\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
+ \alpha\sum_{j=1}^{p}\beta_j^2
$$

A penalidade L2 reduz coeficientes muito grandes, mas em geral não os torna exatamente zero.

### Lasso — `Lasso`

$$
\underset{\beta_0,\ldots,\beta_p}{\operatorname{minimizar}}
\quad
\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
+ \alpha\sum_{j=1}^{p}|\beta_j|
$$

A penalidade L1 pode zerar coeficientes, produzindo uma seleção de features embutida.

### Elastic Net — `ElasticNet`

$$
\underset{\beta_0,\ldots,\beta_p}{\operatorname{minimizar}}
\quad
\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
+ \alpha\left(
    \rho\sum_{j=1}^{p}|\beta_j|
    + \frac{1 - \rho}{2}\sum_{j=1}^{p}\beta_j^2
  \right)
$$

Combina as penalidades L1 e L2. No scikit-learn, $\rho$ é controlado por `l1_ratio`.

### Árvore de decisão — `DecisionTreeRegressor`

Uma árvore divide o espaço das features em $M$ regiões. Em cada região $R_m$, a predição é uma constante $c_m$, normalmente a média do target nas observações daquela folha:

$$
\hat{y}(x) = \sum_{m=1}^{M} c_m\,\mathbb{1}(x \in R_m)
$$

Essa forma descreve regras por faixas e combinações de features, não uma relação linear global.

### Random Forest — `RandomForestRegressor`

Uma Random Forest treina $B$ árvores com amostras e subconjuntos de features aleatórios. A predição final é a média das predições individuais:

$$
\hat{y}_{RF}(x) = \frac{1}{B}\sum_{b=1}^{B}\hat{y}_b(x)
$$

Ao fazer a média, a floresta tende a reduzir a variância de uma árvore isolada.

Nas regressões regularizadas, a escala das features altera a penalização aplicada aos coeficientes; por isso o `MinMaxScaler` é usado no notebook antes de Ridge, Lasso e Elastic Net.

## Modelos demonstrados

| Modelo | Quando tende a funcionar bem | Escala e outliers | Pontos de atenção |
| --- | --- | --- | --- |
| `LinearRegression` | Relação aproximadamente linear; baseline interpretável | Escala não altera a previsão, mas ajuda a comparar coeficientes; outliers podem deslocar bastante a reta | Não captura curvaturas e interações sem criar features |
| `Ridge` | Muitas features correlacionadas; controle de coeficientes muito grandes | Use `MinMaxScaler` para tornar a penalidade comparável; ainda é sensível a outliers | Ajustar `alpha`; quanto maior, maior a regularização |
| `Lasso` | Quando é desejável zerar alguns coeficientes e selecionar features | Use `MinMaxScaler`; outliers podem alterar seleção e coeficientes | Com features muito correlacionadas, pode escolher uma e descartar outras arbitrariamente |
| `ElasticNet` | Features correlacionadas com necessidade de regularização e possível seleção | Use `MinMaxScaler`; sensível a outliers | Combina as penalidades L1 e L2; ajustar `alpha` e `l1_ratio` |
| `DecisionTreeRegressor` | Regras, interações e não linearidades | Não requer escala; costuma ser mais tolerante a outliers nas features | Pode sobreajustar; controlar profundidade e tamanho mínimo das folhas |
| `RandomForestRegressor` | Dados tabulares com padrões não lineares e interações | Não requer escala; relativamente tolerante a outliers nas features | Menos interpretável; não extrapola bem além da faixa observada |

## Escala, outliers e formato da relação

- `MinMaxScaler` transforma cada feature para um intervalo definido (no notebook, `[0, 1]`). Ajuste-o com `fit` **somente em `X_train`** e use `transform` em treino e teste. Assim, os limites do teste não vazam para o treinamento.
- Em modelos regularizados, o escalonamento deixa a penalização dos coeficientes comparável entre features.
- Árvores e Random Forest dividem valores por limiares; por isso não exigem escalonamento. Mesmo assim, outliers, dados ausentes e erros de medição merecem investigação.
- Nenhum modelo deve receber outliers cegamente: confirme se são observações válidas, erros de medição ou casos raros importantes. A decisão de removê-los ou transformá-los depende do problema e deve ser documentada.

## Importância de features e SHAP

`feature_importances_` em modelos de árvore resume a redução de impureza atribuída a cada variável. É rápido, mas pode favorecer features contínuas ou com muitos valores possíveis; não informa direção nem causalidade.

`permutation_importance` embaralha uma feature e mede quanto a métrica piora. Por estar ligada ao modelo e ao conjunto de avaliação, é uma comparação mais útil entre variáveis, embora features correlacionadas possam parecer pouco importantes individualmente.

SHAP (*SHapley Additive exPlanations*) explica uma previsão pela contribuição de cada feature em relação a um valor-base. Ele oferece leituras:

- **locais**: por que uma observação recebeu determinada predição;
- **globais**: quais features mais deslocam as predições, ao agregar valores absolutos de SHAP.

SHAP descreve o comportamento do modelo, não causalidade. Use-o depois de validar o desempenho e interprete-o com conhecimento de domínio. A biblioteca `shap` é uma dependência opcional e não faz parte do scikit-learn.

## Referência

Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2ª edição. Este módulo se apoia especialmente nas discussões sobre regressão linear, regressão polinomial, regularização e curvas de aprendizado (capítulo 4), regressão por árvores (capítulo 6), ensembles e importância de features (capítulo 7).
