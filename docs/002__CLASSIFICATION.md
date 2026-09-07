# Classificação com scikit-learn

Classificação supervisionada prevê categorias discretas: fraude ou não fraude, espécie, diagnóstico, prioridade ou faixa de risco. O notebook usa `load_breast_cancer`, dataset nativo com 569 observações e 30 features numéricas, para comparar classificadores sem downloads externos. A classe positiva é tumor maligno.

## Glossário

| Termo | Significado |
| --- | --- |
| `X` | matriz de features usada como entrada |
| `y` | classe real que se deseja prever |
| classe positiva | evento de interesse, codificado como `1` |
| TP / FP | verdadeiro positivo / falso positivo |
| TN / FN | verdadeiro negativo / falso negativo |
| limiar | valor que converte escore ou probabilidade em classe |
| regularização | restrição que reduz a complexidade e o overfitting |

## Dataset e divisão

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer(as_frame=True)
X = cancer.data
y = (cancer.target == 0).astype(int)  # 1 = maligno
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

`stratify=y` preserva a proporção de classes. Ajuste um `StandardScaler` apenas com `X_train` e transforme treino e teste separadamente. KNN, SVC e regressão logística dependem disso; árvores e ensembles de árvores não.

## Métricas

| Métrica | Fórmula | Quando priorizar |
| --- | --- | --- |
| Accuracy | `(TP + TN) / total` | classes e custos de erro semelhantes |
| Precision | `TP / (TP + FP)` | falso positivo é caro |
| Recall | `TP / (TP + FN)` | falso negativo é caro |
| F1 | média harmônica de precision e recall | é necessário equilibrar os dois |
| ROC-AUC | área sob a curva ROC | capacidade geral de ordenar exemplos |
| Average Precision | resumo da curva precision-recall | classe positiva rara ou alertas positivos caros |

Accuracy isolada pode enganar em dados desbalanceados. A matriz de confusão sempre deve acompanhar a métrica: em diagnóstico, um FN pode ser muito mais grave que um FP.

## Baseline

`DummyClassifier(strategy="most_frequent")` ignora as features e prevê apenas a classe dominante. Um modelo só traz valor se superar essa referência na métrica alinhada ao problema.

## Modelos demonstrados

| Modelo | Quando usar | Hiperparâmetros iniciais | Escala / atenção |
| --- | --- | --- | --- |
| `LogisticRegression` | baseline probabilística, interpretável e aproximadamente linear | `C`, `penalty`, `class_weight`, `max_iter` | requer escala; `C` menor regulariza mais |
| `KNeighborsClassifier` | fronteiras locais não lineares, dados pequenos a médios | `n_neighbors`, `weights`, `p` | requer escala; predição piora em grandes volumes |
| `SVC` | fronteiras complexas e datasets moderados | `C`, `kernel`, `gamma` | requer escala; pode ser caro em grande volume |
| `DecisionTreeClassifier` | regras interpretáveis, interações e não linearidade | `max_depth`, `min_samples_leaf`, `criterion`, `class_weight` | não requer escala; sem limites sobreajusta |
| `RandomForestClassifier` | baseline robusta para tabelas heterogêneas | `n_estimators`, `max_features`, `max_depth`, `min_samples_leaf` | não requer escala; menos interpretável |
| `ExtraTreesClassifier` | alternativa aleatorizada à Random Forest | `n_estimators`, `max_features`, `max_depth`, `min_samples_leaf` | não requer escala; compare sob mesma validação |
| `HistGradientBoostingClassifier` | desempenho em tabelas maiores e relações complexas | `learning_rate`, `max_iter`, `max_leaf_nodes`, `l2_regularization` | não requer escala; validar para evitar overfitting |

`LinearSVC` é uma boa alternativa a `SVC` quando há muitas features, especialmente matrizes esparsas de texto, e a fronteira pode ser linear. Para probabilidades, calibre-o com `CalibratedClassifierCV`.

## Limiar e probabilidades

Muitos classificadores fornecem `predict_proba`; o padrão converte em positivo quando a probabilidade é pelo menos 0,5. O limiar não é uma verdade estatística: diminui-lo costuma aumentar recall e falsos positivos; aumentá-lo costuma aumentar precision e falsos negativos. Escolha-o usando validação no treino e documente o custo de erro considerado.

## Próximo passo

O notebook compara modelos em uma divisão fixa para tornar a API visível. Para seleção real, escolha a métrica antes de ajustar, aplique validação cruzada estratificada no treino e só então faça uma avaliação final no teste. Esse fluxo é aprofundado no módulo 004.

## Referência

Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2ª edição, capítulo 3; documentação oficial do [scikit-learn sobre classificação](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning).
