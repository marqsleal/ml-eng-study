# Clusterização com scikit-learn

Clusterização é aprendizado não supervisionado: o algoritmo recebe features `X`, sem uma variável-alvo `y`, e procura grupos de observações semelhantes segundo uma medida de distância. É útil para segmentação de clientes, exploração de padrões, agrupamento de documentos e identificação de perfis. Um cluster não é uma verdade descoberta automaticamente: ele precisa ser coerente com os dados, estável e interpretável no contexto do problema.

O notebook deste módulo deve usar datasets sintéticos para tornar as hipóteses dos algoritmos visíveis e `load_wine` como exemplo tabular real. Os rótulos disponíveis em `load_wine` não participam do ajuste; podem ser usados apenas depois, como referência didática externa.

## Glossário

| Termo | Significado |
| --- | --- |
| `X` | Matriz de features usada para medir semelhança entre observações. |
| cluster | Grupo de observações mais semelhantes entre si do que às demais, segundo a representação escolhida. |
| centróide | Média das observações atribuídas a um grupo no K-Means. |
| inércia | Soma das distâncias quadradas entre cada ponto e o centróide de seu cluster; menor é melhor apenas ao comparar o mesmo dataset para diferentes valores de `k`. |
| silhouette | Mede coesão dentro do grupo e separação em relação ao grupo vizinho; varia aproximadamente de `-1` a `1`. |
| ponto de ruído | Observação que o DBSCAN não atribui a um grupo, rotulada como `-1`. |
| `eps` | Raio de vizinhança do DBSCAN; depende diretamente da escala e da métrica escolhidas. |
| `min_samples` | Quantidade mínima de pontos na vizinhança para que um ponto seja considerado núcleo pelo DBSCAN. |
| PCA | Redução linear de dimensionalidade que cria componentes ortogonais com máxima variância possível. |

## Antes de ajustar: o que significa “semelhante”?

Em clusterização, a preparação não é apenas uma etapa técnica: ela define a noção de proximidade. Com distância euclidiana, uma feature medida entre 0 e 100.000 pode dominar outra medida entre 0 e 100, mesmo que ambas sejam igualmente importantes para o problema. IDs, chaves, datas brutas e colunas derivadas do mesmo evento também podem fabricar grupos sem significado.

1. Escolha features que descrevem o perfil que se quer segmentar; remova identificadores e variáveis que não devem influenciar a semelhança.
2. Investigue ausências, outliers e cardinalidade antes de transformá-los. Um outlier pode ser erro, caso raro relevante ou exatamente o comportamento que se quer detectar.
3. Impute valores ausentes e aplique uma escala compatível com a distância.
4. Ajuste o algoritmo e interprete os perfis de cada grupo nas unidades originais.

Em um estudo exploratório, é aceitável ajustar a preparação e o clusterizador sobre todos os dados depois de definir as escolhas. Para escolher transformações e hiperparâmetros de maneira mais rigorosa, compare alternativas em reamostragens e avalie a estabilidade dos grupos; não há uma divisão treino/teste automática equivalente à de tarefas supervisionadas.

## Preparação das features

| Situação | Preparação recomendada | Quando usar / cuidado |
| --- | --- | --- |
| Features numéricas em escalas diferentes | `SimpleImputer` + `StandardScaler` | Ponto de partida para K-Means e DBSCAN, que dependem de distância. |
| Outliers distorcem média e desvio | Investigar a causa; considerar `RobustScaler` | Útil se extremos válidos não devem dominar a distância. Não elimine automaticamente pontos raros: no DBSCAN eles podem ser o resultado procurado. |
| Features numéricas já comparáveis em unidade e amplitude | Imputação, se necessária; escala opcional | Mantenha a escala original somente se ela expressar corretamente a importância relativa das variáveis. |
| Categorias nominais | Imputação + `OneHotEncoder(handle_unknown='ignore')` | Serve para segmentação mista, mas alta cardinalidade cria muitas dimensões e pode tornar distância euclidiana pouco informativa. |
| Texto ou matriz esparsa | `TfidfVectorizer` e, em geral, `Normalizer` | Adequado quando importa a direção/perfil do vetor. Para reduzir dimensão esparsa, prefira `TruncatedSVD`, não `PCA`. |
| Perfil relativo por observação | `Normalizer` por linha | Ex.: documentos, composição de produtos ou proporções. Não é substituto de `StandardScaler`, que escala colunas. |
| Muitas numéricas correlacionadas ou ruidosas | `StandardScaler` + `PCA` | Pode reduzir redundância e aliviar problemas de alta dimensionalidade. Avalie se os clusters permanecem estáveis e interpretáveis. |

Para dados tabulares mistos, `ColumnTransformer` permite tratar colunas numéricas e categóricas de modos diferentes. A decisão sobre pesos também é de domínio: se uma feature deve influenciar duas vezes mais a segmentação, isso deve ser documentado e aplicado conscientemente, não surgir da unidade de medida.

## PCA: redução e visualização

PCA projeta dados em componentes lineares ortogonais ordenados pela variância explicada. No scikit-learn, `PCA` já centraliza as colunas, mas **não** as padroniza; portanto, em dados com unidades diferentes, use `StandardScaler` antes.

```python
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

wine = load_wine(as_frame=True)
X = wine.data

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=0.95, random_state=42)
X_reduced = pca.fit_transform(X_scaled)

print('Componentes mantidos:', pca.n_components_)
print('Variância explicada:', pca.explained_variance_ratio_.sum())
```

`n_components=0.95` é um candidato inicial: mantém o menor número de componentes cuja variância acumulada alcance 95%. Não é uma regra de qualidade de clusters. Variância preservada não garante que a separação entre grupos, ou a semântica de uma feature, também foi preservada.

| Objetivo | Uso de PCA |
| --- | --- |
| Visualizar dados em duas dimensões | Use `PCA(n_components=2)` para o gráfico. Trate essa projeção como visualização, não como prova de que os grupos são separados no espaço original. |
| Reduzir muitas features numéricas correlacionadas antes de clusterizar | Use `PCA(n_components=0.90)` ou `0.95` como alternativa a comparar com o espaço escalonado original. |
| Reduzir matriz esparsa de one-hot ou TF-IDF | Use `TruncatedSVD`; PCA tende a exigir uma matriz densa. |
| Preservar interpretação direta de cada feature | Evite PCA no modelo principal. Os componentes são combinações das variáveis originais; ainda é possível interpretar os clusters agregando as features originais. |
| Dados com estrutura fortemente curva | PCA pode não expor a estrutura. Para este módulo, prefira demonstrar DBSCAN em `make_moons`; métodos não lineares podem ficar para um aprofundamento posterior. |

Use a projeção 2D separadamente do dado usado para ajustar o modelo quando necessário. Por exemplo, pode-se ajustar K-Means em `X_scaled` ou `X_reduced` com 95% de variância, e usar outra instância `PCA(n_components=2)` somente para desenhar os rótulos encontrados.

## K-Means

K-Means define antecipadamente `k`, inicializa `k` centróides e repete dois passos até convergir: atribui cada ponto ao centróide mais próximo e atualiza cada centróide pela média dos pontos atribuídos. Ele minimiza a inércia.

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=0.75, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
labels = kmeans.fit_predict(X_scaled)

print('Inércia:', kmeans.inertia_)
print('Centróides:', kmeans.cluster_centers_)
```

| Aspecto | Orientação |
| --- | --- |
| Melhor cenário | Grupos compactos, aproximadamente esféricos, com variância e tamanho relativamente semelhantes. |
| `n_clusters` | Deve ser definido. Use conhecimento de domínio e compare candidatos com inércia, silhouette, estabilidade e utilidade prática. |
| Inicialização | Use `k-means++` (padrão) e mais de uma inicialização; `n_init=20` torna o resultado reproduzível entre versões. |
| Escala | Normalmente necessária; o algoritmo usa distância euclidiana e centróides. |
| Limitações | Não identifica ruído, impõe um grupo a cada ponto e costuma falhar em grupos alongados, anelares ou com densidades muito diferentes. |
| Volume grande | `MiniBatchKMeans` reduz custo com pequenos lotes, em troca de uma solução aproximada. |

### Escolha de `k`

```python
from sklearn.metrics import silhouette_score

resultados = []
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = model.fit_predict(X_scaled)
    resultados.append({
        'k': k,
        'inertia': model.inertia_,
        'silhouette': silhouette_score(X_scaled, labels),
    })
```

- **Cotovelo:** procure no gráfico de `inertia` versus `k` o ponto onde adicionar grupos deixa de reduzir a inércia de forma relevante. Como a inércia sempre cai, ela não escolhe `k` sozinha.
- **Silhouette:** quanto maior, mais coeso e separado é o agrupamento; valores negativos indicam que observações podem estar mais próximas de outro cluster. Compare apenas configurações sobre a mesma representação dos dados.
- **Decisão final:** confirme se os grupos têm tamanhos úteis, se se mantêm sob pequenas mudanças da amostra e se seus perfis respondem a uma pergunta real. Não escolha `k` só pela maior métrica.

## DBSCAN

DBSCAN forma grupos por densidade. Um ponto é núcleo quando possui ao menos `min_samples` observações em seu raio `eps`; pontos conectados a núcleos compõem o mesmo cluster. Pontos que não se conectam a regiões densas recebem o rótulo `-1` de ruído. Diferentemente de K-Means, não é preciso informar previamente o número de grupos.

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X, _ = make_moons(n_samples=500, noise=0.07, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.25, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
print('Rótulos encontrados:', sorted(set(labels)))  # -1 representa ruído
```

| Aspecto | Orientação |
| --- | --- |
| Melhor cenário | Grupos de formas arbitrárias, presença de ruído e número de grupos desconhecido. |
| `eps` | Raio máximo de vizinhança. É o parâmetro mais sensível e muda de significado se a escala ou a métrica mudar. |
| `min_samples` | Controla a densidade mínima. Aumentá-lo torna o algoritmo mais conservador e pode aumentar pontos rotulados como ruído. |
| Escala | Essencial com distância euclidiana. Compare `eps` somente após fixar a preparação. |
| Limitações | Um único par de parâmetros tem dificuldade com clusters de densidades muito diferentes; alta dimensionalidade torna distâncias menos discriminativas. |

Um ponto de partida para `min_samples` é 5, mas ele deve refletir o menor grupo denso que tem significado no domínio. Para explorar `eps`, plote a distância até o `min_samples`-ésimo vizinho mais próximo em ordem crescente e procure uma mudança de inclinação; em seguida teste valores próximos e inspecione o resultado.

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])
# plt.plot(k_distances): o “joelho” sugere candidatos para eps.
```

Ao calcular silhouette no DBSCAN, exclua pontos `-1` e só calcule a métrica se restarem pelo menos dois clusters e observações suficientes. A proporção de ruído deve ser reportada junto da silhouette: uma configuração que rotula quase tudo como ruído não é automaticamente boa.

## Avaliação e interpretação

Sem rótulos verdadeiros, métricas internas medem apenas características geométricas da representação escolhida. Elas não garantem valor de negócio nem causalidade.

| Evidência | O que responde | Limitação |
| --- | --- | --- |
| Inércia | Quão próximos os pontos estão de seus centróides no K-Means | Sempre diminui quando `k` cresce; não existe para DBSCAN. |
| Silhouette | Coesão e separação relativa dos clusters | Favorece grupos compactos e pode não representar bem formas complexas. |
| Tamanho e proporção de ruído | Se há grupos utilizáveis e quanto DBSCAN descartou | Não indica sozinho se os grupos são semanticamente válidos. |
| Estabilidade | Se grupos parecidos reaparecem sob reamostragem ou pequenas variações de parâmetros | Exige comparações adicionais; é especialmente importante em segmentação. |
| Perfil nas features originais | O que distingue os grupos em unidades compreensíveis | Associação não é causalidade. |
| ARI / NMI com rótulo externo | Alinhamento didático ou validação quando existe uma referência independente | Nunca use esses rótulos para escolher ou treinar o agrupamento se o objetivo é não supervisionado. |

Os identificadores numéricos dos clusters são arbitrários: o cluster `0` não é melhor nem anterior ao cluster `1`. Para interpretar, adicione os rótulos gerados para o dataset em questão a uma cópia de suas features originais e compare medianas, médias, distribuições e tamanho dos grupos.

```python
labels_wine = KMeans(n_clusters=3, n_init=20, random_state=42).fit_predict(X_scaled)
perfil = wine.data.copy()
perfil['cluster'] = labels_wine
display(perfil.groupby('cluster').agg(['mean', 'median', 'count']))
```

## Erros comuns

- Usar ID de cliente, CEP bruto ou uma coluna que revela o resultado desejado como feature de distância.
- Aplicar K-Means ou DBSCAN sem considerar escala, deixando uma unidade dominar todas as demais.
- Escolher `k` pelo maior valor de inércia, ou escolher DBSCAN apenas pela maior quantidade de clusters.
- Usar PCA em 2D para o ajuste sem comparar com o espaço completo ou com PCA que preserve variância suficiente.
- Interpretar pontos `-1` como erros automaticamente; podem ser anomalias relevantes, grupos pouco densos ou consequência de `eps`/`min_samples` inadequados.
- Concluir que clusters são classes reais ou causalmente distintos apenas porque um gráfico os separou.
