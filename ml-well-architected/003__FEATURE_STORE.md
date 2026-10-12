# Feature store

## Responsabilidade

<!-- Explicar como centralizar definições e versões de features para evitar divergência entre treino e inferência. -->

## Entradas e saídas

| Entradas | Saídas |
| --- | --- |
| <!-- Features calculadas, entidades e metadados --> | <!-- Features históricas offline e consulta online, quando necessária --> |

## Recursos de infraestrutura

- <!-- Store offline para treino e batch scoring. -->
- <!-- Store online de baixa latência para inferência em tempo real. -->
- <!-- Catálogo, versionamento e política de atualização. -->

## Decisões de arquitetura

- <!-- Quando um feature store é necessário; consistência ponto no tempo; latência versus custo. -->
- <!-- FUTURO: explicar como a store offline produz datasets de treino point-in-time corretos e como a store online serve a mesma definição de feature na inferência, evitando training-serving skew. -->

## Opções AWS

- <!-- Ex.: SageMaker Feature Store, Amazon S3, DynamoDB, ElastiCache. -->

## Relação com este repositório

<!-- Relacionar com as transformações persistidas pelo Pipeline do scikit-learn. -->

## Referências

- [AWS Well-Architected ML Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/architecture-diagram.html)
