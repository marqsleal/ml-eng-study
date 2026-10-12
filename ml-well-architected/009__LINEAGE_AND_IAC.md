# Lineage e infraestrutura como código

## Responsabilidade

<!-- Explicar como reproduzir uma versão do sistema ao rastrear infraestrutura, dados, código, features e modelo. -->

## Entradas e saídas

| Entradas | Saídas |
| --- | --- |
| <!-- Commits, configurações, metadados, artefatos e versões de dados --> | <!-- Linhagem auditável e ambiente reproduzível em um ponto no tempo --> |

## Recursos de infraestrutura

- <!-- Repositório Git e revisão de mudanças. -->
- <!-- IaC, estado da infraestrutura e gerenciamento de segredos. -->
- <!-- Metadados de dados, features, imagens e modelos. -->
- <!-- FUTURO: detalhar a ligação entre uma solicitação de exclusão/anonimização e todos os snapshots, features, experimentos e modelos derivados para suportar análise de impacto e auditoria. -->

## Decisões de arquitetura

- <!-- Granularidade de versionamento; imutabilidade; retenção; separação entre ambientes. -->

## Opções AWS

- <!-- Ex.: CloudFormation ou CDK, AWS Config, ECR, S3 versionado, SageMaker Lineage. -->

## Relação com este repositório

<!-- Relacionar com Git, requirements, Dockerfile, notebooks e artefato do modelo. -->

## Referências

- [AWS Well-Architected ML Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/architecture-diagram.html)
