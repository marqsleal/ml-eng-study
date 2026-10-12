# Preparação de dados

## Responsabilidade

<!-- Descrever limpeza, validação, anonimização, transformação e criação de datasets prontos para treino ou inferência. -->

## Entradas e saídas

| Entradas | Saídas |
| --- | --- |
| <!-- Dados brutos e regras de qualidade --> | <!-- Dados validados, transformados e versionados --> |

## Recursos de infraestrutura

- <!-- Processamento batch ou distribuído. -->
- <!-- Orquestração, validação de esquema e armazenamento intermediário. -->
- <!-- Registro de execução e de qualidade. -->

## Decisões de arquitetura

- <!-- Transformações idempotentes, reprocessamento e segregação de dados sensíveis. -->
- <!-- FUTURO: detalhar point-in-time correctness: preservar event_time e as-of time, executar joins temporais, evitar atributos conhecidos após a data de corte e dividir treino/validação/teste respeitando a ordem temporal. -->

## Opções AWS

- <!-- Ex.: AWS Glue, Amazon EMR, SageMaker Processing, Step Functions. -->

## Relação com este repositório

<!-- Relacionar com imputação, escalonamento, codificação e prevenção de leakage. -->

## Referências

- [AWS Well-Architected ML Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/architecture-diagram.html)
