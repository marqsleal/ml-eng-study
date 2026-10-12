# Automação de retreinamento

## Responsabilidade

<!-- Explicar como scheduler, alarm manager e pipelines controlam novas versões de modelo. -->

## Entradas e saídas

| Entradas | Saídas |
| --- | --- |
| <!-- Agenda, alerta de drift ou feedback de negócio --> | <!-- Execução auditável de preparação, treino, avaliação e possível promoção --> |

## Recursos de infraestrutura

- <!-- Scheduler, gerenciador de alarmes e orquestrador de pipeline. -->
- <!-- Fila/eventos e gates de aprovação. -->
- <!-- Registro de execução e rollback. -->

## Decisões de arquitetura

- <!-- Retreinamento periódico versus orientado por evento; automático versus revisão humana. -->
- <!-- FUTURO: definir a resposta quando retirada de consentimento, exclusão de dados ou correção material invalidar um snapshot de treino: avaliar impacto, aprovar retreino e substituir ou descontinuar o modelo. -->

## Opções AWS

- <!-- Ex.: EventBridge, CloudWatch Alarms, Step Functions, SageMaker Pipelines, SNS/SQS. -->

## Relação com este repositório

<!-- Definir quais evidências justificariam reexecutar o notebook/pipeline ponta a ponta. -->

## Referências

- [AWS Well-Architected ML Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/architecture-diagram.html)
