# Coleta de dados

A coleta de dados é a primeira parte operacional do ciclo de ML. Ela transforma dados de negócio em um conjunto rastreável, acessível e autorizado para treinamento, validação e inferência. Não é uma etapa de “copiar tudo para um bucket”: o que entra define o limite de qualidade, segurança e utilidade do modelo que será treinado depois.

No diagrama do *AWS Well-Architected Machine Learning Lens*, a coleta contém dois blocos: **rotulagem** e **ingestão/agregação**. As fontes podem ser séries temporais, eventos, sensores, dispositivos IoT e redes sociais. A ingestão pode ser em lote ou em tempo real; o destino pode combinar banco transacional, data lake, data warehouse, catálogo e pipeline ETL. [AWS: Data collection](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/data-collection.html)

## Responsabilidade

A camada de coleta deve:

- identificar os dados estritamente necessários para a pergunta de negócio e para a variável-alvo;
- registrar origem, proprietário, permissão de uso, esquema, frequência e período de retenção de cada conjunto;
- capturar e preservar dados brutos de fontes confiáveis, sem alterar silenciosamente o conteúdo original;
- produzir datasets que possam ser encontrados, acessados e auditados pelas etapas posteriores;
- obter ou criar rótulos quando o problema for supervisionado.

Uma fonte acessível não é automaticamente uma fonte válida para ML. Dados coletados para atender uma transação podem não ter consentimento, granularidade, cobertura temporal ou qualidade adequados para treinar um modelo. A AWS recomenda validar permissões, privacidade e termos de uso especificamente para a aplicação de ML, e não assumir que o consentimento original cobre treinamento de modelos. [AWS: validate permissions and privacy](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/mlsec01-bp01.html)

## Entradas e saídas

| Entradas | Saídas |
| --- | --- |
| Fontes operacionais, eventos, arquivos, sensores, APIs externas ou dados públicos | Dados brutos preservados em armazenamento durável, sob política de ciclo de vida |
| Contrato de dados: esquema, frequência, proprietário e finalidade | Metadados catalogados e pesquisáveis |
| Política de acesso, consentimento e retenção | Evidência de origem, acesso controlado e trilha de auditoria |
| Rótulos existentes ou regras de rotulagem | Dataset rotulado, com versão e qualidade de rótulo conhecidas |

O contrato de dados deve existir antes da automação. Ele define, no mínimo, campos esperados, tipos, chave de negócio, atraso aceitável, comportamento para registros inválidos e responsável pela fonte. Sem esse contrato, uma mudança em uma API ou tabela pode degradar o dataset sem interromper a pipeline, o pior tipo de falha, porque produz um modelo aparentemente funcional com dados errados.

### Critérios de aceite da ingestão

O contrato só é útil quando a ingestão o verifica. Para cada fonte, defina limites mensuráveis e a ação quando forem violados:

| Dimensão | Exemplos de critério | Ação quando falhar |
| --- | --- | --- |
| Estrutura | Esquema, tipos, campos obrigatórios e versão esperada | Rejeitar a promoção; registrar a alteração e encaminhar o lote para quarentena. |
| Completude e integridade | Nulos máximos, chaves únicas, referências válidas e intervalos permitidos | Isolar registros inválidos e alertar o proprietário da fonte. |
| Frescor e volume | Atraso máximo, janela de chegada e intervalo esperado de registros | Alertar; impedir que um dataset atrasado seja promovido como atual. |
| Segurança | Classificação, finalidade de ML, consentimento e origem autorizada | Bloquear acesso ou processamento até a regularização. |

Quarentena não significa apagar a evidência. Preserve o registro de erro, o lote, a regra violada e a versão do contrato para investigação. Dados que falharam na aceitação podem existir na SoR com acesso restrito, mas não podem avançar para SoT, SPEC ou treinamento até serem corrigidos e reprocessados.

## Blocos de arquitetura

### Fontes de dados

Comece pelas fontes que explicam o resultado que o negócio quer prever. Para previsão de demanda, por exemplo, vendas históricas não bastam: calendário, estoque, preço e promoções podem ser necessários. Para classificação supervisionada, confirme também onde o rótulo será obtido e quanto tempo ele leva para se tornar conhecido.

Classifique cada fonte como interna ou externa, batch ou streaming, estruturada ou não estruturada, e confiável ou sujeita a validação reforçada. Dados de terceiros e conteúdo enviado por usuários exigem controles adicionais de procedência e qualidade; usá-los sem validação abre espaço para vieses, erros e *data poisoning*.

### Rotulagem

Problemas supervisionados dependem de rótulos. A rotulagem pode vir de um sistema já existente, por exemplo, uma compra confirmada ou um diagnóstico revisado, ou de revisão humana. O resultado deve registrar a versão das instruções de rotulagem, quem ou qual processo rotulou o item, a data e a confiança ou discordância quando aplicável.

Rotulagem humana exige controle de qualidade próprio: amostre itens para auditoria, mantenha um conjunto de referência revisado por especialistas e meça a concordância entre anotadores. Uma baixa concordância é um sinal de que a definição do problema ou as instruções estão ambíguas; não é um detalhe a ser escondido pela média das métricas do modelo.

Não trate o rótulo como verdade absoluta. Rótulos atrasados, ambíguos ou obtidos depois do momento que a previsão ocorreria podem introduzir *leakage* ou tornar a avaliação irrealista. Separe explicitamente a janela em que uma feature é conhecida da janela em que o rótulo é confirmado.

### Ingestão e agregação

Use **batch** quando os dados são históricos ou quando a decisão tolera atualização em intervalos definidos. Use **streaming** quando o evento precisa estar disponível em segundos ou minutos para alimentar uma decisão, uma feature online ou uma detecção operacional. A escolha deve ser motivada pelo SLA do caso de uso, não pela preferência tecnológica.

Preserve uma camada bruta, particionada e com acesso restrito antes de realizar transformações. A preparação detalhada pertence à próxima seção; nesta etapa, a prioridade é capturar dados de modo completo, idempotente e observável. Registre volume recebido, registros rejeitados, atraso, duplicidade e mudanças de esquema.

### Armazenamento, catálogo e governança

O data lake mantém dados brutos e históricos para reprocessamento. Um catálogo torna conjuntos, tabelas, esquemas e proprietários localizáveis. Bancos transacionais continuam sendo fontes de verdade para operações; data warehouses atendem análises agregadas. Não force todas as necessidades em uma única tecnologia: escolha o armazenamento conforme o padrão de acesso, latência, custo e governança.

Colete somente atributos justificáveis para o caso de uso e defina retenção e descarte desde o início. Manter dados indefinidamente “caso sejam úteis” aumenta custo, superfície de ataque e risco de privacidade. A AWS recomenda documentar a necessidade de cada atributo e automatizar retenção, redação e remoção. [AWS: keep only relevant data](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/mlsec03-bp05.html)

Imutabilidade para auditoria não significa retenção ilimitada. O fluxo deve atender solicitações de retirada de consentimento ou exclusão: localizar os registros pelo identificador do titular, revogar o acesso, remover ou anonimizar o dado conforme a política aplicável, invalidar derivados e registrar quais snapshots e modelos podem ter sido afetados. A decisão de retreinar ou descontinuar um modelo deve seguir a política de risco e a obrigação regulatória do domínio.

## Camadas de dados: SoR, SoT e SPEC

SoR, SoT e SPEC descrevem **o papel e a autoridade** de um dado; Bronze, Silver e Gold descrevem **o estágio de qualidade e transformação** no lakehouse. É comum mapeá-los lado a lado, mas o mapeamento é uma convenção de arquitetura, não uma equivalência física obrigatória. O AWS Well-Architected ML Lens não prescreve essa nomenclatura: ele apresenta a sequência *Collect Data*, *Preprocess Data* e *Engineer Features*. Use estas camadas para tornar a implementação mais clara, sem atribuí-las ao framework.

Neste guia, adotamos a seguinte convenção:

| Camada | Papel | Correspondência medallion usual | Regra para ML |
| --- | --- | --- | --- |
| **SoR, System of Record** | Espelho rastreável da origem: preserva o que o sistema produtor enviou, com proveniência e histórico. | **Bronze** | Não treine diretamente daqui. Use-a para auditoria, reprocessamento e recuperação. |
| **SoT, Source of Truth** | Dado padronizado, validado, deduplicado e reconciliado entre fontes; estabelece qual valor é aceito como verdade para cada domínio e instante. | **Silver** | É a base preferencial para criar datasets analíticos e de ML. Ainda mantém granularidade suficiente para *joins*, análise e *point-in-time correctness*. |
| **SPEC, Specialized** | Produto de dados destinado a um consumidor ou decisão específica: por exemplo, uma tabela de features para churn, fraude ou demanda. | **Gold**, com uma ressalva | É a camada de entrada do treinamento quando contém features, rótulo e recorte temporal próprios do caso de uso. Não a reduza a agregados de BI. |

Há uma ambiguidade frequente na sigla SoT: algumas equipes a usam como *Source of Truth*, outras como *System of Transformation*. Neste material, **SoT significa Source of Truth**; as transformações que a produzem pertencem à pipeline entre SoR e SoT. O termo escolhido deve constar no glossário e nos contratos de dados da organização, para não transformar uma discussão de governança em divergência de vocabulário.

O padrão medallion organiza a melhoria progressiva da qualidade dos dados: Bronze recebe o bruto, Silver aplica limpeza e validação, e Gold publica dados enriquecidos para consumo. A Databricks ressalta que é um padrão recomendado, não um requisito, e que a camada Silver pode manter o detalhe necessário a análises preditivas. [Databricks: arquitetura medallion](https://docs.databricks.com/aws/pt/lakehouse/medallion)

### Fluxo recomendado para treinamento

```text
Sistemas de origem
        │
        ▼
SoR / Bronze ── dados brutos preservados, metadados de origem e ciclo de vida definido
        │  validação, parsing, deduplicação, reconciliação
        ▼
SoT / Silver ── entidades e eventos confiáveis, detalhados e versionados
        │  recorte temporal, joins, criação de features e rótulo
        ▼
SPEC / Gold ── dataset especializado e versionado por caso de uso
        │
        ▼
Treinamento, validação e registro do modelo
```

O modelo deve consumir uma **versão imutável da SPEC**, identificada por período, código de transformação, definição de features, definição de rótulo e política de divisão treino/validação/teste. Isso permite reproduzir o treinamento e investigar uma predição sem depender do estado atual das tabelas.

Para viabilizar o controle temporal nas próximas etapas, a coleta deve preservar pelo menos o instante do evento, o instante de ingestão, a versão/origem do registro e a data de atualização. A definição detalhada de *point-in-time correctness*, *joins* temporais e divisão temporal pertence à preparação e ao feature store.

### O que cada camada não deve fazer

| Camada | Limite necessário |
| --- | --- |
| SoR / Bronze | Não corrigir o registro original em silêncio, não executar *joins* de negócio e não servir diretamente como dataset de treinamento. Registros inválidos devem ser identificados ou colocados em quarentena, mas a entrada original precisa continuar rastreável. |
| SoT / Silver | Não ser um depósito de métricas finais de um único time. Ela resolve e compartilha entidades de domínio; uma transformação específica de churn não deve redefinir a verdade corporativa de cliente. |
| SPEC / Gold | Não ser uma planilha sem dono nem uma tabela genérica chamada `gold`. Cada SPEC deve ter consumidor, finalidade, contrato, granularidade, SLA de atualização e política de versionamento explícitos. |

### Exemplo: previsão de churn

| Camada | Dataset possível | Uso |
| --- | --- | --- |
| SoR / Bronze | `crm_customer_raw`, `billing_invoice_raw`, `support_ticket_raw` | Retém a cópia recebida de CRM, faturamento e suporte, inclusive para reprocessar correções futuras. |
| SoT / Silver | `customer`, `invoice`, `support_interaction` | Padroniza chaves, resolve duplicidades, valida datas e constrói a visão confiável de cada entidade. |
| SPEC / Gold | `churn_training_snapshot_2026_01` | Materializa uma linha por cliente e data de corte, com features permitidas até o corte e o rótulo observado na janela posterior. |

## Uso das camadas no treinamento de modelos

1. Defina o evento de previsão e a data de corte antes de selecionar colunas.
2. Construa features a partir da SoT, aplicando o mesmo critério temporal para todas as fontes.
3. Publique a SPEC com um contrato claro: granularidade, chave, janela de features, janela do rótulo, esquema e proprietário.
4. Divida treino, validação e teste sem permitir que uma observação futura influencie uma anterior.
5. Versione o *snapshot* SPEC usado no experimento junto de código, parâmetros, métricas e artefato do modelo.

A camada Gold tradicional costuma conter agregados para BI. Para ML, uma SPEC pode estar nessa camada por ser um produto de dados especializado, mas normalmente precisa preservar mais detalhe e histórico do que um painel executivo. Portanto, **não use “Gold” como justificativa para agregar demais os dados do modelo**.

## Recursos AWS e por que utilizá-los

| Necessidade | Recurso AWS | Por que usar |
| --- | --- | --- |
| Camada durável de dados brutos, históricos e artefatos de coleta | [Amazon S3](https://aws.amazon.com/s3/) | Oferece armazenamento de objetos escalável para estruturar um data lake por origem, data e versão; políticas de ciclo de vida ajudam a aplicar retenção e reduzir custo. |
| Descoberta de datasets, esquemas e metadados | [AWS Glue Data Catalog](https://docs.aws.amazon.com/glue/latest/dg/catalog-and-crawler.html) | Centraliza metadados técnicos para que dados no lake sejam encontrados e consultados; crawlers podem descobrir tabelas, mas contratos e revisão de mudanças continuam necessários. |
| Transformação entre SoR, SoT e SPEC | [AWS Glue](https://docs.aws.amazon.com/glue/latest/dg/what-is-glue.html) | Executa jobs de ETL sem servidor e ajuda a tornar as transformações entre camadas repetíveis, agendáveis e auditáveis. |
| Ingestão e sincronização de bases relacionais existentes | [AWS Database Migration Service (AWS DMS)](https://docs.aws.amazon.com/dms/latest/userguide/Welcome.html) | Migra dados históricos e pode replicar alterações contínuas (CDC), reduzindo impacto sobre o banco operacional e evitando extrações manuais recorrentes. |
| Eventos e telemetria em tempo real | [Amazon Kinesis Data Streams](https://docs.aws.amazon.com/streams/latest/dev/introduction.html) | Recebe eventos ordenados por partição e permite que múltiplos consumidores processem o mesmo fluxo; é adequado quando a atualização contínua é uma necessidade do produto. |
| Entrega de streaming para destinos de armazenamento | [Amazon Data Firehose](https://docs.aws.amazon.com/firehose/latest/dev/what-is-this-service.html) | Simplifica a entrega gerenciada de eventos para destinos como S3, reduzindo a operação de consumidores próprios quando não é necessário controlar o processamento evento a evento. |
| Catálogo, permissões granulares e compartilhamento governado no data lake | [AWS Lake Formation](https://docs.aws.amazon.com/lake-formation/latest/dg/what-is-lake-formation.html) | Ajuda a aplicar governança de acesso sobre dados catalogados, importante quando times de dados, ciência de dados e aplicações não devem ter as mesmas permissões. |
| Controle de acesso, criptografia e auditoria | [AWS IAM](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html), [AWS KMS](https://docs.aws.amazon.com/kms/latest/developerguide/overview.html) e [AWS CloudTrail](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-user-guide.html) | IAM aplica o menor privilégio, KMS protege chaves de criptografia e CloudTrail registra atividades de acesso e administração para auditoria e investigação. |
| Rotulagem humana ou assistida | [Amazon SageMaker Ground Truth](https://docs.aws.amazon.com/sagemaker/latest/dg/sms.html) | Organiza trabalhos de rotulagem, auditoria e fluxos humano-no-loop quando o rótulo ainda não existe na origem. |
| Descoberta de dados sensíveis | [Amazon Macie](https://docs.aws.amazon.com/macie/latest/user/what-is-macie.html) | Ajuda a identificar dados sensíveis no S3 para que o conjunto seja minimizado, protegido ou redigido antes de alcançar ambientes de ML. |

Esses serviços não formam um pacote obrigatório. Um projeto didático ou uma carga batch pequena pode começar com S3, Glue Data Catalog e um processo agendado. Kinesis só se justifica quando o requisito realmente é contínuo; Ground Truth só se justifica quando a criação de rótulos é parte do problema; Lake Formation se torna valioso quando a governança entre domínios e contas deixa de caber em permissões simples.

## Decisões de arquitetura

| Decisão | Direção recomendada |
| --- | --- |
| Batch ou streaming? | Escolha batch como padrão quando a previsão não requer atualização imediata. Adote streaming somente quando o SLA de negócio exigir baixa latência. |
| Copiar ou consultar a fonte? | Preserve uma cópia histórica e versionada para treinamento reproduzível; consultar apenas o sistema operacional impede reproduzir o dataset usado por um modelo. |
| Em que camada treinar? | Construa o dataset de treino a partir da SoT e publique uma SPEC versionada por caso de uso. Nunca trate a SoR/Bronze como fonte direta de features. |
| Dados completos ou minimizados? | Colete os atributos necessários e justificáveis. Mais colunas não compensam ausência de qualidade, permissão ou significado causal. |
| Schema rígido ou flexível? | Aceite evolução controlada, mas alerte e bloqueie mudanças incompatíveis antes que contaminem o dataset de treino. |
| Rótulo automático ou humano? | Use rótulo automático quando há evento de negócio confiável; use revisão humana quando a semântica exige julgamento e meça a concordância entre revisores. |

## Anti-padrões

- Treinar diretamente de uma tabela de produção sem preservar a versão consultada.
- Treinar diretamente da SoR/Bronze porque ela “tem todos os dados”.
- Chamar qualquer tabela agregada de Gold/SPEC sem definir seu consumidor, granularidade, dono e contrato.
- Usar atributos registrados após a data de corte da previsão.
- Usar dados pessoais porque estão disponíveis, sem validar finalidade, consentimento e retenção.
- Construir streaming para uma previsão diária.
- Permitir mudanças de esquema sem monitoramento.
- Misturar dados de treino, validação e produção no mesmo local e com as mesmas permissões.
- Aceitar fontes externas ou conteúdo de usuários sem checar procedência, distribuição e anomalias.

## Relação com este repositório

Os notebooks usam datasets locais providos pelo scikit-learn; isso é adequado para demonstrar algoritmos, mas não representa a coleta de um produto real. Em uma evolução do exemplo de classificação de câncer, a coleta precisaria definir explicitamente quais dados clínicos são autorizados, qual evento confirma o rótulo, como dados identificáveis são protegidos e quem pode acessar cada ambiente. O arquivo `model.pickle` só é reproduzível de forma robusta quando também é possível identificar a versão exata do dataset e da coleta que o gerou.

## Checklist de saída

- [ ] Cada fonte tem proprietário, finalidade de ML, permissão de uso e contrato de dados documentados.
- [ ] Dados brutos são preservados em local durável, com acesso mínimo necessário e retenção definida.
- [ ] A ingestão mede volume, atraso, duplicidades, falhas e alterações de esquema.
- [ ] Existem critérios de aceite automatizados, quarentena e responsáveis pela correção de violações.
- [ ] O dataset de treinamento tem origem e versão rastreáveis.
- [ ] Rótulos têm definição, origem, janela temporal, amostragem de auditoria e qualidade documentadas.
- [ ] Dados sensíveis são minimizados, protegidos e separados dos ambientes que não precisam deles.
- [ ] A política de retenção inclui o fluxo para retirada de consentimento, exclusão ou anonimização e análise de impacto em derivados.

## Correspondência com o diagrama de arquitetura

Esta seção corresponde nominalmente ao bloco **“Collect Data”**, dentro da fase **“Process Data”** do diagrama. No fluxo visual, ele é o primeiro bloco à esquerda e antecede **“Preprocess Data”** e **“Engineer Features”**.

| Elemento do diagrama | Como esta seção o resolve | Limite da seção |
| --- | --- | --- |
| **Collect Data** | Define fontes, rotulagem, ingestão batch ou streaming, armazenamento bruto, catálogo, permissões, retenção e proveniência. O resultado é a SoR/Bronze: cópia rastreável do que foi recebido. | Não limpa, deduplica nem cria features de negócio. |
| **Process Data** | Estabelece a fronteira de entrada da fase: dados confiáveis e versionados podem seguir para tratamento. | A implementação de transformação fica nas seções de preparação e engenharia de features. |
| **Preprocess Data** | A coleta entrega dados brutos, contrato de dados e registros inválidos/quarentenados para que essa próxima etapa padronize, valide e reconcilie os dados na SoT/Silver. | Não é executada na coleta. |
| **Engineer Features** | A coleta preserva a origem e o tempo de cada atributo, condição necessária para features reproduzíveis e sem *leakage*. | A criação e o armazenamento das features serão detalhados nas seções seguintes. |
| **“Store Artifacts (Data, Code, Model)”** | A versão da SoR e o *snapshot* da SPEC devem ser registrados como artefatos de dados, ligados ao código de transformação e, depois, ao modelo treinado. | O registro de modelos e a linhagem completa pertencem à seção de lineage. |
| **Performance feedback loop** e **Model Drift / Active Learning feedback loop** | Alertas de baixa qualidade, drift ou necessidade de novos rótulos retornam à coleta como revisão de fonte, contrato, frequência ou estratégia de rotulagem. | Monitoramento e alarmes são tratados em suas seções próprias. |

Em termos de camadas, a resolução é deliberadamente transversal: **Collect Data produz a SoR/Bronze; Preprocess Data consolida a SoT/Silver; Engineer Features publica a SPEC consumida pelo treinamento**. Essa separação preserva a fidelidade ao diagrama e evita atribuir à coleta responsabilidades que pertencem às etapas seguintes.

## Referências

- [AWS Well-Architected Machine Learning Lens, Data collection](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/data-collection.html)
- [AWS Well-Architected Machine Learning Lens, ML lifecycle architecture diagram](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/architecture-diagram.html)
- [AWS Well-Architected Machine Learning Lens, Validate ML data permissions, privacy, software, and license terms](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/mlsec01-bp01.html)
- [AWS Well-Architected Machine Learning Lens, Keep only relevant data](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/mlsec03-bp05.html)
- [Databricks, O que é a arquitetura medallion lakehouse?](https://docs.databricks.com/aws/pt/lakehouse/medallion)
