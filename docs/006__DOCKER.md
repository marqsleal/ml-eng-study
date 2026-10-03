# Docker: empacotando a API de classificação

Docker é uma ferramenta para empacotar uma aplicação e tudo de que ela precisa para executar — sistema base, runtime, bibliotecas e arquivos da aplicação — em uma **imagem**. A imagem é usada para criar um **container**, uma instância isolada e executável da aplicação.

Isso reduz diferenças entre máquinas: a mesma imagem pode ser executada localmente, em um servidor ou em uma plataforma de containers. Docker não substitui testes, monitoramento ou práticas de segurança; ele torna o ambiente de execução reproduzível.

## Conceitos essenciais

| Conceito | Significado |
| --- | --- |
| Imagem | Modelo imutável com os arquivos e instruções necessários para executar a aplicação. |
| Container | Instância em execução de uma imagem. Pode ser iniciada, parada e removida. |
| Dockerfile | Receita declarativa usada para construir uma imagem. |
| Registry | Serviço que armazena imagens, como Docker Hub ou um registry privado. |
| Porta publicada | Mapeamento entre uma porta da máquina hospedeira e uma porta do container. |

## Comandos básicos

| Comando | Uso |
| --- | --- |
| `docker --version` | Exibe a versão instalada. |
| `docker images` | Lista as imagens locais. |
| `docker ps` | Lista containers em execução. Use `docker ps -a` para incluir os parados. |
| `docker build -t nome:tag diretorio` | Constrói uma imagem a partir de um Dockerfile. |
| `docker run imagem` | Cria e inicia um container. |
| `docker logs <container>` | Exibe os logs de um container. |
| `docker exec -it <container> sh` | Abre um shell dentro de um container em execução. |
| `docker stop <container>` | Para um container. |
| `docker rm <container>` | Remove um container parado. |
| `docker rmi <imagem>` | Remove uma imagem local que não esteja em uso. |

Os identificadores abreviados retornados por `docker ps` podem ser usados no lugar de `<container>`. A opção `--rm` em `docker run` remove o container automaticamente quando ele para, o que é útil para testes locais.

## Dockerfile da API

A API criada em `src/main.py` carrega o pipeline salvo em `src/model.pickle` e é iniciada pelo Uvicorn. O arquivo `src/Dockerfile` transforma esses itens em uma imagem:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py model.pickle ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Cada instrução tem uma função:

- `FROM` escolhe a imagem-base com Python.
- `WORKDIR` define `/app` como diretório de trabalho no container.
- O primeiro `COPY` e `RUN` instalam as dependências listadas em `src/requirements.txt`. Copiá-las antes do código aproveita o cache de camadas do Docker quando apenas a aplicação muda.
- O segundo `COPY` adiciona a API e o pipeline treinado à imagem.
- `EXPOSE 8000` documenta a porta que a aplicação usa.
- `CMD` define o processo padrão do container. O host `0.0.0.0` permite receber conexões externas ao container.

`EXPOSE` não publica a porta por si só: a publicação é feita com `-p` ao executar o container.

## Construir e executar

Execute os comandos a partir da raiz do repositório:

```bash
docker build -t cancer-api:1.0 src
docker run --rm --name cancer-api -p 8000:8000 cancer-api:1.0
```

O argumento final de `docker build` é o **contexto de build**. Neste caso é `src`, portanto o Dockerfile consegue copiar `main.py`, `model.pickle` e `requirements.txt` sem incluir notebooks ou outros arquivos do projeto.

Com o container ativo, verifique a API e envie os casos de exemplo:

```bash
bash scripts/001__health.sh
bash scripts/002__predict_benign.sh
bash scripts/003__predict_malignant.sh
```

Para inspecionar a execução em outro terminal:

```bash
docker ps
docker logs cancer-api
```

Abra `http://127.0.0.1:8000/docs` para consultar e testar os endpoints pela documentação interativa.

## Boas práticas iniciais

- Fixe versões em `src/requirements.txt` para reduzir variações entre builds.
- Use uma imagem-base pequena e adequada ao runtime, como `python:3.13-slim`.
- Não inclua segredos, chaves ou arquivos `.env` na imagem. Passe configurações sensíveis por variáveis de ambiente ou um gerenciador de segredos.
- Crie um `.dockerignore` quando o contexto de build puder conter artefatos desnecessários, como ambientes virtuais, caches e arquivos de dados grandes.
- Em produção, use uma tag versionada em vez de depender de `latest` e registre qual imagem foi implantada.

Para esta API didática, o modelo é empacotado diretamente na imagem. Em sistemas reais, defina uma estratégia explícita de versionamento e atualização do modelo antes da implantação.
