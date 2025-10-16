# Desafio MBA Engenharia de Software com IA - Full Cycle

Este repositório contém a solução do desafio de ingestão e busca semântica utilizando LangChain com PostgreSQL e pgVector. Siga as instruções abaixo para preparar o ambiente, configurar as dependências, executar a ingestão do PDF e interagir com o chat via linha de comando.

## Configuração do Ambiente

Antes de iniciar, instale o Python (foi utilizado Python 3.14.0 durante o desenvolvimento) e o Docker Desktop ou compatível.

1. **Criar e ativar o ambiente virtual (`venv`):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

2. **Instalar as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

   Esse comando assegura que todos os pacotes necessários (LangChain, conectores de banco, embeddings, etc.) sejam instalados.

## Variáveis de Ambiente

1. **Criar o arquivo `.env`:**

   ```bash
   cp .env.example .env
   ```

2. **Preencher os valores obrigatórios:**

   - **Conexão com o Postgres**
     - `PGVECTOR_URL` ou `DATABASE_URL` – string de conexão com o banco (ex.: `postgresql+psycopg://postgres:postgres@localhost:5432/rag`).
     - `PGVECTOR_COLLECTION` ou `PG_VECTOR_COLLECTION_NAME` – nome da coleção onde os vetores serão armazenados (padrão `documents`).
   - **Ingestão (embeddings)**
     - `EMBEDDINGS_PROVIDER` – `openai`, `google` ou `fake`. Caso não informado o script tentará detectar automaticamente.
     - `OPENAI_API_KEY` e `OPENAI_EMBEDDING_MODEL` – chave e modelo de embedding da OpenAI (padrão `text-embedding-3-small`).
     - `GOOGLE_API_KEY` e `GOOGLE_EMBEDDING_MODEL` – chave e modelo de embedding Gemini (padrão `models/embedding-001`).
   - **Chat (LLM)**
     - `LLM_PROVIDER` – `openai`, `google` ou `fake`, com fallback automático se a chave preferida estiver ausente.
     - `OPENAI_CHAT_MODEL` – modelo conversacional OpenAI (padrão `gpt-5-nano`).
     - `GOOGLE_CHAT_MODEL` – modelo conversacional Gemini (padrão `gemini-2.5-flash-lite`).
   - **Documento**
     - `PDF_PATH` – caminho absoluto ou relativo do PDF que será ingerido (padrão `./document.pdf`).

### Criando uma API Key na OpenAI

1. Acesse [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys).
2. Faça login ou crie uma conta.
3. Clique em **API Keys** no menu lateral.
4. Clique em **Create new secret key**, nomeie a chave e confirme.
5. Copie a chave exibida uma única vez e coloque-a na variável `OPENAI_API_KEY` dentro do `.env`.

### Criando uma API Key no Google Gemini

1. Acesse [https://ai.google.dev/gemini-api/docs/api-key?hl=pt-BR](https://ai.google.dev/gemini-api/docs/api-key?hl=pt-BR).
2. Faça login com sua conta Google.
3. Abra a seção **API Keys**.
4. Clique em **Create API Key**, nomeie e confirme.
5. Copie a chave gerada e preencha a variável `GOOGLE_API_KEY` no `.env`.

Documentação de referência: [Como usar chaves da API Gemini](https://ai.google.dev/gemini-api/docs/api-key?hl=pt-BR)

## Banco de Dados Vetorial

O repositório inclui um `docker-compose.yml` que sobe um Postgres 17 com a extensão pgVector.

```bash
docker compose up -d
```

Após a subida, o serviço auxiliar `bootstrap_vector_ext` cria a extensão `vector` automaticamente. Caso deseje acompanhar os logs:

```bash
docker compose logs -f bootstrap_vector_ext
```

Para desligar os serviços ao final:

```bash
docker compose down
```

## Ingestão do PDF

Com o ambiente virtual ativado e o banco em execução:

```bash
python src/ingest.py
# ou(make ingest)
make ingest
```

O script:

- Carrega o PDF apontado por `PDF_PATH`.
- Fragmenta o conteúdo em chunks de 1.000 caracteres com overlap de 150.
- Gera embeddings utilizando os modelos configurados.
- Persiste os vetores no Postgres usando pgVector.

Execute a ingestão sempre que trocar o PDF ou ajustar a forma de chunking.

## Execução da Busca Semântica (CLI)

Após a ingestão:

```bash
python src/chat.py
# ou
make chat
# sobrescreva argumentos, por exemplo:
make chat ARGS="--embedding-provider fake --llm-provider fake"
```

O chat:

- Recebe perguntas via terminal.
- Vetoriza a consulta e busca os 10 chunks mais relevantes no banco.
- Monta o prompt com base no template definido em `src/search.py`.
- Retorna respostas baseadas exclusivamente no conteúdo do PDF ingerido.

Perguntas fora do contexto resultarão em: `"Não tenho informações necessárias para responder sua pergunta."`

> Para testar offline (sem chamadas externas) utilize `--embedding-provider fake --llm-provider fake`, o que ativa implementações locais de teste.

### Busca direta e testes rápidos

Para inspecionar os resultados retornados pelo vetor store sem abrir o chat, utilize:

```bash
python src/search.py --query "pergunta de validação" --k 3
```

Opcionalmente, force o provedor de embeddings com `--provider` (`openai`, `google` ou `fake`).

### Limpando a coleção

Se precisar reiniciar o banco vetorial, execute:

```bash
make clear-db
```

O alvo remove a coleção configurada em `PGVECTOR_COLLECTION` (ou `documents`, por padrão) usando o Postgres via Docker. Execute `make ingest` em seguida para recriar os registros.

### Exemplos com providers OpenAI e Google

Use as variáveis de ambiente para direcionar os comandos do `make` para o provedor desejado.

```bash
# Fluxo completo usando OpenAI
make clear-db
EMBEDDINGS_PROVIDER=openai make ingest
make chat ARGS="--embedding-provider openai --llm-provider openai"

# Fluxo completo usando Google
make clear-db
EMBEDDINGS_PROVIDER=google make ingest
make chat ARGS="--embedding-provider google --llm-provider google"
```

## Estrutura do Projeto

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── document.pdf
├── src/
│   ├── ingest.py
│   ├── search.py
│   └── chat.py
├── Makefile
└── docs/
    ├── desafio.md
    └── execucao-hoje.md
```

## Execução com Python 3.14.0

Se estiver utilizando o Python 3.14.0 instalado via Homebrew, observe:

- Crie um ambiente dedicado com `python3.14 -m venv venv314` para contornar a proteção PEP 668.
- Instale as dependências com `venv314/bin/pip install -r requirements.txt`; a compilação de pacotes como `grpcio` pode levar alguns minutos.
- Execute os scripts apontando para o Python desse ambiente, por exemplo: `venv314/bin/python src/ingest.py`.
- Caso ajuste o `requirements.txt`, mantenha versões compatíveis com o Python 3.14 (ex.: `psycopg-binary==3.2.10`, `tiktoken==0.12.0`, `numpy==2.3.3`).

Com esses passos você terá o pipeline completo de ingestão e busca semântica em funcionamento localmente.
