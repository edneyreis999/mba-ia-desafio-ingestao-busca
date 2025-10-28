# Carta de Correção — Constatação sobre o top‑k com Gemini

Ao realizar o desafio de ingestão e busca (RAG), constatei um comportamento específico quando utilizo embeddings do Google em conjunto com o modelo Gemini:

- Com `--k=10` (valor padrão em `src/chat.py:116`), o Gemini não encontra a resposta correta em perguntas que a OpenAI responde normalmente.
- Ao aumentar para `--k=20`, o Gemini passa a responder corretamente as mesmas perguntas.

Observação no código:

- A CLI define `k` em `src/chat.py:116` via `parser.add_argument("--k", type=int, default=10, help="Quantidade de chunks recuperados.")`.

Como reproduzir (sempre casando LLM e embeddings e limpando a coleção antes de cada teste):

- Google (embeddings Google + LLM Gemini):
  - `make clear-db`
  - `python src/ingest.py --provider google`
  - `python src/chat.py --embedding-provider google --llm-provider google --k 10` → não encontra a resposta
  - `python src/chat.py --embedding-provider google --llm-provider google --k 20` → passa a responder corretamente

- OpenAI (embeddings OpenAI + LLM OpenAI):
  - `make clear-db`
  - `python src/ingest.py --provider openai`
  - `python src/chat.py --embedding-provider openai --llm-provider openai --k 10` → responde corretamente

Hipótese técnica:

- A distribuição das similaridades produzidas pelos embeddings do Google (ex.: `models/embedding-001`) exige um `top‑k` maior para manter o recall suficiente dos trechos relevantes no contexto.
- Como o pipeline atual não faz reranking adicional nem aplica corte por limiar de score, um `k=10` pode omitir partes críticas; com `k=20` o contexto recuperado fica abrangente o bastante para o Gemini produzir a resposta.

Resumo da constatação:

- OpenAI: responde com `k=10`.
- Gemini (com embeddings do Google): precisa de `k=20` para recuperar contexto suficiente e responder corretamente.
