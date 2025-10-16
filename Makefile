VENV_BIN ?= venv/bin
PYTHON ?= $(VENV_BIN)/python

.PHONY: ingest chat clear-db

ingest:
	@$(PYTHON) src/ingest.py $(ARGS)

chat:
	@$(PYTHON) src/chat.py $(ARGS)

clear-db:
	@docker compose exec postgres psql -U postgres -d rag -c "DELETE FROM langchain_pg_collection WHERE name = '$${PG_VECTOR_COLLECTION_NAME:-documents}';" >/dev/null || true
	@echo "Coleção '$${PG_VECTOR_COLLECTION_NAME:-documents}' removida (ou já inexistente). Execute a ingestão novamente para recriar."
