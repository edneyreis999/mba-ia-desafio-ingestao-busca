import argparse
import logging
from typing import List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_postgres import PGVector

from config import resolve_collection_name, resolve_embeddings, resolve_pgvector_url


LOGGER = logging.getLogger("search")
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
""".strip()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def search_similar_chunks(
    query: str,
    k: int = 10,
    *,
    provider: Optional[str] = None,
) -> Tuple[List[Tuple[Document, float]], str, str]:
    if not query:
        raise ValueError("Query must be a non-empty string")
    backend, embeddings = resolve_embeddings(provider)
    store = PGVector(
        embeddings=embeddings,
        collection_name=resolve_collection_name(),
        connection=resolve_pgvector_url(),
        use_jsonb=True,
    )
    results = store.similarity_search_with_score(query, k=k)
    context = "\n\n".join(
        doc.page_content.strip() for doc, _ in results if doc.page_content.strip()
    )
    return results, context, backend


def build_prompt(question: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(contexto=context or "N/A", pergunta=question)


def search_prompt(
    question: str,
    *,
    k: int = 10,
    provider: Optional[str] = None,
) -> Tuple[str, List[Tuple[Document, float]], str]:
    results, context, backend = search_similar_chunks(question, k=k, provider=provider)
    prompt = build_prompt(question, context)
    return prompt, results, backend


def format_cli_results(results: Sequence[Tuple[Document, float]]) -> str:
    if not results:
        return "Nenhum resultado encontrado."
    lines: list[str] = []
    for index, (doc, score) in enumerate(results, start=1):
        lines.append("=" * 80)
        lines.append(f"Resultado {index} | score={score:.4f}")
        lines.append("-" * 80)
        lines.append(doc.page_content.strip())
        if doc.metadata:
            lines.append("\nMetadados:")
            lines.extend(f"- {key}: {value}" for key, value in doc.metadata.items())
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute uma busca semântica no PGVector.")
    parser.add_argument("--query", "-q", required=True, help="Pergunta ou termo de busca.")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Quantidade de chunks mais similares que devem ser retornados (default: 5).",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "google", "fake"],
        help="Força o provedor de embeddings a ser utilizado.",
    )
    parser.add_argument("--verbose", action="store_true", help="Ativa logging detalhado.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    try:
        results, _, backend = search_similar_chunks(
            args.query, k=args.k, provider=args.provider
        )
    except Exception as exc:  # pragma: no cover - CLI feedback
        LOGGER.error("Falha ao executar busca: %s", exc)
        raise SystemExit(1) from exc

    LOGGER.info("Busca executada com embeddings %s", backend)
    print(format_cli_results(results))


if __name__ == "__main__":
    main()
