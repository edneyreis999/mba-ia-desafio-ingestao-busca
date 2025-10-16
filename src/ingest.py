import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    resolve_collection_name,
    resolve_embeddings,
    resolve_pgvector_url,
)


load_dotenv()
LOGGER = logging.getLogger("ingest")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def resolve_pdf_path(override: str | None = None) -> Path:
    candidate = override or os.getenv("PDF_PATH") or "document.pdf"
    pdf_path = Path(candidate).expanduser()
    if not pdf_path.is_absolute():
        pdf_path = Path.cwd() / pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    return pdf_path


def sanitize_metadata(documents: Iterable[Document]) -> list[Document]:
    sanitized: list[Document] = []
    for item in documents:
        metadata = {k: v for k, v in item.metadata.items() if v not in ("", None)}
        sanitized.append(Document(page_content=item.page_content, metadata=metadata))
    return sanitized


def load_documents(pdf_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    if not splits:
        raise RuntimeError("No chunks were generated from the provided PDF")
    return sanitize_metadata(splits)


async def ingest_documents(
    documents: Sequence[Document],
    embeddings: Embeddings,
    collection_name: str,
    connection_url: str,
    *,
    reset_collection: bool,
) -> int:
    store = PGVector(
        embeddings=embeddings,
        connection=connection_url,
        collection_name=collection_name,
        use_jsonb=True,
    )

    if reset_collection:
        try:
            await asyncio.to_thread(store.delete_collection)
            LOGGER.debug("Collection %s removed before re-creating", collection_name)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Unable to drop collection %s: %s", collection_name, exc)

    try:
        await asyncio.to_thread(store.create_collection)
    except Exception as exc:
        if "already exists" not in str(exc).lower():
            raise
        LOGGER.debug("Collection %s already exists; reusing", collection_name)

    ids = [f"doc-{i}" for i in range(len(documents))]
    await asyncio.to_thread(store.add_documents, list(documents), ids=ids)
    return len(ids)


async def run_ingestion(pdf_path: Path, backend: str, reset_collection: bool) -> int:
    documents = load_documents(pdf_path)
    LOGGER.info("Loaded %s chunks from %s", len(documents), pdf_path)

    backend, embeddings = resolve_embeddings(backend)
    connection_url = resolve_pgvector_url()
    collection_name = resolve_collection_name()

    LOGGER.info(
        "Starting ingestion using %s embeddings into collection '%s'",
        backend,
        collection_name,
    )
    ingested = await ingest_documents(
        documents,
        embeddings,
        collection_name,
        connection_url,
        reset_collection=reset_collection,
    )
    LOGGER.info("Ingested %s documents into PGVector", ingested)
    return ingested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a PDF into the pgVector collection.")
    parser.add_argument("--pdf-path", help="Override PDF path; defaults to $PDF_PATH or document.pdf")
    parser.add_argument(
        "--provider",
        choices=["openai", "google"],
        help="Embedding provider to use. Defaults to environment discovery.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the existing collection instead of recreating it.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")
    return parser


def ingest_pdf(args: argparse.Namespace | None = None) -> int:
    args = args or build_parser().parse_args()
    configure_logging(verbose=args.verbose)
    pdf_path = resolve_pdf_path(args.pdf_path)
    backend = args.provider
    reset_collection = not args.append
    return asyncio.run(run_ingestion(pdf_path, backend, reset_collection))


if __name__ == "__main__":
    ingest_pdf()
