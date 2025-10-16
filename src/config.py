import logging
import os
import socket
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain_openai import OpenAIEmbeddings

try:  # Optional dependency: only required when using Google embeddings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:  # pragma: no cover - optional feature
    GoogleGenerativeAIEmbeddings = None


load_dotenv()
LOGGER = logging.getLogger("config")


def get_env_value(name: str, required: bool = True) -> str:
    value = os.getenv(name)
    if value:
        return value
    if required:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return ""


def resolve_backend(explicit: str | None = None) -> str:
    choice = (explicit or os.getenv("EMBEDDINGS_PROVIDER") or "").strip().lower()
    if not choice:
        if os.getenv("OPENAI_API_KEY"):
            choice = "openai"
        elif os.getenv("GOOGLE_API_KEY"):
            choice = "google"
        else:
            choice = "fake"
    if choice not in {"openai", "google", "fake"}:
        raise ValueError("Supported embedding backends: 'openai', 'google' or 'fake'")
    return choice


def resolve_embeddings(explicit_backend: str | None = None) -> tuple[str, Embeddings]:
    backend = resolve_backend(explicit_backend)
    if backend == "openai":
        api_key = get_env_value("OPENAI_API_KEY")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return backend, OpenAIEmbeddings(model=model, api_key=api_key)

    if backend == "google":
        if GoogleGenerativeAIEmbeddings is None:
            raise RuntimeError(
                "Google embeddings requested but 'langchain-google-genai' is not installed"
            )
        api_key = get_env_value("GOOGLE_API_KEY")
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return backend, GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

    size = 1536
    return backend, FakeEmbeddings(size=size)


def resolve_pgvector_url() -> str:
    connection = get_env_value("DATABASE_URL")
    parsed = urlsplit(connection)
    if parsed.hostname:
        try:
            socket.gethostbyname(parsed.hostname)
        except OSError as exc:
            LOGGER.warning("Hostname %s could not be resolved (%s)", parsed.hostname, exc)
            raise
    return urlunsplit(parsed)


def resolve_collection_name() -> str:
    return get_env_value("PG_VECTOR_COLLECTION_NAME")
