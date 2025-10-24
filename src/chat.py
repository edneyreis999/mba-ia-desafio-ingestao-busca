import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from config import resolve_backend as resolve_embedding_backend
from search import build_prompt, search_similar_chunks

try:  # Optional dependency
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional feature
    ChatGoogleGenerativeAI = None


LOGGER = logging.getLogger("chat")
DEFAULT_OUT_OF_CONTEXT = "Não tenho informações necessárias para responder sua pergunta."
OPENAI_DEFAULT_CHAT_MODEL = "gpt-5-nano"
GOOGLE_DEFAULT_CHAT_MODEL = "gemini-2.5-flash-lite"


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@dataclass
class ChatBackend:
    provider: str
    model: str
    client: object

    def invoke(self, prompt: str) -> AIMessage:
        response = self.client.invoke(prompt)
        if isinstance(response, AIMessage):
            return response
        if hasattr(response, "content"):
            return AIMessage(content=str(response.content))
        return AIMessage(content=str(response))


class _LocalEchoLLM:
    def __init__(self, model_name: str = "fake-local") -> None:
        self.model = model_name

    def invoke(self, prompt: str) -> AIMessage:
        context_section = prompt.split("CONTEXTO:", 1)[-1].split("REGRAS:", 1)[0].strip()
        if context_section in ("", "N/A"):
            return AIMessage(content=DEFAULT_OUT_OF_CONTEXT)
        _, _, tail = prompt.partition("PERGUNTA DO USUÁRIO:")
        question = tail.strip().splitlines()[0] if tail else ""
        preview = context_section[:180].replace("\n", " ")
        message = f"(Simulação offline) {question or 'Pergunta não informada.'} | Contexto: {preview}..."
        return AIMessage(content=message.strip())


def resolve_llm(
    preferred_provider: Optional[str] = None,
    model_override: Optional[str] = None,
) -> ChatBackend:
    candidates = []
    if preferred_provider:
        candidates.append(preferred_provider.lower())
    else:
        candidates.extend(
            [os.getenv("LLM_PROVIDER", "").strip().lower() or "", "openai", "google", "fake"]
        )

    normalized = [c for c in candidates if c]
    normalized = normalized or ["openai", "google", "fake"]

    errors: dict[str, str] = {}
    for provider in normalized:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                errors[provider] = "OPENAI_API_KEY ausente"
                continue
            model = model_override or OPENAI_DEFAULT_CHAT_MODEL
            client = ChatOpenAI(model=model, api_key=api_key, temperature=0)
            return ChatBackend(provider="openai", model=model, client=client)

        if provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                errors[provider] = "GOOGLE_API_KEY ausente"
                continue
            if ChatGoogleGenerativeAI is None:
                errors[provider] = "pacote langchain-google-genai não instalado"
                continue
            model = model_override or GOOGLE_DEFAULT_CHAT_MODEL
            client = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
            return ChatBackend(provider="google", model=model, client=client)

        if provider == "fake":
            model = model_override or "fake-local"
            return ChatBackend(provider="fake", model=model, client=_LocalEchoLLM(model))

        errors[provider] = "provedor desconhecido"

    details = "; ".join(f"{prov}: {msg}" for prov, msg in errors.items())
    raise RuntimeError(f"Não foi possível inicializar um LLM ({details}).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI para conversar com o assistente RAG.")
    parser.add_argument("--k", type=int, default=10, help="Quantidade de chunks recuperados.")
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "google", "fake"],
        help="Sobrescreve o provedor de embeddings.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "google", "fake"],
        help="Sobrescreve o provedor do modelo de linguagem.",
    )
    parser.add_argument("--llm-model", help="Modelo específico do provedor escolhido.")
    parser.add_argument("--verbose", action="store_true", help="Ativa logs detalhados.")
    return parser


def handle_question(
    question: str,
    *,
    k: int,
    embedding_provider: Optional[str],
    llm_backend: ChatBackend,
) -> str:
    results, context, embedding_backend = search_similar_chunks(
        question, k=k, provider=embedding_provider
    )
    LOGGER.debug(
        "Recuperados %s chunks com backend %s para a pergunta: %s",
        len(results),
        embedding_backend,
        question,
    )
    if not context:
        return DEFAULT_OUT_OF_CONTEXT

    prompt = build_prompt(question, context)
    response = llm_backend.invoke(prompt)
    return response.content.strip()


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    embedding_provider = args.embedding_provider or resolve_embedding_backend(None)

    try:
        llm_backend = resolve_llm(args.llm_provider, args.llm_model)
    except Exception as exc:  # pragma: no cover - CLI feedback
        LOGGER.error("Falha ao inicializar LLM: %s", exc)
        raise SystemExit(1) from exc

    print("Assistente iniciado. Digite sua pergunta ou '/exit' para sair.")
    while True:
        try:
            user_input = input("Você> ").strip()
        except (EOFError, KeyboardInterrupt):  # pragma: no cover - CLI feedback
            print("\nEncerrando. Até logo!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "sair", "exit", "quit"}:
            print("Encerrando. Até logo!")
            break

        try:
            answer = handle_question(
                user_input,
                k=args.k,
                embedding_provider=embedding_provider,
                llm_backend=llm_backend,
            )
        except Exception as exc:  # pragma: no cover - CLI feedback
            LOGGER.error("Erro ao processar pergunta: %s", exc)
            print("Assistente> Ocorreu um erro ao processar sua pergunta.")
            continue

        print(f"Assistente> {answer}")


if __name__ == "__main__":
    main()
