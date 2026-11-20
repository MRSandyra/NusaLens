"""Download the Hugging Face models used by retrieval before a long ingest/index run.

Ollama models are fetched separately by ``make models`` because they are managed by Ollama, not
the Python cache.  This script deliberately does not create a serving claim or benchmark output.
"""
from sentence_transformers import CrossEncoder, SentenceTransformer

from nusalens.config import settings


def main() -> None:
    for label, loader, model_name in (
        ("embedding", SentenceTransformer, settings.embedding_model),
        ("reranker", CrossEncoder, settings.reranker_model),
    ):
        print(f"Downloading/verifying {label} model: {model_name}")
        loader(model_name, device="cpu")
    print("Python retrieval models are cached.")


if __name__ == "__main__":
    main()
