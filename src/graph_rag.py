import os
from pathlib import Path
import numpy as np
import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
from llama_index.core import (
    PropertyGraphIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.core.indices.property_graph import ImplicitPathExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORAGE = REPO_ROOT / "graph_storage"
DEFAULT_ARTICLES = REPO_ROOT / "articles"

load_dotenv(REPO_ROOT / ".env")


def setup_settings(embed_model_name: str = "BAAI/bge-m3"):
    device = os.environ.get("EMBED_DEVICE")
    if not device:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name, device=device)
    print(f"[graph_rag] embed device: {device}", flush=True)

    # Явный chunk size — защита от изменений default в LlamaIndex
    from llama_index.core.node_parser import SentenceSplitter
    Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

    api_key = os.environ["YANDEX_CLOUD_API_KEY"]
    folder = os.environ["YANDEX_CLOUD_FOLDER"]
    model = os.environ.get("AI4SCIENCE_GRAPH_MODEL", "qwen3.6-35b-a3b/latest")

    Settings.llm = OpenAILike(
        model=f"gpt://{folder}/{model}",
        api_base="https://llm.api.cloud.yandex.net/v1",
        api_key=api_key,
        is_chat_model=True,
        temperature=0.2,
        max_tokens=8192,
        timeout=300,
    )

def _load_pdf_documents(articles_dir: Path):
    """Clean per-page text extraction via pypdf.

    The llama-index default PDF reader was dumping raw PDF object streams
    (endstream/endobj/compressed bytes) instead of rendered text, producing
    ~70x text bloat (4.8M chars vs 67K for a 16-page article) and ~19k junk
    chunks. pypdf extracts the real text, so we build Documents ourselves.
    """
    from pypdf import PdfReader

    documents = []
    pdfs = sorted(p for p in Path(articles_dir).glob("*.pdf"))
    for pdf in pdfs:
        try:
            reader = PdfReader(str(pdf))
        except Exception as e:
            print(f"[graph_rag] SKIP {pdf.name}: {e}", flush=True)
            continue
        n_pages = len(reader.pages)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(Document(
                text=text,
                metadata={"file_name": pdf.name, "page_label": str(i + 1)},
            ))
        print(f"[graph_rag] {pdf.name}: {n_pages} pages", flush=True)
    total = sum(len(d.text) for d in documents)
    print(f"[graph_rag] loaded {len(documents)} page-docs, {total} chars", flush=True)
    return documents


def build_or_load_graph(
    articles_dir: Path = DEFAULT_ARTICLES,
    storage_dir: Path = DEFAULT_STORAGE,
    force_rebuild: bool = False,
) -> PropertyGraphIndex:
    setup_settings()
    storage_dir = Path(storage_dir)
    articles_dir = Path(articles_dir)

    if storage_dir.exists() and not force_rebuild:
        storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
        return load_index_from_storage(storage_context)

    articles_dir.mkdir(parents=True, exist_ok=True)
    documents = _load_pdf_documents(articles_dir)
    if not documents:
        raise RuntimeError(f"No supported files in {articles_dir}")

    index = PropertyGraphIndex.from_documents(
        documents,
        embed_model=Settings.embed_model,
        kg_extractors=[ImplicitPathExtractor()],
        show_progress=True,
    )
    index.storage_context.persist(persist_dir=str(storage_dir))
    return index


_CACHE = {}


def _get_chunks(index):
    if "chunks" in _CACHE:
        return _CACHE["chunks"], _CACHE["E"]
    chunks = [n for n in index.property_graph_store.graph.nodes.values()
              if type(n).__name__ == "ChunkNode" and n.embedding]
    E = np.array([c.embedding for c in chunks], dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    _CACHE["chunks"] = chunks
    _CACHE["E"] = E
    return chunks, E


def retrieve(query: str, top_k: int = 5):
    index = build_or_load_graph()
    chunks, E = _get_chunks(index)
    q = np.array(Settings.embed_model.get_query_embedding(query), dtype=np.float32)
    q /= np.linalg.norm(q)
    sims = E @ q
    top = sims.argsort()[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in top]


def query_knowledge_graph(query: str, similarity_top_k: int = 5) -> str:
    hits = retrieve(query, similarity_top_k)
    parts = []
    for c, s in hits:
        m = c.properties
        text = c.text[:2000]
        parts.append(f"[{m.get('file_name','?')} p.{m.get('page_label','?')} | score={s:.3f}]\n{text}")
    context = "\n\n---\n\n".join(parts)

    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say so. "
        "Keep reasoning brief. "
        "Cite each claim as [file p.page].\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    resp = Settings.llm.complete(prompt)
    text = str(resp).strip()
    if not text:
        try:
            msg = resp.raw.choices[0].message
            text = (getattr(msg, 'content', None) or
                    getattr(msg, 'reasoning_content', None) or
                    '[empty response]')
        except Exception:
            text = '[empty response]'
    return text


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", default="What topics are covered in the documents?")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--retrieve-only", action="store_true", help="show chunks, skip LLM")
    args = p.parse_args()

    if args.rebuild:
        build_or_load_graph(force_rebuild=True)
    if args.retrieve_only:
        for c, s in retrieve(args.query, args.top_k):
            m = c.properties
            print(f"[{s:.3f}] {m.get('file_name','?')} p.{m.get('page_label','?')}")
            print(c.text[:300])
            print()
    else:
        print(query_knowledge_graph(args.query, similarity_top_k=args.top_k))
