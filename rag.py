"""RAG core: load PDF -> split (512/50) -> embed -> Chroma (page metadata) -> retrieve -> answer."""
import os
import sys
import time

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
PDF_PATH = os.getenv("PDF_PATH", "manual.pdf")
COLLECTION = "manual"
TOP_K = 3

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma = chromadb.PersistentClient(path="chroma_db")

PROMPT = """You are a product-manual assistant.
Answer ONLY from the CONTEXT below. Do not use any outside knowledge and do not guess.
If the CONTEXT does not clearly contain the answer, reply with exactly: I don't know.
Otherwise answer in one or two short sentences.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""


def with_retry(fn, tries=4):
    """Retry on free-tier rate limits (429) / temporary overload (503)."""
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            if i == tries - 1 or ("429" not in msg and "503" not in msg):
                raise
            time.sleep(20 * (i + 1))


def embed(texts, task):
    vectors = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        resp = with_retry(lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task),
        ))
        vectors.extend(e.values for e in resp.embeddings)
        if i + 50 < len(texts):
            time.sleep(2)
    return vectors


def build_index():
    reader = PdfReader(PDF_PATH)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs, metas = [], []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        for chunk in splitter.split_text(text):
            docs.append(chunk)
            metas.append({"page": page_no})
    try:
        chroma.delete_collection(COLLECTION)
    except Exception:
        pass
    col = chroma.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    vectors = embed(docs, "RETRIEVAL_DOCUMENT")
    col.add(
        ids=[f"chunk-{i}" for i in range(len(docs))],
        documents=docs,
        embeddings=vectors,
        metadatas=metas,
    )
    print(f"Indexed {len(docs)} chunks from {len(reader.pages)} pages")
    return col


def get_collection():
    try:
        return chroma.get_collection(COLLECTION)
    except Exception:
        return build_index()


def retrieve(question, k=TOP_K):
    col = get_collection()
    qv = embed([question], "RETRIEVAL_QUERY")[0]
    res = col.query(query_embeddings=[qv], n_results=k)
    return [
        {"text": d, "page": m["page"], "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def answer_question(question):
    chunks = retrieve(question)
    context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks)
    resp = with_retry(lambda: client.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(context=context, question=question),
        config=types.GenerateContentConfig(temperature=0),
    ))
    answer = (resp.text or "").strip().replace("\u2019", "'")
    if answer.lower().startswith("i don't know"):
        return {"answer": "I don't know.", "sources": []}
    pages = sorted({c["page"] for c in chunks})
    return {"answer": answer, "sources": [{"page": p} for p in pages]}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        build_index()
    else:
        print("Usage: python rag.py ingest")