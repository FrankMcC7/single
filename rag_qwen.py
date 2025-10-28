
import os, argparse, json, re, pickle
from typing import List, Tuple, Dict
import numpy as np

# ---- Embeddings (CPU) ----
def load_embedder(name: str):
    # Examples: "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2"
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name, device="cpu")
    return model

def embed_texts(model, texts: List[str]) -> np.ndarray:
    return np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")

# ---- PDF/TXT reading ----
def read_pdf(path: str) -> str:
    from pypdf import PdfReader
    r = PdfReader(path)
    return "\n".join([p.extract_text() or "" for p in r.pages])

def read_txt(path: str) -> str:
    return open(path, "r", encoding="utf-8", errors="ignore").read()

# ---- Chunking ----
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+\n", "\n", text).strip()
    if chunk_size <= 0: return [text]
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks

# ---- FAISS index ----
def build_index(docs_dir: str, embed_model_name: str, index_path: str, chunk_size: int, overlap: int) -> None:
    import faiss
    paths, chunks = [], []
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            p = os.path.join(root, fn)
            ext = os.path.splitext(fn.lower())[-1]
            try:
                if ext == ".pdf":
                    text = read_pdf(p)
                elif ext == ".txt":
                    text = read_txt(p)
                else:
                    continue
            except Exception as e:
                print(f"[skip] {p}: {e}")
                continue

            for c in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
                if c.strip():
                    paths.append(p)
                    chunks.append(c)

    if not chunks:
        raise RuntimeError("No chunks found. Put .txt or .pdf files in the folder.")

    emb_model = load_embedder(embed_model_name)
    vecs = embed_texts(emb_model, chunks)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine since normalized
    index.add(vecs)

    meta = {"paths": paths, "chunks": chunks, "embed_model_name": embed_model_name, "chunk_size": chunk_size, "overlap": overlap}
    faiss.write_index(index, index_path + ".faiss")
    with open(index_path + ".meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"[ok] Indexed {len(chunks)} chunks from {len(set(paths))} files.")
    print(f"[ok] Saved: {index_path}.faiss  and  {index_path}.meta.pkl")

def search_index(index_path: str, query: str, top_k: int) -> List[Tuple[int, float]]:
    import faiss
    with open(index_path + ".meta.pkl", "rb") as f:
        meta = pickle.load(f)
    index = faiss.read_index(index_path + ".faiss")
    emb_model = load_embedder(meta["embed_model_name"])
    q = embed_texts(emb_model, [query])
    D, I = index.search(q, top_k)
    res = list(zip(I[0].tolist(), D[0].tolist()))
    return res

def gather_context(index_path: str, hits: List[Tuple[int, float]]) -> Tuple[str, List[Dict]]:
    with open(index_path + ".meta.pkl", "rb") as f:
        meta = pickle.load(f)
    chunks = meta["chunks"]
    paths  = meta["paths"]
    ctx = []
    cites = []
    for idx, score in hits:
        if 0 <= idx < len(chunks):
            ctx.append(chunks[idx])
            cites.append({"path": paths[idx], "score": float(score)})
    return "\n\n----\n\n".join(ctx), cites

# ---- LLM (llama-cpp-python) ----
def load_llm(model_path: str, n_ctx: int = 4096, n_threads: int = 18, n_batch: int = 512):
    from llama_cpp import Llama
    return Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, n_batch=n_batch, verbose=False)

def qwen_prompt(system_prompt: str, user_prompt: str) -> str:
    # Qwen chat template: <|im_start|>user ... <|im_end|><|im_start|>assistant
    sys = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
    usr = f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
    return sys + usr + "<|im_start|>assistant\n"

def answer_with_rag(llm, question: str, context: str, max_tokens: int = 256, json_mode: bool = False) -> str:
    sys = "You answer only using the provided context. If the answer isn't present, say 'INSUFFICIENT_CONTEXT'."
    if json_mode:
        sys += "\nAlways return STRICT JSON only, no extra text."
    user = f"Context:\n{context}\n\nQuestion:\n{question}"
    prompt = qwen_prompt(sys, user)
    out = llm(prompt, max_tokens=max_tokens, temperature=0.1, stop=["<|im_end|>"])
    return out["choices"][0]["text"].strip()

def main():
    ap = argparse.ArgumentParser(description="CPU-only local RAG for Qwen2.5-1.5B-Instruct GGUF")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build FAISS index from folder")
    b.add_argument("--docs", required=True, help="Folder with .txt/.pdf")
    b.add_argument("--embed", default="BAAI/bge-small-en-v1.5", help="SentenceTransformer model name")
    b.add_argument("--index", required=True, help="Output index base path (without extension)")
    b.add_argument("--chunk_size", type=int, default=1000)
    b.add_argument("--overlap", type=int, default=150)

    q = sub.add_parser("query", help="Query the index with local GGUF")
    q.add_argument("--index", required=True, help="Index base path (without extension)")
    q.add_argument("--model", required=True, help="Path to .gguf (Qwen2.5-1.5B-Instruct-Q4_K_M/IQ4_XS)")
    q.add_argument("--question", required=True, help="Your question")
    q.add_argument("--top_k", type=int, default=6)
    q.add_argument("--n_ctx", type=int, default=4096)
    q.add_argument("--n_threads", type=int, default=18)
    q.add_argument("--n_batch", type=int, default=512)
    q.add_argument("--max_tokens", type=int, default=256)
    q.add_argument("--json", action="store_true", help="Strict JSON answers")

    args = ap.parse_args()

    if args.cmd == "build":
        build_index(args.docs, args.embed, args.index, args.chunk_size, args.overlap)
    else:
        hits = search_index(args.index, args.question, args.top_k)
        context, cites = gather_context(args.index, hits)
        llm = load_llm(args.model, n_ctx=args.n_ctx, n_threads=args.n_threads, n_batch=args.n_batch)
        ans = answer_with_rag(llm, args.question, context, max_tokens=args.max_tokens, json_mode=args.json)

        print("=== ANSWER ===")
        print(ans)
        print("\n=== SOURCES ===")
        for c in cites:
            print(f"{c['path']}  (score={c['score']:.3f})")

if __name__ == "__main__":
    main()
