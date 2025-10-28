
import os, argparse, re
from typing import List, Optional

# Optional PDF support: install with `pip install pypdf`
def read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("PDF reading requires 'pypdf'. Install via: pip install pypdf") from e
    reader = PdfReader(path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, chunk_tokens: int = 2000) -> List[str]:
    # naive chunker by chars (~4 chars per token heuristic)
    approx_chars = chunk_tokens * 4
    text = re.sub(r"\s+\n", "\n", text).strip()
    return [text[i:i+approx_chars] for i in range(0, len(text), approx_chars)] or [""]

def load_model(model_path: str, n_ctx: int, n_threads: Optional[int], n_batch: int):
    from llama_cpp import Llama
    if n_threads is None:
        import os
        cpu = os.cpu_count() or 8
        n_threads = min(24, max(4, cpu - 2))  # stay friendly on Xeon/desktop
    return Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, n_batch=n_batch, verbose=False)

def qwen_chat_prompt(user_msg: str) -> str:
    # Qwen uses <|im_start|> / <|im_end|> chat formatting (per model README)
    return f"<|im_start|>user\n{user_msg}\n<|im_end|>\n<|im_start|>assistant\n"

def summarize_chunk(llm, chunk: str, bullets: int = 5, max_tokens: int = 256) -> str:
    prompt = qwen_chat_prompt(f"Summarize the following text into {bullets} crisp bullet points:\n\n{chunk}")
    out = llm(prompt, max_tokens=max_tokens, temperature=0.2, stop=["<|im_end|>"])
    return out["choices"][0]["text"].strip()

def extract_fields_chunk(llm, chunk: str, fields: List[str], max_tokens: int = 256) -> str:
    schema = ", ".join(fields)
    instr = (
        "Extract the following fields from the text and return STRICT JSON with only these keys "
        f"({schema}). If a field is missing, use null. No prose.\n\n"
    )
    prompt = qwen_chat_prompt(instr + chunk)
    out = llm(prompt, max_tokens=max_tokens, temperature=0.1, stop=["<|im_end|>"])
    return out["choices"][0]["text"].strip()

def main():
    p = argparse.ArgumentParser(description="CPU-only document analysis with Qwen2.5-1.5B-Instruct GGUF (llama.cpp)")
    p.add_argument("--model", required=True, help="Path to .gguf (e.g., D:\\models\\qwen2.5-1.5b-instruct-q4_k_m.gguf)")
    p.add_argument("--input", required=True, help="Path to input .txt or .pdf")
    p.add_argument("--task", choices=["summary", "extract"], default="summary")
    p.add_argument("--fields", nargs="*", default=["invoice_number","date","total","vendor"], help="Keys for extract task")
    p.add_argument("--bullets", type=int, default=5, help="Bullet count for summary")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--n_ctx", type=int, default=8192)
    p.add_argument("--n_threads", type=int, default=None, help="Defaults to sensible value based on CPU")
    p.add_argument("--n_batch", type=int, default=512)
    args = p.parse_args()

    # Load input
    ext = os.path.splitext(args.input.lower())[-1]
    if ext == ".pdf":
        text = read_pdf(args.input)
    elif ext == ".txt":
        text = read_txt(args.input)
    else:
        raise ValueError("Only .txt and .pdf are supported")

    # Init model
    llm = load_model(args.model, n_ctx=args.n_ctx, n_threads=args.n_threads, n_batch=args.n_batch)

    # Process in chunks
    chunks = chunk_text(text, chunk_tokens=min(2000, args.n_ctx // 4))
    results = []
    for i, ch in enumerate(chunks, 1):
        if not ch.strip():
            continue
        if args.task == "summary":
            res = summarize_chunk(llm, ch, bullets=args.bullets, max_tokens=args.max_tokens)
            results.append(f"--- Chunk {i} ---\n{res}")
        else:
            res = extract_fields_chunk(llm, ch, fields=args.fields, max_tokens=args.max_tokens)
            results.append(f"--- Chunk {i} ---\n{res}")

    print("\n\n".join(results) if results else "No content processed.")

if __name__ == "__main__":
    main()
