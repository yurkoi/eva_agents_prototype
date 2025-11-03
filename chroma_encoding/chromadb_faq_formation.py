# build_faq_chroma.py
import os, re, time, json
from typing import List
from uuid import uuid4
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

# --- Конфіг ---
FILE_PATH   = "./data/FAQ_EVA.txt"
MODEL_EMB   = os.getenv("MODEL_EMB", "text-embedding-3-small")
PERSIST_DIR = "./chroma_data"
COLLECTION  = "faq_eva"   # <-- нова колекція
BATCH       = 1

client_oa = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
os.makedirs(PERSIST_DIR, exist_ok=True)
client_ch = chromadb.PersistentClient(path=PERSIST_DIR)
col = client_ch.get_or_create_collection(name=COLLECTION)

def split_blocks(text: str) -> List[str]:
    parts = [b.strip() for b in re.split(r"(?m)^\s*-----\s*$", text)]
    return [p for p in parts if p]

def as_text(x) -> str:
    s = "" if x is None else str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def embed_texts(texts: List[str], batch_size: int = 256, max_retries: int = 5) -> List[List[float]]:
    out = []
    texts = [as_text(t) for t in texts]
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        for attempt in range(max_retries):
            try:
                resp = client_oa.embeddings.create(model=MODEL_EMB, input=chunk)
                out.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt * 0.5)
    return out

# --- індексація ---
def run_faq_formation():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    blocks = split_blocks(raw)
    print(f"Знайдено блоків: {len(blocks)}")

    for s in tqdm(range(0, len(blocks), BATCH)):
        chunk = blocks[s:s+BATCH]
        vecs  = embed_texts(chunk, batch_size=min(BATCH, 256))

        ids = [f"faq-{s+i:06d}" for i in range(len(chunk))]
        metas = []
        for i, txt in enumerate(chunk):
            first_line = txt.split("\n", 1)[0].strip()
            metas.append({
                "idx": s + i,
                "title": first_line[:160],
                "len": len(txt),
            })

        col.add(ids=ids, embeddings=vecs, documents=chunk, metadatas=metas)

    try:
        cnt = col.count()
    except Exception:
        cnt = "?"
    print(f"✅ Готово. Колекція: {COLLECTION} | items: {cnt} | path: {PERSIST_DIR}")

    q = "підкажіть будь-ласка як повернути товар?"
    qvec = embed_texts([q])[0]
    res = col.query(query_embeddings=[qvec], n_results=5, include=["metadatas", "documents", "distances"])
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    print("\n== Приклад відповідей ==")
    for i, doc in enumerate(docs, 1):
        sim = 1 - float(dists[i-1]) if dists else None
        ttl = metas[i-1].get("title") if metas else ""
        print(f"\n#{i} similarity={sim:.4f} | {ttl}\n{doc[:500]}...")

if __name__ == "__main__":
    run_faq_formation()
