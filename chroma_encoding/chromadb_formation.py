import os, re, time, unicodedata
import pandas as pd
from typing import List
from tqdm import tqdm
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
load_dotenv()

final = pd.read_parquet("./data/final_products.parquet")
client_oa = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL_EMB = "text-embedding-3-small"

html_re = re.compile(r"<[^>]+>")
def clean_text(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(html_re, " ", s)
    s = (s.replace("&nbsp;", " ")
           .replace("&mdash;", "—")
           .replace("&amp;", "&"))
    s = unicodedata.normalize("NFKC", s)  # ключевое: нормализация Юникода
    s = re.sub(r"\s+", " ", s).strip()
    return s

df = final.copy()
df["SKU"]   = df["SKU"].astype(str)
df["brand"] = df["PGrName2"].apply(clean_text).str.lower()

df["ProdName_clean"] = df["ProdName"].apply(clean_text)

# --- токен-эмулятор по длине (если нет tiktoken под рукой) ---
# эмпирически для embedding-3-small: 4-5 символов ≈ 1 токен
def clamp_tokens_like(s: str, max_tokens: int = 96) -> str:
    max_chars = max_tokens * 5
    return s if len(s) <= max_chars else s[:max_chars].rstrip() + "…"

def as_title_text(name: str) -> str:
    # НИКАКОЙ desc/usage: только нормализованный title, чуть укоротим
    s = clean_text(name)
    s = s.strip()
    if not s:
        s = "-"
    return clamp_tokens_like(s, max_tokens=96)

# --- эмбеддинг батчами ---
def embed_texts(texts: List[str], batch_size: int = 256, max_retries: int = 5):
    out = []
    texts = [as_title_text(t) for t in texts]
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        for attempt in range(max_retries):
            try:
                resp = client_oa.embeddings.create(model=MODEL_EMB, input=chunk)
                out.extend([e.embedding for e in resp.data])
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt * 0.5)
    return out

# --- Chroma ---
persist_path = "./chroma_data"
os.makedirs(persist_path, exist_ok=True)
client = chromadb.PersistentClient(path=persist_path)
col = client.get_or_create_collection(name="eva_products_title")

def _safe(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)

# --- Заливка (минимальные метаданные) ---
def insert_chunk(chunk: pd.DataFrame):
    rows = chunk.reset_index(drop=True)
    titles = [as_title_text(n) for n in rows["ProdName_clean"].tolist()]

    vec_title = embed_texts(titles)
    ids = rows["SKU"].astype(str).tolist()

    metas = []
    for i, (_, row) in enumerate(rows.iterrows()):
        metas.append({
            "sku": row["SKU"],
            "url": _safe(row.get("URL", "")),
            "brand": _safe(row.get("brand", "")),
            "price": float(row["Price"]) if pd.notna(row.get("Price")) else None,
            "title": titles[i],  
            "pcat": _safe(row.get("PCatName", "")),
            "pgr":  _safe(row.get("PGrName", "")),
            "pgr1": _safe(row.get("PGrName1", ""))
        })

    col.add(ids=ids, embeddings=vec_title, metadatas=metas)


df_sub = df.drop_duplicates(subset=["SKU"]).copy().iloc[:1256]

BATCH = 256
for s in tqdm(range(0, len(df_sub), BATCH)):
    ch = df_sub.iloc[s:s+BATCH]
    insert_chunk(ch)

print(f"✅ Индексация завершена! База: {persist_path} | items: {col.count()}")
print("Коллекции:", client.list_collections())
