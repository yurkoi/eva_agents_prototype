# -*- coding: utf-8 -*-
import os
from pathlib import Path
from tqdm import tqdm

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("PERFUMES_DIR", ROOT / "data" / "parfumes_knowledge_db")).expanduser()
PERSIST_PATH = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_data")).expanduser()

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "eva_perfumes")
MODEL_EMB = os.getenv("MODEL_EMB", "text-embedding-3-small")  # OpenAI embeddings

def parse_kv_text(txt: str):
    """
    Дуже простий парсер формату 'Ключ: Значення' построчно.
    Непарсені рядки потрапляють у 'content'.
    """
    kv = {}
    content_lines = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k:
                kv[k] = v
                continue
        content_lines.append(line)
    kv["content"] = "\n".join(content_lines).strip()
    return kv

def run_parfumes_formation():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    # 1) Embeddings через OpenAI
    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name=MODEL_EMB,
    )

    # 2) Персистентний клієнт Chroma
    PERSIST_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(PERSIST_PATH))
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )

    # 4) Індексування всіх .txt (один документ = один файл)
    ids, docs, metas = [], [], []

    for p in tqdm(sorted(DATA_DIR.iterdir())):
        if p.suffix != ".txt" or not p.is_file():
            continue

        text = p.read_text(encoding="utf-8", errors="ignore")
        kv = parse_kv_text(text)

        # ID документа — SKU, або початок імені файлу
        sku = kv.get("SKU") or p.stem.split("-", 1)[0]
        doc_id = str(sku)

        # Нормалізований текст (можна замінити просто на `text`)
        normalized = []
        for k in ("Назва", "Бренд", "Серія", "Країна", "Стать",
                  "Основниі акорди", "Класифікація аромату", "Клас",
                  "Початкова нота", "Нота серця", "Кінцева нота",
                  "Тип застосування"):
            if kv.get(k):
                normalized.append(f"{k}: {kv[k]}")
        if kv.get("content"):
            normalized.append(kv["content"])

        doc_text = "\n".join(normalized) or text

        meta = {
            "sku": sku,
            "name": kv.get("Назва"),
            "brand": kv.get("Бренд"),
            "series": kv.get("Серія"),
            "country": kv.get("Країна"),
            "price": float(kv.get("Ціна")),
            "url": kv.get("URL"),
            "image": kv.get("Посилання на фото"),
            "gender": kv.get("Стать"),
        }

        ids.append(doc_id)
        docs.append(doc_text)
        metas.append(meta)
        
    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        print(f"Indexed {len(ids)} documents into '{COLLECTION_NAME}' at {PERSIST_PATH}")
    else:
        print(f"No .txt files found in {DATA_DIR}")

if __name__ == "__main__":
    run_parfumes_formation()
