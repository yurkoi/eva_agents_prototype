import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

PERSIST_PATH = "./chroma_data"
# COLLECTION_NAME = "eva_perfumes"
COLLECTION_NAME = "eva_products_title"
MODEL_EMB = "text-embedding-3-small"

load_dotenv()

embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name=MODEL_EMB,
)
client = chromadb.PersistentClient(path=PERSIST_PATH)
col = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

q = "амінокислоти"
res = col.query(query_texts=[q], n_results=5)
                #where={"price": {"$gte": 1000}})

print(len(res['documents'][0]))
for i, (doc, meta) in enumerate(zip(res["documents"][0], res["metadatas"][0]), 1):
    print(doc)
    print(meta)
    print("*"*20)

# print(res)