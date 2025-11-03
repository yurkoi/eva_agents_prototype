FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && rm -rf /root/.cache/pip

COPY mcp_search.py ./mcp_search.py

RUN mkdir -p /chroma_data

ENV CHROMA_PATH=/chroma_data \
    SHOP_DB_PATH=/shop.db \
    CHROMA_COLLECTION_PREFIX=eva_products_ \
    MODEL_EMB=text-embedding-3-small \
    VECTOR_NAME=title \
    SEARCH_TOPK=5

EXPOSE 8000
CMD ["python3", "mcp_search.py"]
