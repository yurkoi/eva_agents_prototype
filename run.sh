#!/bin/bash
set -e

if [ -d ./chroma_data ]; then
  echo "chroma_data уже есть, не трогаю"
else
  mkdir -p ./chroma_data
  echo "создал ./chroma_data"
fi

docker build -t eva-mcp:latest .
( cd ui && docker build -t eva-ui:latest . )

docker compose up -d
