#!/bin/bash
set -euo pipefail

docker stop $(docker ps -aq) 2>/dev/null; \
docker rm -f $(docker ps -aq) 2>/dev/null; \
docker rmi -f $(docker images -aq) 2>/dev/null; \
docker volume rm -f $(docker volume ls -q) 2>/dev/null; \
docker network prune -f; \
docker builder prune -a -f; \
docker system prune -a --volumes -f
echo "✅ deepclean завершён."
