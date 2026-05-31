#!/usr/bin/env bash
# Build and push Docker image for DS-004 Pipeline
# Usage: ./build-and-push.sh
set -euo pipefail

IMAGE="kona01z/ds004-pipeline:latest"

echo "==> Building Docker image: ${IMAGE}"
docker build -t "${IMAGE}" .

echo "==> Pushing to registry: ${IMAGE}"
docker push "${IMAGE}"

echo "==> Done: ${IMAGE}"
