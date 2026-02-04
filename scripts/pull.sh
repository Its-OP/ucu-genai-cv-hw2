#!/bin/bash
# Pull latest changes from repository
# Run from the root of the repository
# Usage: bash scripts/pull.sh

set -e

echo "=========================================="
echo "Pulling latest changes"
echo "=========================================="

git pull

echo "=========================================="
echo "Pull complete!"
echo "=========================================="
