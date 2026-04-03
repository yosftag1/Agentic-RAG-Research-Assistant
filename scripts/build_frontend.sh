#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src/research_assistant/static"
DIST_DIR="$ROOT_DIR/frontend_dist"

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"
cp -R "$SRC_DIR/"* "$DIST_DIR/"

API_BASE_URL_VALUE="${API_BASE_URL:-}"
ESCAPED_API_BASE_URL="${API_BASE_URL_VALUE//&/\\&}"

# Inject backend URL into index.html for static deploys.
sed -i "s|\${API_BASE_URL}|${ESCAPED_API_BASE_URL}|g" "$DIST_DIR/index.html"

echo "Frontend build complete -> $DIST_DIR"
