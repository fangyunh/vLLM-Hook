#!/usr/bin/env bash
# Build a PDF from TokenHighlighter.md (requires pandoc + xelatex + TeX Gyre fonts).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pandoc TokenHighlighter.md \
  --from markdown+tex_math_dollars \
  --metadata-file=docs/token-highlighter-pdf.yaml \
  --lua-filter=docs/pandoc-pdf-tables.lua \
  --pdf-engine=xelatex \
  --syntax-highlighting=tango \
  --resource-path="$ROOT" \
  -o TokenHighlighter.pdf

echo "Wrote $ROOT/TokenHighlighter.pdf"
