#!/usr/bin/env bash
# One-command setup: download a model from Hugging Face, convert it, build the engine.
#   ./scripts/setup.sh                       # SmolLM2-360M (default)
#   ./scripts/setup.sh HuggingFaceTB/SmolLM2-360M-Instruct
set -euo pipefail

REPO="${1:-HuggingFaceTB/SmolLM2-360M}"
NAME="$(basename "$REPO")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/models/$NAME"
mkdir -p "$DIR"

for f in config.json tokenizer.json model.safetensors; do
  if [ ! -s "$DIR/$f" ]; then
    echo "downloading $REPO/$f"
    curl -fL --retry 3 -o "$DIR/$f.part" "https://huggingface.co/$REPO/resolve/main/$f"
    mv "$DIR/$f.part" "$DIR/$f"
  fi
done

if [ ! -d "$ROOT/.venv" ]; then
  python3 -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/pip" install -q numpy

"$ROOT/.venv/bin/python" "$ROOT/scripts/convert.py" "$DIR" "$DIR/model.bin"

cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/build" -j >/dev/null

echo
echo "ready. try:"
echo "  ./build/engine --model models/$NAME/model.bin --tokenizer models/$NAME/tokenizer.json --prompt \"The capital of France is\""
