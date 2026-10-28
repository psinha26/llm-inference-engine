#!/usr/bin/env bash
# One-command setup: download a model + WikiText-2 from Hugging Face, convert, build.
#   ./scripts/setup.sh                        # SmolLM2-360M
#   ./scripts/setup.sh HuggingFaceTB/SmolLM2-360M-Instruct
#   BENCH=1 ./scripts/setup.sh                # also write the fp32-tiled file the benchmark needs
#
# Disk: ~0.7 GB download + 1.45 GB (fp32) + 0.4 GB (int8) [+ 1.45 GB fp32-tiled with BENCH=1].
set -euo pipefail

REPO="${1:-HuggingFaceTB/SmolLM2-360M}"
NAME="$(basename "$REPO")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/models/$NAME"
mkdir -p "$DIR" "$ROOT/models/wikitext-2"

fetch() {  # fetch <url> <dest>
  if [ ! -s "$2" ]; then
    echo "downloading $1"
    curl -fL --retry 3 -o "$2.part" "$1"
    mv "$2.part" "$2"
  fi
}

for f in config.json tokenizer.json model.safetensors; do
  fetch "https://huggingface.co/$REPO/resolve/main/$f" "$DIR/$f"
done

# WikiText-2 (raw) for the perplexity evaluation; same file llama.cpp's scripts use.
WT="$ROOT/models/wikitext-2/wiki.test.raw"
if [ ! -s "$WT" ]; then
  fetch "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" "$ROOT/models/wikitext-2/wt2.zip"
  unzip -o -j -q "$ROOT/models/wikitext-2/wt2.zip" "wikitext-2-raw/wiki.test.raw" -d "$ROOT/models/wikitext-2"
fi

# Python is only used by the converter and tests, inside a local virtualenv.
if [ ! -d "$ROOT/.venv" ]; then
  python3 -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/pip" install -q numpy
PY="$ROOT/.venv/bin/python"

[ -s "$DIR/model.bin" ] || "$PY" "$ROOT/scripts/convert.py" "$DIR" "$DIR/model.bin"
[ -s "$DIR/model-int8-tiled.bin" ] || "$PY" "$ROOT/scripts/convert.py" "$DIR" "$DIR/model-int8-tiled.bin" --quant int8 --layout tiled
if [ "${BENCH:-0}" = "1" ] && [ ! -s "$DIR/model-tiled.bin" ]; then
  "$PY" "$ROOT/scripts/convert.py" "$DIR" "$DIR/model-tiled.bin" --layout tiled
fi

if ! command -v cmake >/dev/null; then
  echo "cmake not found; installing it into the virtualenv"
  "$ROOT/.venv/bin/pip" install -q cmake
  export PATH="$ROOT/.venv/bin:$PATH"
fi
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/build" -j >/dev/null

THREADS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
echo
echo "ready. try:"
echo "  ./build/engine --model models/$NAME/model-int8-tiled.bin --tokenizer models/$NAME/tokenizer.json \\"
echo "                 --threads $THREADS --temp 0.7 --top-p 0.9 --prompt \"The capital of France is\""
