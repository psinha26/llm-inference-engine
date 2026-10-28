#!/usr/bin/env bash
# Correctness checks against Hugging Face on the real model, plus fp32 vs int8 perplexity.
# Run after scripts/setup.sh. Writes bench/results/verification.md.
#   ./scripts/verify.sh [model-name]          # default SmolLM2-360M
#   PPL_TOKENS=16384 ./scripts/verify.sh      # score more WikiText-2 tokens (default 8192)
set -euo pipefail

NAME="${1:-SmolLM2-360M}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/models/$NAME"
PY="$ROOT/.venv/bin/python"
PPL_TOKENS="${PPL_TOKENS:-8192}"
THREADS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
OUT="$ROOT/bench/results/verification.md"
mkdir -p "$ROOT/bench/results"

echo "installing the reference libraries (torch, transformers, tokenizers) into .venv"
"$ROOT/.venv/bin/pip" install -q torch transformers tokenizers

echo "== tokenizer vs Hugging Face"
TOK=$("$PY" "$ROOT/tests/check_tokenizer.py" "$ROOT/build/tokenize" "$DIR/tokenizer.json" | tee /dev/stderr | tail -1)

echo "== logits and greedy decoding vs Hugging Face transformers (fp32)"
LOGITS=$("$PY" "$ROOT/tests/check_logits.py" "$ROOT/build/engine" "$DIR" | tee /dev/stderr | tail -1)

echo "== perplexity on WikiText-2 ($PPL_TOKENS tokens, ctx 512)"
PPL32=$("$ROOT/build/engine" --model "$DIR/model.bin" --tokenizer "$DIR/tokenizer.json" --threads "$THREADS" \
        --perplexity "$ROOT/models/wikitext-2/wiki.test.raw" --ppl-tokens "$PPL_TOKENS" --ctx 512 | tail -1)
echo "fp32: $PPL32"
PPL8=$("$ROOT/build/engine" --model "$DIR/model-int8-tiled.bin" --tokenizer "$DIR/tokenizer.json" --threads "$THREADS" \
       --perplexity "$ROOT/models/wikitext-2/wiki.test.raw" --ppl-tokens "$PPL_TOKENS" --ctx 512 | tail -1)
echo "int8: $PPL8"

P32=$(echo "$PPL32" | awk '{print $2}')
P8=$(echo "$PPL8" | awk '{print $2}')
DELTA=$(awk -v a="$P32" -v b="$P8" 'BEGIN { printf "%+.2f%%", (b - a) / a * 100 }')
SIZE32=$(du -m "$DIR/model.bin" | cut -f1)
SIZE8=$(du -m "$DIR/model-int8-tiled.bin" | cut -f1)

cat > "$OUT" <<EOF
# Verification: $NAME

| Check | Result |
|---|---|
| Tokenizer vs HF \`tokenizers\` | $TOK |
| Logits + 100-token greedy vs HF \`transformers\` (fp32) | $LOGITS |
| WikiText-2 perplexity, fp32 | $P32 |
| WikiText-2 perplexity, int8 | $P8 ($DELTA vs fp32) |
| Weight file size | fp32 ${SIZE32} MB, int8 ${SIZE8} MB ($(awk -v a="$SIZE32" -v b="$SIZE8" 'BEGIN{printf "%.1f", a/b}')× smaller) |

Perplexity: first $PPL_TOKENS tokens of wiki.test.raw, non-overlapping 512-token windows.
EOF
echo
cat "$OUT"
