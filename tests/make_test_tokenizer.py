#!/usr/bin/env python3
"""Train a small byte-level BPE tokenizer configured like SmolLM2's (Digits + ByteLevel
pre-tokenizers, <|endoftext|>/<|im_start|>/<|im_end|> specials), so the C++ tokenizer can be
checked against Hugging Face without downloading anything.

  pip install tokenizers
  python3 tests/make_test_tokenizer.py out/tokenizer.json
  python3 tests/check_tokenizer.py build/tokenize out/tokenizer.json
"""
import sys

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from check_tokenizer import EDGE_CASES, random_strings  # noqa: E402

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Digits(individual_digits=True),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
])
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=2000,
    special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
# Train on different strings than the check uses (different seed), plus the edge cases.
tok.train_from_iterator(random_strings(5000, seed=7) + EDGE_CASES, trainer)
tok.save(sys.argv[1])
print(f"saved {sys.argv[1]} with vocab {tok.get_vocab_size()}")
