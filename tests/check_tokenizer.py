#!/usr/bin/env python3
"""Compare the C++ tokenizer against the Hugging Face `tokenizers` library on 1,000+ strings.

  pip install tokenizers
  python3 tests/check_tokenizer.py build/tokenize models/SmolLM2-360M/tokenizer.json
"""
import json
import random
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer

WORDS = ("the of and to in is was for on that with as it by at from his an were are which this be or has had "
         "not but first one their its new after who they have her she two been other when there all during into "
         "school time may years more most only over city some world would where later up such used many can state "
         "about national out known university united then made people transformer attention gradient kernel cache "
         "Michigan Ann Arbor GPU CPU SIMD AVX2 NEON llama.cpp tokenizer BPE").split()

EDGE_CASES = [
    "", " ", "  ", "\n", "\n\n", "\t", " \n ", "a", " a", "a ", "  a  b   c   ", "\r\n",
    "Hello, world!", "Hello,world!", "I'm here", "you're they've we'll he'd it's I'M SHOUTING", "don't 'quote'",
    "1234567890", "3.14159", "x = 42 + 7", "v2.0.1-beta", "$1,000,000", "2026-10-28", "phone: 555-0199",
    "def f(x):\n    return x ** 2\n", "int main() { return 0; }", "for (int i = 0; i < n; ++i) {}",
    "<html><body>hi</body></html>", "https://example.com/path?q=1&r=2", "user@umich.edu",
    "café naïve résumé", "Ünïcödé", "日本語のテキスト", "中文测试", "한국어", "Привет, мир", "مرحبا بالعالم",
    "emoji 😀🎉 test", "👍🏽", "mixed 日本 english 123", "٣٤٥ arabic digits", "²³ superscripts", "½ ¾",
    "tabs\tand\tnewlines\nmixed  \n  ", "trailing spaces    ", "    leading spaces", "!!!???...", "---===+++",
    "<|endoftext|>", "text <|endoftext|> more", "<|im_start|>user\nhi<|im_end|>", "a<|endoftext|>b",
    " non-breaking space", "zero​width", "é combining", " line sep",
]


def random_strings(n, seed=0):
    rng = random.Random(seed)
    out = []
    punct = list(",.!?;:'\"()[]{}-_/\\@#$%^&*+=<>|~`")
    for _ in range(n):
        kind = rng.random()
        if kind < 0.5:  # sentence-like
            words = [rng.choice(WORDS) for _ in range(rng.randint(1, 20))]
            s = " ".join(words)
            if rng.random() < 0.5:
                s = s[0].upper() + s[1:] + rng.choice([".", "!", "?", ",", ""])
        elif kind < 0.75:  # mixed tokens: numbers, punctuation, whitespace
            parts = []
            for _ in range(rng.randint(1, 15)):
                r = rng.random()
                if r < 0.3: parts.append(str(rng.randint(0, 10 ** rng.randint(1, 8))))
                elif r < 0.5: parts.append("".join(rng.choice(punct) for _ in range(rng.randint(1, 3))))
                elif r < 0.7: parts.append(rng.choice([" ", "  ", "\n", "\t", " \n", "   "]))
                else: parts.append(rng.choice(WORDS))
            s = "".join(parts)
        else:  # random printable + some non-ASCII codepoints
            pool = [chr(c) for c in range(32, 127)] + list("éüñçßøåæœ日本語中文한국어αβγδЖЩ€£¥©®™…—–“”‘’😀🎉")
            s = "".join(rng.choice(pool) for _ in range(rng.randint(1, 40)))
        out.append(s)
    return out


def main():
    binary, tok_path = sys.argv[1], sys.argv[2]
    strings = EDGE_CASES + random_strings(1200)
    hf = Tokenizer.from_file(tok_path)
    expected = [hf.encode(s, add_special_tokens=False).ids for s in strings]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(strings, f)
    res = subprocess.run([binary, tok_path, f.name], capture_output=True, text=True, check=True)
    got = [[int(x) for x in line.split()] for line in res.stdout.split("\n")[: len(strings)]]

    bad = [(s, e, g) for s, e, g in zip(strings, expected, got) if e != g]
    for s, e, g in bad[:10]:
        print(f"MISMATCH {s!r}\n  hf:   {e}\n  ours: {g}")
    print(res.stderr.strip())
    print(f"{len(strings) - len(bad)}/{len(strings)} strings match Hugging Face exactly")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
