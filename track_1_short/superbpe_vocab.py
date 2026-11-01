"""The SuperBPE vocabulary and dataset this branch trains on, in one place.

Upstream trains on GPT-2 tokens. This branch swaps in a 50,304-token SuperBPE vocabulary, so
everything derived from the tokenizer reads it from here instead of tiktoken's "gpt2": the data paths
(config.py), <|endoftext|> (BOS_ID in data.py), the prefix-token table, the canonical mask and the
bits-per-byte byte lengths. data/cached_superbpe.py downloads HF_REPO into DATA_DIR, the .tiktoken
vocabulary included.

The SuperBPE branches differ from each other only in the constants below, except that this branch
(superbpe_vocab_gpt2split) also enlarges the cu_seqlens table in data.py: its documents are shorter in
tokens than the FineWeb-Edu ones and overflow upstream's.
"""
import base64
import functools
import os

HF_REPO = "alexandermorgan/fineweb10B-superbpe50304-gpt2split"
DATA_DIR = "data/fineweb10B_superbpe_gpt2split"  # relative to DATA_PATH, like upstream's data/fineweb10B
TRAIN_SHARDS = 81  # fineweb_train_000001.bin .. fineweb_train_000081.bin

# <|endoftext|>, the document separator in these shards. Id 50303 has no bytes and no merge rank.
EOT_ID = 50303
VOCAB_FILE = os.path.join(os.environ.get("DATA_PATH", "."), DATA_DIR, "superbpe_50304.tiktoken")


@functools.cache
def superbpe_ranks() -> dict[bytes, int]:
    """{token bytes: id} for ids 0..50302, as tiktoken's _mergeable_ranks: an id is its merge's rank."""
    try:
        text = open(VOCAB_FILE).read()
    except FileNotFoundError:
        raise SystemExit(f"{VOCAB_FILE} is missing. Run:\n    python data/cached_superbpe.py 9")
    ranks = {}
    for line in text.splitlines():
        if line.strip():
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    return ranks
