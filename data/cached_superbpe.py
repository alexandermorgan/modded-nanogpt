import os
import sys
from huggingface_hub import hf_hub_download
# Download this branch's SuperBPE-tokenized shards and vocabulary (track_1_short/superbpe_vocab.py
# names the dataset). Same .bin shard format and file names as the GPT-2 data, but <|endoftext|> is
# 50303, and the shards live in their own directory so they cannot collide with data/fineweb10B/ on a
# machine that also runs the GPT-2 control.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from track_1_short.superbpe_vocab import DATA_DIR, HF_REPO, TRAIN_SHARDS

def get(fname):
    local_dir = os.path.join(ROOT, DATA_DIR)
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id=HF_REPO, filename=fname,
                        repo_type="dataset", local_dir=local_dir)

# The vocabulary itself: train_gpt.py reads it for the prefix-token table, the canonical mask and the
# bits-per-byte metric.
get("superbpe_50304.tiktoken")
get("fineweb_val_%06d.bin" % 0)
num_chunks = TRAIN_SHARDS # the full dataset. Each chunk is 100M tokens
if len(sys.argv) >= 2: # we can pass an argument to download less
    num_chunks = int(sys.argv[1])
if num_chunks > TRAIN_SHARDS:
    raise SystemExit(f"{HF_REPO} has {TRAIN_SHARDS} training shards; asked for {num_chunks}")
for i in range(1, num_chunks+1):
    get("fineweb_train_%06d.bin" % i)
