import os
import sys
from huggingface_hub import hf_hub_download
# Download FineWeb-Edu 10B tokenized with the 50,304-token SuperBPE vocabulary whose first
# phase split numbers into groups of up to 3 digits. Same .bin shard format and file names
# as the GPT-2 data, but <|endoftext|> is 50303 (see BOS_ID in train_gpt.py), and the shards
# live in their own directory so they cannot collide with data/fineweb10B/ on a machine that
# also runs the master-branch baseline.
REPO = "alexandermorgan/fineweb-edu10B-superbpe50304-3digit"
MAX_CHUNKS = 77  # the dataset has train shards 000001..000077

def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'finewebedu10B_superbpe')
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id=REPO, filename=fname,
                        repo_type="dataset", local_dir=local_dir)

# The vocabulary itself: train_gpt.py reads it for the prefix-token table and for the
# bits-per-byte metric.
get("superbpe_50304.tiktoken")
get("fineweb_val_%06d.bin" % 0)
num_chunks = MAX_CHUNKS # full FineWeb-Edu 10B. Each chunk is 100M tokens
if len(sys.argv) >= 2: # we can pass an argument to download less
    num_chunks = int(sys.argv[1])
if num_chunks > MAX_CHUNKS:
    raise SystemExit(f"{REPO} has {MAX_CHUNKS} training shards; asked for {num_chunks}")
for i in range(1, num_chunks+1):
    get("fineweb_train_%06d.bin" % i)
