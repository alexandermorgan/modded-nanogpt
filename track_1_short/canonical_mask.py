"""Canonical token masking for the final validation (record #350), rebuilt for the SuperBPE vocabulary.

The tokenizer never emits certain (prev, cur) token pairs. At the final validation those
tokens are masked out of the softmax, which can only lower the loss. The mask is built in a
forked child process during training so its cost stays on the clock without stalling it.
"""
import os
import signal
import time
import traceback
import unicodedata
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from track_1_short.superbpe_vocab import superbpe_ranks

# Whether a merge may span the seam between a prev-token end and a cur-token start, by the
# pretokenizer class of the character on either side of it.
#
# Upstream this encodes GPT-2's pretokenizer, which stops merges at word boundaries. The SuperBPE
# vocabulary is encoded with pat_str = r"[\s\S]+", i.e. no pretokenization at all, so no seam is
# protected and every entry is True. The two word-boundary rules upstream sets -- a whitespace run
# being one pretoken, and contractions -- do not apply here; keeping them would only mask less than
# is justified.
_CLS = "SNLO?"  # whitespace, number, letter, other, inside a character
_SEAM_OK = np.ones((len(_CLS), len(_CLS)), dtype=bool)
_SEAM_OK[:, _CLS.index("?")] = False  # cur's pretoken keeps going past cur's end

def _char_cls(ch: str) -> str:
    if ch.isspace() and not "\x1c" <= ch <= "\x1f":
        return "S"
    cat = unicodedata.category(ch)[0]
    return cat if cat in "LN" else "O"

def _edge_cls(b: bytes, first: bool) -> int:
    for n in range(1, 5):
        try:
            s = (b[:n] if first else b[-n:]).decode()
        except UnicodeDecodeError:
            continue
        return _CLS.index(_char_cls(s[0] if first else s[-1]))
    return _CLS.index("?")

def build_canonical_mask(vocab_size: int, ranks: dict | None = None) -> np.ndarray:
    """Bit-packed (vocab_size, vocab_size // 8) mask of non-canonical (prev, cur) pairs.

    Bit x of row p is set when the encoder would never emit token x directly after token p,
    i.e. encode(decode([..., p, x])) != [..., p, x], so softmax can drop it.

    Set bits have to hold for the pair *in context*, which is stricter than proving
    encode(decode([p, x])) != [p, x]: the mask is applied mid-document, so text on either
    side of the pair gets a vote. Pairs whose answer depends on it are left unset.

    * Right. x's first-piece trajectory ends in an unbounded interval, which assumes the
      piece stops at x. When x ends mid-character it demonstrably does not, and a merge
      inside the continuation can preempt the seam merge at a lower rank -- so the pair
      survives re-encoding after all.
    * Piece boundaries. This vocabulary does not pretokenize, but its documents are not
      encoded in one pass either: the dataset's tokenizer splits at blank lines, and cuts
      paragraphs over 8,192 characters before a space. A piece therefore starts at a
      document start, a "\\n", or a " ", and its first token follows the previous piece's
      last token with no merge between them. Tokens starting with a space or a newline are
      cleared from every row for that reason. Without this rule, 428 adjacent pairs in the
      3digit dataset's 10,485,760-token validation window would be masked -- each an actual
      target, which would take a -60 logit and wreck the loss.
    """
    ranks = superbpe_ranks() if ranks is None else ranks
    tok = {v: k for k, v in ranks.items()}
    never = 1 << 30

    # Trajectory of each token's first and last BPE piece as (start_rank, piece) intervals,
    # plus the merge rule that finally forms the token. Keyed by token id, which is also the
    # rank of that final merge -- tiktoken numbers a merged token by its own rank.
    firsts, lasts, rules = {}, {}, {}
    for tid, b in tok.items():
        pieces = [bytes([c]) for c in b]
        first_traj, last_traj = [(0, pieces[0])], [(0, pieces[-1])]
        while len(pieces) > 1:
            best = best_i = None
            for i in range(len(pieces) - 1):
                r = ranks.get(pieces[i] + pieces[i + 1])
                if r is not None and (best is None or r < best):
                    best, best_i = r, i
            if best is None:
                break
            rules[tid] = (pieces[best_i], pieces[best_i + 1])
            pieces[best_i:best_i + 2] = [pieces[best_i] + pieces[best_i + 1]]
            if best_i == 0:
                first_traj.append((best + 1, pieces[0]))
            if best_i == len(pieces) - 1:
                last_traj.append((best, pieces[-1]))
        firsts[tid], lasts[tid] = first_traj, last_traj

    def by_piece(trajs):
        idx = {}
        for tid, traj in trajs.items():
            for i, (start, piece) in enumerate(traj):
                end = traj[i + 1][0] if i + 1 < len(traj) else never
                if start < end:
                    idx.setdefault(piece, []).append((start, end, tid))
        return idx

    by_last, by_first = by_piece(lasts), by_piece(firsts)

    # Character class of each token edge. Upstream also tracks tokens ending in a
    # contraction pretoken; without pretokenization there is no such thing here.
    end_cls = np.full(vocab_size, _CLS.index("?"), dtype=np.intp)
    start_cls = np.full(vocab_size, _CLS.index("?"), dtype=np.intp)
    for tid, b in tok.items():
        end_cls[tid], start_cls[tid] = _edge_cls(b, first=False), _edge_cls(b, first=True)

    mask = np.zeros((vocab_size, vocab_size), dtype=bool)
    for rank, (a, b) in rules.items():
        ps = np.array([p for s, e, p in by_last.get(a, ()) if s <= rank < e], dtype=np.intp)
        xs = np.array([x for s, e, x in by_first.get(b, ()) if s <= rank < e], dtype=np.intp)
        if len(ps) and len(xs):
            mask[np.ix_(ps, xs)] |= _SEAM_OK[np.ix_(end_cls[ps], start_cls[xs])]

    # A token that can begin a piece can follow anything: see "Piece boundaries" above.
    piece_start = np.zeros(vocab_size, dtype=bool)
    for tid, b in tok.items():
        piece_start[tid] = b[:1] in (b" ", b"\n")
    mask[:, piece_start] = False

    return np.packbits(mask, axis=1, bitorder="little")

class BackgroundCanonicalMask:
    """Builds the canonical mask concurrently with training, in a forked child process.

    The build is a few seconds of pure-Python work, so running it in a thread would hold the GIL
    and throttle the training loop's kernel launches. A forked child has its own interpreter and
    only computes, writing the result into shared memory.
    """

    def __init__(self, vocab_size: int, owner: bool, print0):
        self.vocab_size = vocab_size
        self.print0 = print0
        self.buf = self.ranks = self.pid = None
        self.pinned = False
        if owner:
            self.buf = torch.empty(vocab_size, vocab_size // 8, dtype=torch.uint8).share_memory_()
            self.ranks = superbpe_ranks()  # cached
            # Page-lock the buffer so that collect's H2D is a direct DMA rather than a staged
            # copy, roughly 10ms instead of 50. Registering is itself slow, which is why it
            # belongs here, before the clock. The mapping is MAP_SHARED, so the child still
            # writes to these same pages and the fork stays safe.
            cudart = torch.cuda.cudart()
            err = cudart.cudaHostRegister(self.buf.data_ptr(), self.buf.nbytes, 0)
            self.pinned = err == cudart.cudaError.success
            if not self.pinned:
                print0(f"NOTE: could not page-lock the canonical mask buffer ({err}), "
                       "so its copy to device will be slower", console=True)

    def start(self):
        if self.buf is None:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # fork out of a threaded process
            self.pid = os.fork()
        if self.pid == 0:
            code = 1
            try:
                np.copyto(self.buf.numpy(), build_canonical_mask(self.vocab_size, self.ranks))
                code = 0
            except BaseException:
                traceback.print_exc()
            os._exit(code)  # skips atexit/CUDA/NCCL teardown, which is not ours to run

    def wait(self, timeout=60.0):
        """Block until the mask is ready. Called from the timed region."""
        if self.pid is None:
            return
        deadline = time.perf_counter() + timeout
        while not (reaped := os.waitpid(self.pid, os.WNOHANG))[0]:
            if time.perf_counter() > deadline:
                os.kill(self.pid, signal.SIGKILL)
                reaped = os.waitpid(self.pid, 0)
                break
            time.sleep(0.05)
        self.pid = None
        if reaped[1]:
            self.print0(f"WARNING: background canonical mask build failed ({reaped[1]}), building it inline", console=True)
            np.copyto(self.buf.numpy(), build_canonical_mask(self.vocab_size, self.ranks))

    def collect(self, out: Tensor):
        """Fill `out` on every rank, then release the shared buffer."""
        assert self.pid is None, "collect before wait"
        if self.buf is not None:
            out.copy_(self.buf)  # kept blocking: the source is released just below
            if self.pinned:
                torch.cuda.cudart().cudaHostUnregister(self.buf.data_ptr())
        dist.broadcast(out, 0)
        self.buf = self.ranks = None
