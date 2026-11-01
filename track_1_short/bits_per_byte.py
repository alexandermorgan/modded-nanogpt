"""Bits per byte: a validation loss that can be compared across tokenizers.

Nats per token depend on how much text a token holds, so runs with different vocabularies cannot be
compared on val_loss. Bits per byte divides the same total loss by the UTF-8 bytes of the target
tokens instead:

    val_bpb = val_loss * num_target_tokens / (num_target_bytes * ln 2)

<|endoftext|> and the padding ids count 0 bytes: they cost nats but stand for no text, so their loss
is spread over the bytes that are there. Only the validation reads this; training is unchanged.
"""
import math

import torch
from torch import Tensor

from track_1_short.superbpe_vocab import superbpe_ranks


def token_byte_lengths(vocab_size: int, device: torch.device) -> Tensor:
    """UTF-8 byte length of every token id, 0 for <|endoftext|> and the padding ids."""
    ranks = superbpe_ranks()
    lengths = torch.zeros(vocab_size, dtype=torch.int64)
    for token_bytes, token_id in ranks.items():
        lengths[token_id] = len(token_bytes)
    return lengths.to(device)


def bits_per_byte(mean_nats_per_token: float, num_tokens: int, num_bytes: int) -> float:
    return mean_nats_per_token * num_tokens / (num_bytes * math.log(2))
