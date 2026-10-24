# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-token attention coefficients (``kv_coeff``).

The engine never interprets the payload: 1 keeps a token visible, 0 hides it.
Two places can go wrong *silently*, so they are covered here:

* decoding — a bitmap shorter than the prompt shifts every position;
* block reduction — a wrong reduction either leaks hidden content or leaves a
  query with no visible keys, which turns softmax into NaN.
"""

import base64

import torch

from vllm.v1.attention.backends.flex_attention import _make_kv_coeff_hint
from vllm.v1.worker.gpu.model_runner import _decode_kv_coeff


def _bits(flags):
    out = bytearray((len(flags) + 7) // 8)
    for i, f in enumerate(flags):
        if f:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


class _SP:
    def __init__(self, raw):
        self.extra_args = {"kv_coeff": raw} if raw is not None else {}


# ---------------------------------------------------------------- decoding
def test_decode_base64():
    flags = [True, False, True, True] + [True] * 28
    raw = base64.b64encode(_bits(flags)).decode()
    assert _decode_kv_coeff(_SP(raw), 32) == _bits(flags)


def test_decode_raw_bytes_in_process():
    flags = [True] * 32
    assert _decode_kv_coeff(_SP(_bits(flags)), 32) == _bits(flags)


def test_absent_payload_is_none():
    assert _decode_kv_coeff(_SP(None), 32) is None
    assert _decode_kv_coeff(object(), 32) is None


def test_short_payload_is_ignored():
    """Fail open: hiding nothing is safe, shifting everything is not."""
    assert _decode_kv_coeff(_SP(base64.b64encode(b"\xff").decode()), 32) is None


def test_mismatched_length_is_ignored():
    """A longer bitmap means the server truncated the prompt and we cannot tell
    which end was cut; guessing would mislabel everything after the cut."""
    raw = _bits([False] * 64)          # 8 bytes = 64 bits, prompt is 32
    assert _decode_kv_coeff(_SP(raw), 32) is None


def test_invalid_base64_is_ignored():
    """An unparseable payload must not raise: this runs in the engine loop, and an
    exception there takes the whole server down."""
    assert _decode_kv_coeff(_SP("A"), 32) is None           # bad base64 length
    assert _decode_kv_coeff(_SP("%%%"), 32) is None


def test_non_bytes_payload_is_ignored():
    """Numbers and booleans would otherwise be reinterpreted as bytes."""
    assert _decode_kv_coeff(_SP(123), 32) is None
    assert _decode_kv_coeff(_SP(True), 32) is None
    assert _decode_kv_coeff(_SP([1, 2, 3]), 32) is None


# ------------------------------------------------------------------- hint
class _MD:
    """Minimal stand-in for FlexAttentionMetadata."""

    def __init__(self, doc_ids, block_size=16, n_req=1):
        self.doc_ids = torch.tensor(doc_ids)
        self.block_size = block_size
        self.block_table = torch.zeros(n_req, 8, dtype=torch.int32)


def _mask(md, kv_coeff, n_tokens, n_blocks, q_block=0):
    hint = _make_kv_coeff_hint(md, kv_coeff)
    q_blk = torch.full((n_tokens, 1), q_block, dtype=torch.long)
    kv_blk = torch.arange(n_blocks, dtype=torch.long)[None, :]
    return hint(q_blk, kv_blk, md.block_size)


def test_hint_drops_block_with_hidden_token():
    """A block is dropped as soon as one of its tokens is hidden, so a fully
    hidden block and a single hidden token behave the same."""
    md = _MD(doc_ids=[0] * 32)

    # block 1 fully hidden
    m = _mask(md, {0: _bits([True] * 16 + [False] * 16)}, 32, 2)
    assert bool(m[:16, 0].all())          # block 0 visible
    assert not bool(m[:16, 1].any())      # block 1 hidden

    # a single hidden token in block 0 drops it just the same
    flags = [True] * 32
    flags[3] = False
    m = _mask(md, {0: _bits(flags)}, 32, 2, q_block=1)
    assert not bool(m[:, 0].any())        # block 0 hidden by one token
    assert bool(m[:, 1].all())            # own block still visible


def test_hint_keeps_query_own_block():
    md = _MD(doc_ids=[0] * 32)
    hint = _make_kv_coeff_hint(md, {0: _bits([False] * 32)})
    q_blk = torch.tensor([[0]] + [[1]] * 31, dtype=torch.long)
    kv_blk = torch.arange(2, dtype=torch.long)[None, :]
    m = hint(q_blk, kv_blk, 16)
    assert bool(m[0, 0])                  # a query always sees its own block
    assert bool(m[1:, 1].all())


def test_hint_pads_partial_block_with_visible():
    """Padding bits are not real tokens; they must not hide the last block.

    The decoder owns that distinction (it is the only layer that knows the
    prompt length), so this goes through it the way the runtime does.
    """
    md = _MD(doc_ids=[0] * 20)             # 20 tokens -> 2 blocks, last partial
    raw = _decode_kv_coeff(_SP(base64.b64encode(_bits([True] * 20)).decode()), 20)
    m = _mask(md, {0: raw}, 20, 2)
    assert bool(m[:, 1].any())


def test_hint_leaves_unmasked_requests_visible():
    md = _MD(doc_ids=[0] * 32 + [1] * 32, n_req=2)
    m = _mask(md, {1: _bits([False] * 32)}, 64, 2)
    assert bool(m[:32].all())             # request 0 has no bitmap -> all visible


def test_hint_single_hidden_token_hides_block():
    flags = [True] * 32
    flags[5] = False
    md = _MD(doc_ids=[0] * 32)
    # Queries sit in block 1, so the own-block guard cannot mask the check.
    m = _mask(md, {0: _bits(flags)}, 32, 2, q_block=1)
    assert not bool(m[:, 0].any())        # block 0 has a hidden token
    assert bool(m[:, 1].all())            # own block stays visible


def test_decode_marks_tail_bits_visible():
    """A prompt that is not a multiple of 8 leaves spare bits in the last byte.

    They are padding, not tokens: leaving them at 0 hides the final block, which
    is exactly where the live question sits.
    """
    import numpy as np

    n = 20
    raw = _decode_kv_coeff(_SP(_bits([True] * n)), n)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    assert bits[n:].all()


def test_bit_order_is_lsb_first():
    """Bit 0 of a byte is the earliest token, per the wire format.

    With block_size=1 every token is its own block, so a reversed unpack shows up
    immediately instead of being averaged away by a 16-token block.
    """
    md = _MD(doc_ids=[0] * 8, block_size=1)
    raw = _decode_kv_coeff(_SP(bytes([0b1111_1110])), 8)   # only token 0 hidden
    hint = _make_kv_coeff_hint(md, {0: raw})
    q_blk = torch.full((8, 1), 99, dtype=torch.long)       # no own-block overlap
    kv_blk = torch.arange(8, dtype=torch.long)[None, :]
    m = hint(q_blk, kv_blk, 1)
    assert not bool(m[:, 0].any())        # token 0 (LSB) hidden
    assert bool(m[:, 7].all())            # token 7 (MSB) visible

