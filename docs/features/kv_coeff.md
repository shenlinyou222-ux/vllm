# Per-request attention coefficients (`kv_coeff`)

Agent systems routinely build a prompt out of parts that deserve very different
amounts of attention: instructions, a few conclusions, and a large amount of
scaffolding (tool output, sub-agent transcripts) that must stay in the context
for correctness but does not need to steer the next token.

`kv_coeff` lets the caller say, per prompt token, whether that token takes part
in attention. Hidden tokens are removed from the attention block mask, so the
kernel never loads their KV blocks. Nothing is freed, moved, or renumbered.

## Use cases

1. **Tool gating.** Keep one shared system prompt that defines every tool, and
   let each agent hide the tools it may not use. Compared to maintaining one
   trimmed system prompt per agent, this keeps the prefix byte-identical, so the
   KV of the tool catalogue is computed once and reused by every agent, and the
   restriction is enforced in attention rather than requested in prose.
2. **Sub-agent isolation.** A sub-agent's transcript can stay in the trunk's
   context (for audit and rollback) while the trunk does not attend to it. Only
   the sub-agent's summary remains visible to the trunk.
3. **History de-noising.** On a long trajectory, hide tool output that has
   already been summarized, so decode attention scales with the part that
   matters instead of with the whole history.

In all three cases the token sequence is unchanged, so the same KV blocks serve
requests with different visible sets.

## API

Pass a per-token bitmap through the existing `extra_args` escape hatch:

```json
{
  "prompt": [ ... token ids ... ],
  "vllm_xargs": { "kv_coeff": "<base64>" }
}
```

* 1 bit per prompt token, LSB first within each byte.
* `1` (default) — the token takes part in attention.
* `0` — the token is hidden from attention.
* Omitted — feature disabled; behavior is identical to a build without it.
* In-process callers may pass raw `bytes` instead of base64.

The payload must be exactly `ceil(prompt_len / 8)` bytes. Anything else is
ignored, which degrades to "nothing hidden": one byte short would shift every
position, and even one byte long means the server truncated the prompt — we
cannot tell which end was cut, and guessing wrong mislabels everything after the
cut.

## Semantics

* Granularity is the KV block. A block is hidden if **any** of its tokens is
  hidden. Callers that need sub-block precision must pad or re-segment.
* A query always sees its own block, even when that block is hidden. Without
  this a row could have no visible keys at all, and softmax would produce NaN
  for the whole batch. As a consequence, hidden tokens inside the query's own
  block stay visible to that block: the last prompt token always sees the rest
  of its block, and a single-block prompt cannot hide anything from its own
  prefill.
* Blocks made of generated tokens are not part of the bitmap and stay visible.
* Hidden blocks keep their position. A sequence `1 2 3 4 5 6` with `2 3 4`
  hidden still places `5` at position 4, so a gap looks like a gap rather than
  a shorter sentence. Compacting positions is a different feature and requires
  recomputing the KV after the cut.

## Caller contract

The engine does not second-guess the caller.

1. **Coordinates are identical to the tokens you send.** If you render a chat
   template or truncate, do it before computing the bitmap.
2. **Visibility is a client-side decision.** The engine does not interpret
   labels, roles, or message types. Mapping "tool output → hidden" belongs in
   the client.
3. **A request's bitmap is fixed for its lifetime.** Do not change it between
   decode steps; the decoded per-block view is cached per request.
4. **On rollback, regenerate the tail.** Cached blocks downstream of a hidden
   span were computed without it. Reading the hidden content itself stays safe
   (its own KV is mask-independent), but any representation derived from it is
   not, so forking with a regenerated tail is the reliable pattern.

## Cost model

| | effect |
|---|---|
| prompt prefill | **unchanged**. Hidden blocks are still computed as queries; their KV must exist for a later rollback |
| decode | attention work scales with the number of visible blocks |
| memory | **unchanged**. Nothing is compacted or offloaded; hidden blocks still occupy cache |
| window | **unchanged**. Positions still run to the full length; this does not make room |

Hidden blocks are first-class candidates for offloading, since nothing reads
them while they stay hidden, but that is not implemented here.

## Observability

The payload is caller-supplied, so a rejected bitmap is reported rather than
silently dropped:

* `WARNING` once per distinct cause (`decode_error`, `length_mismatch`,
  `bad_type`), at debug level for every occurrence. A client that always sends
  the same broken payload cannot flood the log.
* `INFO` on the first request where the bitmap actually hides something, with
  `req_id`, `prompt_len` and `hidden_tokens`; later requests log at debug.
* `INFO` with the derived block counts (`blocks_per_req`, `hidden_blocks`) for
  the first few mask builds, so "the mask is on but hid nothing" is visible.

A malformed payload never raises: it degrades to an unmasked request and the
engine keeps running. There are no metrics counters yet; a counter would have to
travel through the scheduler's `IterationStats` to reach the API server process.

## Future work

The wire format is a bitmap because by the time a request reaches the scheduler
the engine only knows prompt token ids. Callers therefore have to reproduce the
tokenizer to build the bitmap.

vLLM already exposes the character spans of every prompt token through
`return_token_offsets` (surfaced as `prompt_token_offsets`, see
`vllm/renderers/base.py::_tokenize_prompt`), so a follow-up change can accept
`(start, end)` character ranges and resolve them immediately after tokenization.
That removes the need for callers to run the same tokenizer locally, and it is
where sub-block alignment can be handled in one place.

## Not supported

* No compaction, no KV eviction, no position renumbering.
* No CUDA-graph guarantee. The block mask is rebuilt per step and copied into
  persistent buffers, which *should* replay correctly, but this has not been
  validated; use `--enforce-eager` until it is.
* No interaction with speculative decoding or pipeline parallel; untested.
* No metrics. Rejections are logged, but nothing is exported on `/metrics`.
* Batched outputs are not bit-reproducible against a solo run: batching already
  changes numerics upstream of this feature, so tie-breaking can differ.
