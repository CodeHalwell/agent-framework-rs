# Tool-call serialization on both hosting surfaces (same upstream baseline, `dc8e226`)

The [PR #28 review round](2026-09-30-dc8e226-pr28-review.md) recorded "neither hosting surface serializes tool calls" as
a capability gap and declined to close it inside a review cycle. The
repository owner overruled that: close it. This is that work.

### What was broken

Core deliberately leaves a `FunctionCallContent` intact for the *caller* to
execute — that is the whole point of a client-side tool. Both hosting
surfaces then serialized only the text:

- `/v1/chat/completions` built its message from `resp.text()`, so a turn
  whose only content was a call became `content: ""` with no `tool_calls`.
- `/v1/responses` built exactly one assistant text item, buffered and
  streaming alike.

The client was told a call had been requested and given nothing to act on:
no `call_id`, no name, no arguments. The previous round's mitigation was to
degrade the `tool_calls` finish reason to `stop`, which kept the surface
honest but left the capability missing.

### What landed

| Surface | Buffered | Streaming |
|---|---|---|
| `/v1/chat/completions` | `message.tool_calls` with `id` / `type` / `function.{name,arguments}`; `content: null` for a call-only turn, as OpenAI sends | `delta.tool_calls` fragments keyed by a stable per-`call_id` `index`; the first sighting identifies the call, later ones carry arguments only |
| `/v1/responses` | `function_call` output items as *siblings* of the message, each with its own item `id` distinct from `call_id` | `response.output_item.added` once per call, then `response.function_call_arguments.delta` per fragment |

Four details were worth getting right rather than approximating:

**Arguments are a JSON string, not an object.** Both wire formats type the
field as a string and clients `JSON.parse` it. `FunctionArguments` is either
a raw string (what a provider streams, possibly a fragment) or a parsed
object, so `util::arguments_string` re-serializes the object case and maps
absent arguments to `"{}"` rather than `null`.

**A streamed call is one call, not one per fragment.** A provider may stream
a single call's arguments across several updates. OpenAI's contract is that
fragments sharing an `index` (chat completions) or an `item_id` (responses)
concatenate, so both paths remember the calls they have already announced
and re-announce nothing. Re-sending `id`/`name` per fragment would have a
strict client either ignore them or read them as further calls.

**The tool finish reason now passes through** whenever the response actually
carries a call — the promise can be kept. Without one it still degrades to
`stop` with the provider's reason in `x_finish_reason`, because a turn that
reports `tool_calls` while declaring none is still an instruction with
nothing behind it.

**The two Responses paths differ on the message item, deliberately.** The
buffered path omits it for a call-only turn, as OpenAI does — an empty
assistant message reads as a blank answer rather than as work to do. The
streaming path keeps it, because the preamble already announced it at
`output_index` 0 before any content was known, and every call event numbered
itself from there; dropping it would shift each call one index away from the
event that announced it. The terminal payload also reuses the item ids the
client already saw, or it cannot correlate the two.

### Codex review on the same PR — five findings, all real

The outbound half above shipped first and drew a review. All five findings
held up, and three of them were in what had just landed.

| Finding | Verdict | Fix |
|---|---|---|
| **P1** A resolved call is re-advertised | **Real, and mine.** Core keeps a `FunctionCallContent` *and* its `FunctionResultContent` in the response — after a local tool ran, and when a provider executed a hosted tool itself. `function_calls_of` collected every historical call, so the client was asked to re-run work already done. `FunctionInvokingChatClient` filters exactly this way, for exactly this reason, twenty lines from code I had read: the precedent was already in the repo. | Both surfaces now serialize only calls with no matching result. |
| **P1** `function_call_output` is not parsed | **Real.** The Responses protocol returns a tool result as a *top-level* item with `call_id` and `output` — no `role`, no `content` — so `item_to_message` turned it into an empty user turn and dropped the result. | `function_call_output` → `FunctionResultContent`, `function_call` → `FunctionCallContent`. |
| **P1** Chat Completions tool-result messages are not parsed | **Real.** The client's follow-up replays the assistant turn with `tool_calls` and adds a `role: "tool"` message with `tool_call_id`; `IncomingMessage` read neither, and every provider converter builds its wire tool messages from `FunctionResultContent` rather than tool-role text. | `IncomingMessage` gains both fields and rebuilds the call/result contents. |
| **P2** An argumentless announcement emits `{}` | **Real, and mine.** A streamed call is often announced before any arguments exist — this repository's own Responses parser builds exactly that, `arguments: None`. Mapping it to `"{}"` put a literal `{}` at the head of the fragment sequence, so a client concatenating deltas parsed `{}{"city":"Oslo"}`. | `arguments_delta` returns `None` there; `"{}"` stays the *buffered* default, where it is the whole value rather than the first of several. |
| **P2** The client ignores `response.incomplete` | **Real, and from the pass above**, which added the event to the host without adding the arm to `parse_responses_event`. A client pointed at this endpoint lost the finish reason, usage and response id for every truncated or filtered stream — exactly what the distinct event name exists to report. | Handled on the same arm as `response.completed`. |

**The first finding turned out to be worse than reported, and the obvious
fix did not close it.** Filtering per update works on the buffered path,
where the whole response is in hand. On the streaming path the result
arrives *after* the call: with local tools,
`FunctionInvokingChatClient::get_streaming_response` runs the entire loop
and then replays each message as its own update, so the call update always
precedes the tool-result update answering it. A per-update filter has
already put the call on the wire, and no later event can recall it — so the
common case, an agent with local tools, still told the client to re-execute
them. A test written to the reported shape passed; one written to the
replay shape failed.

So the streaming paths now **hold calls until the stream ends** and emit
only those still unanswered. The cost is that a call's arguments arrive in
one delta rather than forming incrementally — a presentation detail, since
a client cannot execute a call before it has the whole argument object.
Duplicating a tool's side effects is not a presentation detail, which is
what settles the trade.

### Verification

Twenty-seven tests across both rounds: buffered serialization and
`content: null`, index stability and fragment reassembly, the function call
output items, the announce-then-fill event sequence with monotonic sequence
numbers and id correlation into the terminal payload, the resolved-call
filter on both surfaces and both paths, both inbound round trips (JSON and
non-JSON results), the `response.incomplete` terminal arm, and six
no-regression tests pinning that an ordinary turn is unchanged.

Sixteen mutation probes, each reintroducing one specific bug and each
failing exactly the test written for it and only that test. Full workspace:
**2147 passing, 0 failing**, clippy `-D warnings`, rustfmt and `cargo doc`
clean.

`ResponseObject::output` changes type from `Vec<OutputMessage>` to
`Vec<OutputItem>`, which is a breaking change for callers that read it;
`OutputItem::as_message()` recovers the old view.

The lesson, and it is the same one this session keeps relearning: the fix
was written to the shape the report described rather than to the invariant
it named. "A resolved call must never be advertised" does not stop being
true because the result arrives late, and the test that would have caught
it was the one modelling how the framework actually streams — not the one
modelling the example in the finding.
