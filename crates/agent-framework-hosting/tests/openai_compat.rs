//! OpenAI Chat Completions hosting tests: non-streaming JSON and streaming
//! chunk framing.

mod common;

use agent_framework_hosting::openai_compat::OpenAiRouter;
use axum::http::StatusCode;
use serde_json::json;

use common::{
    parse_sse, parse_sse_json, post_json, post_raw, AnnounceThenStreamAgent, CancelTrackingAgent,
    MockAgent, RecordingAgent, ResolvedCallAgent, StreamStep, StreamingAgent, ToolCallingAgent,
};

fn router() -> axum::Router {
    OpenAiRouter::for_agent("assistant", MockAgent::new("a1").with_usage(5, 3).arc()).into_router()
}

#[tokio::test]
async fn chat_completions_non_stream() {
    let body = json!({
        "model": "assistant",
        "messages": [
            { "role": "system", "content": "be terse" },
            { "role": "user", "content": "ping" },
        ],
    });
    let (status, resp) = post_json(router(), "/v1/chat/completions", &body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["object"], "chat.completion");
    assert_eq!(resp["model"], "assistant");
    assert_eq!(resp["choices"][0]["index"], 0);
    assert_eq!(resp["choices"][0]["message"]["role"], "assistant");
    assert_eq!(
        resp["choices"][0]["message"]["content"],
        "echo: be terse ping"
    );
    assert_eq!(resp["choices"][0]["finish_reason"], "stop");
    // Usage flows through from the agent.
    assert_eq!(resp["usage"]["prompt_tokens"], 5);
    assert_eq!(resp["usage"]["completion_tokens"], 3);
    assert_eq!(resp["usage"]["total_tokens"], 8);
}

#[tokio::test]
async fn chat_completions_stream_chunks() {
    let body = json!({
        "model": "assistant",
        "stream": true,
        "messages": [{ "role": "user", "content": "ping" }],
    });
    let (status, text) = post_raw(router(), "/v1/chat/completions", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let payloads = parse_sse(&text);
    assert_eq!(payloads.last().unwrap(), "[DONE]");

    let chunks: Vec<serde_json::Value> = payloads
        .iter()
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();

    // Every chunk is a chat.completion.chunk with a consistent id.
    let id = chunks[0]["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("chatcmpl-"));
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["id"], id);
    }

    // First chunk sets the assistant role.
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");

    // A content chunk carries the reply text.
    let content = chunks
        .iter()
        .find_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .unwrap();
    assert_eq!(content, "echo: ping");

    // The final chunk closes with finish_reason "stop".
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn chat_completions_content_parts_array() {
    // OpenAI clients may send content as an array of parts.
    let body = json!({
        "messages": [{
            "role": "user",
            "content": [{ "type": "text", "text": "hi there" }],
        }],
    });
    let (status, resp) = post_json(router(), "/v1/chat/completions", &body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["choices"][0]["message"]["content"], "echo: hi there");
}

#[tokio::test]
async fn chat_completions_stream_emits_incremental_content_chunks() {
    // A multi-delta streaming agent yields one content chunk per update.
    let router = OpenAiRouter::for_agent(
        "assistant",
        StreamingAgent::new("s1", vec!["Hel", "lo ", "world"]).arc(),
    )
    .into_router();

    let body = json!({
        "model": "assistant",
        "messages": [{ "role": "user", "content": "hi" }],
        "stream": true,
    });
    let (status, text) = post_raw(router, "/v1/chat/completions", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let data = parse_sse(&text);
    assert_eq!(data.last().unwrap(), "[DONE]");
    let chunks: Vec<serde_json::Value> = data
        .iter()
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();

    // First chunk: role. Then one content chunk per delta. Last: finish_reason.
    let contents: Vec<&str> = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(contents, vec!["Hel", "lo ", "world"]);
    assert_eq!(
        chunks.first().unwrap()["choices"][0]["delta"]["role"],
        "assistant"
    );
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
}

// ---------------------------------------------------------------------------
// Streaming backpressure & disconnect cancellation (bounded channel)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_disconnect_cancels_the_agent_run() {
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use std::sync::atomic::Ordering;
    use std::time::Duration;
    use tower::ServiceExt;

    let agent = CancelTrackingAgent::new("a1");
    let cancelled = agent.cancelled();
    let produced = agent.produced();
    let app = OpenAiRouter::for_agent("assistant", agent.arc()).into_router();

    let body = json!({
        "model": "assistant",
        "messages": [{ "role": "user", "content": "hi" }],
        "stream": true,
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");

    let response = app.oneshot(request).await.expect("router responds");
    assert_eq!(response.status(), StatusCode::OK);

    // Read a couple of SSE frames so streaming has genuinely started, then
    // "disconnect" by dropping the response body.
    let mut resp_body = response.into_body();
    let _ = resp_body.frame().await;
    let _ = resp_body.frame().await;
    drop(resp_body);

    // The producer must observe the disconnect and drop the agent stream,
    // flipping the cancel flag. Poll briefly for the async task to react.
    let mut cancelled_observed = false;
    for _ in 0..200 {
        if cancelled.load(Ordering::SeqCst) {
            cancelled_observed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(
        cancelled_observed,
        "the agent run must be cancelled when the client disconnects"
    );

    // Backpressure + cancellation means the producer stopped early — it did not
    // run away producing the full million-delta stream for a client that left.
    let count = produced.load(Ordering::SeqCst);
    assert!(
        count < 10_000,
        "producer kept generating after disconnect (produced {count})"
    );
}

/// Chat Completions' `finish_reason` is a closed set. A provider string
/// outside it — Anthropic's converter deliberately preserves
/// `model_context_window_exceeded` — can make a strict generated client
/// reject the whole response, so it is approximated to a legal non-success
/// value and carried verbatim beside it rather than lost.
#[tokio::test]
async fn an_unfamiliar_finish_reason_stays_inside_the_closed_enum() {
    let agent = MockAgent::new("a1")
        .with_finish_reason("model_context_window_exceeded")
        .arc();
    let router = OpenAiRouter::for_agent("assistant", agent).into_router();
    let body = json!({ "model": "assistant", "messages": [{ "role": "user", "content": "ping" }] });

    let (status, resp) = post_json(router, "/v1/chat/completions", &body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        resp["choices"][0]["finish_reason"], "length",
        "approximated to a legal value that still says 'not a complete answer'"
    );
    assert_eq!(
        resp["choices"][0]["x_finish_reason"], "model_context_window_exceeded",
        "and the provider's own reason is not lost"
    );
}

/// The reasons OpenAI already names pass through untouched, with no
/// extension field to explain away.
#[tokio::test]
async fn schema_named_finish_reasons_pass_through_unchanged() {
    for reason in ["stop", "length", "content_filter"] {
        let agent = MockAgent::new("a1").with_finish_reason(reason).arc();
        let router = OpenAiRouter::for_agent("assistant", agent).into_router();
        let body =
            json!({ "model": "assistant", "messages": [{ "role": "user", "content": "ping" }] });

        let (_, resp) = post_json(router, "/v1/chat/completions", &body).await;
        assert_eq!(resp["choices"][0]["finish_reason"], reason);
        assert!(
            resp["choices"][0]["x_finish_reason"].is_null(),
            "{reason} needs no approximation"
        );
    }
}

/// A tool finish reason is a promise that `message.tool_calls` carries a
/// call to execute. A turn that reports one while declaring no call cannot
/// keep that promise — advertising it would hand the client an instruction
/// with no id, name or arguments — so it degrades to `stop` with the real
/// reason preserved instead.
#[tokio::test]
async fn a_tool_finish_reason_is_not_advertised_without_the_calls() {
    for reason in ["tool_calls", "function_call"] {
        let agent = MockAgent::new("a1").with_finish_reason(reason).arc();
        let router = OpenAiRouter::for_agent("assistant", agent).into_router();
        let body =
            json!({ "model": "assistant", "messages": [{ "role": "user", "content": "ping" }] });

        let (_, resp) = post_json(router, "/v1/chat/completions", &body).await;
        let choice = &resp["choices"][0];
        assert!(
            choice["message"]["tool_calls"].is_null(),
            "precondition: this agent declares no call"
        );
        assert_eq!(
            choice["finish_reason"], "stop",
            "{reason} must not be advertised without the calls it promises"
        );
        assert_eq!(
            choice["x_finish_reason"], reason,
            "and the real reason is still reported"
        );
    }
}

/// The other half of that rule: when the turn *does* declare calls, the
/// promise is kept. Core leaves a `FunctionCallContent` intact for the
/// caller to execute, so this surface has to put the id, name and arguments
/// on the wire — serializing only `resp.text()` would tell the client to run
/// something it was never shown.
#[tokio::test]
async fn declared_calls_are_serialized_and_back_the_tool_finish_reason() {
    let agent = ToolCallingAgent::new("a1")
        .with_call("call_1", "get_weather", r#"{"city":"Oslo"}"#)
        .with_call("call_2", "get_time", r#"{"tz":"UTC"}"#)
        .with_finish_reason("tool_calls")
        .arc();
    let router = OpenAiRouter::for_agent("assistant", agent).into_router();
    let body = json!({ "model": "assistant", "messages": [{ "role": "user", "content": "ping" }] });

    let (status, resp) = post_json(router, "/v1/chat/completions", &body).await;
    assert_eq!(status, StatusCode::OK);
    let choice = &resp["choices"][0];
    assert_eq!(
        choice["finish_reason"], "tool_calls",
        "the calls are there, so the reason passes through"
    );
    assert!(
        choice["x_finish_reason"].is_null(),
        "nothing was approximated"
    );

    let calls = choice["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls serialized");
    assert_eq!(calls.len(), 2, "every declared call, in order");
    assert_eq!(calls[0]["id"], "call_1");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(
        calls[0]["function"]["arguments"], r#"{"city":"Oslo"}"#,
        "arguments are a JSON *string*, as the wire format has it"
    );
    assert_eq!(calls[1]["id"], "call_2");
    assert_eq!(calls[1]["function"]["name"], "get_time");
    assert_eq!(calls[1]["function"]["arguments"], r#"{"tz":"UTC"}"#);
}

/// OpenAI sends `content: null` for a turn that is only a tool call. A
/// client that renders `content` verbatim would otherwise print an empty
/// string where the call should be.
#[tokio::test]
async fn a_call_only_turn_sends_null_content() {
    let calls_only = ToolCallingAgent::new("a1")
        .with_call("call_1", "get_weather", r#"{"city":"Oslo"}"#)
        .arc();
    let router = OpenAiRouter::for_agent("assistant", calls_only).into_router();
    let body = json!({ "model": "assistant", "messages": [{ "role": "user", "content": "ping" }] });
    let (_, resp) = post_json(router, "/v1/chat/completions", &body).await;
    assert!(
        resp["choices"][0]["message"]["content"].is_null(),
        "a turn with no text and a call is `content: null`, not \"\""
    );

    // Text alongside the call keeps the text: `null` is about the absence of
    // content, not about the presence of a call.
    let both = ToolCallingAgent::new("a1")
        .with_text("checking now")
        .with_call("call_1", "get_weather", r#"{"city":"Oslo"}"#)
        .arc();
    let router = OpenAiRouter::for_agent("assistant", both).into_router();
    let (_, resp) = post_json(router, "/v1/chat/completions", &body).await;
    assert_eq!(resp["choices"][0]["message"]["content"], "checking now");
    assert_eq!(
        resp["choices"][0]["message"]["tool_calls"][0]["id"],
        "call_1"
    );
}

/// An ordinary completion is unchanged: no `tool_calls` key at all, rather
/// than an empty array a strict client might read as "a call was attempted".
#[tokio::test]
async fn an_ordinary_completion_carries_no_tool_calls_key() {
    let body = json!({ "model": "assistant", "messages": [{ "role": "user", "content": "ping" }] });
    let (_, resp) = post_json(router(), "/v1/chat/completions", &body).await;
    let message = resp["choices"][0]["message"]
        .as_object()
        .expect("message object");
    assert!(
        !message.contains_key("tool_calls") || message["tool_calls"].is_null(),
        "no calls declared, so the key is absent"
    );
}

/// Streaming deltas follow OpenAI's contract: each call is identified by a
/// stable `index`, and a call streamed across several updates arrives as
/// one delta with its fragments reassembled. Emitting them per fragment
/// would have had to send each call before knowing whether the agent would
/// answer it itself, so they are held until the stream ends.
#[tokio::test]
async fn streamed_call_fragments_share_one_stable_index() {
    let agent = ToolCallingAgent::new("a1")
        .with_finish_reason("tool_calls")
        .streaming(vec![
            StreamStep::Call {
                call_id: "call_1".into(),
                name: "get_weather".into(),
                arguments: r#"{"city":"#.into(),
            },
            StreamStep::Call {
                call_id: "call_2".into(),
                name: "get_time".into(),
                arguments: r#"{"tz":"UTC"}"#.into(),
            },
            StreamStep::Call {
                call_id: "call_1".into(),
                name: "get_weather".into(),
                arguments: r#""Oslo"}"#.into(),
            },
        ])
        .arc();
    let router = OpenAiRouter::for_agent("assistant", agent).into_router();
    let body = json!({
        "model": "assistant",
        "stream": true,
        "messages": [{ "role": "user", "content": "ping" }],
    });
    let (status, text) = post_raw(
        router,
        "/v1/chat/completions",
        serde_json::to_string(&body).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let deltas: Vec<serde_json::Value> = parse_sse_json(&text)
        .into_iter()
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["tool_calls"]
                .as_array()
                .cloned()
        })
        .flatten()
        .collect();
    assert_eq!(deltas.len(), 2, "one delta per call, not per fragment");

    // Index is position, and each call is identified exactly once.
    assert_eq!(deltas[0]["index"], 0);
    assert_eq!(deltas[0]["id"], "call_1");
    assert_eq!(deltas[0]["function"]["name"], "get_weather");
    assert_eq!(
        deltas[0]["function"]["arguments"], r#"{"city":"Oslo"}"#,
        "call_1's two fragments are reassembled, in order"
    );
    assert_eq!(deltas[1]["index"], 1);
    assert_eq!(deltas[1]["id"], "call_2");
    assert_eq!(deltas[1]["function"]["arguments"], r#"{"tz":"UTC"}"#);

    // The deltas precede the terminal chunk, which can now keep the
    // tool promise.
    let chunks = parse_sse_json(&text);
    let last = chunks.last().expect("terminal chunk");
    assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    assert!(
        last["choices"][0]["delta"]["tool_calls"].is_null(),
        "the terminal chunk carries the reason, not the calls"
    );
}

/// A streamed turn that reports `tool_calls` but streams no call gets the
/// same degradation as the buffered path — the two must not disagree.
#[tokio::test]
async fn a_streamed_tool_reason_without_calls_degrades_too() {
    let agent = StreamingAgent::new("a1", vec!["hello"]).arc();
    let router = OpenAiRouter::for_agent("assistant", agent).into_router();
    let body = json!({
        "model": "assistant",
        "stream": true,
        "messages": [{ "role": "user", "content": "ping" }],
    });
    let (_, text) = post_raw(
        router,
        "/v1/chat/completions",
        serde_json::to_string(&body).unwrap(),
    )
    .await;
    let last = parse_sse_json(&text)
        .into_iter()
        .last()
        .expect("terminal chunk");
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
}

/// A call that already carries its result is **not** work for the client.
/// Core keeps the pair in the response — after a local tool ran, and when a
/// provider executed a hosted tool itself — so serializing every historical
/// call would ask the client to re-run something already done, duplicating
/// whatever side effects it had.
#[tokio::test]
async fn an_already_resolved_call_is_not_re_advertised() {
    let agent = ResolvedCallAgent::new("a1")
        .with_resolved("call_done", "charge_card", r#"{"amount":50}"#, json!("ok"))
        .arc();
    let router = OpenAiRouter::for_agent("assistant", agent).into_router();
    let body = json!({ "model": "assistant", "messages": [{ "role": "user", "content": "ping" }] });

    let (_, resp) = post_json(router, "/v1/chat/completions", &body).await;
    let choice = &resp["choices"][0];
    assert!(
        choice["message"]["tool_calls"].is_null(),
        "the only call was already answered, so nothing is outstanding"
    );
    assert_eq!(
        choice["finish_reason"], "stop",
        "and no tool reason is implied"
    );
}

/// The other half: an outstanding call alongside a resolved one is still
/// advertised. The filter must not swallow real work.
#[tokio::test]
async fn an_outstanding_call_survives_alongside_a_resolved_one() {
    let agent = ResolvedCallAgent::new("a1")
        .with_resolved("call_done", "charge_card", r#"{"amount":50}"#, json!("ok"))
        .with_outstanding("call_todo", "send_receipt", r#"{"to":"a@b.c"}"#)
        .arc();
    let router = OpenAiRouter::for_agent("assistant", agent).into_router();
    let body = json!({ "model": "assistant", "messages": [{ "role": "user", "content": "ping" }] });

    let (_, resp) = post_json(router, "/v1/chat/completions", &body).await;
    let calls = resp["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("the outstanding call is serialized");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["id"], "call_todo");
}

/// The same filter on the streaming path, where the result arrives in a
/// later update than the call it answers.
#[tokio::test]
async fn a_resolved_call_is_not_streamed_either() {
    let agent = ResolvedCallAgent::new("a1")
        .with_resolved("call_done", "charge_card", r#"{"amount":50}"#, json!("ok"))
        .with_outstanding("call_todo", "send_receipt", r#"{"to":"a@b.c"}"#)
        .arc();
    let router = OpenAiRouter::for_agent("assistant", agent).into_router();
    let body = json!({
        "model": "assistant",
        "stream": true,
        "messages": [{ "role": "user", "content": "ping" }],
    });
    let (_, text) = post_raw(
        router,
        "/v1/chat/completions",
        serde_json::to_string(&body).unwrap(),
    )
    .await;
    let ids: Vec<String> = parse_sse_json(&text)
        .into_iter()
        .filter_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array().cloned())
        .flatten()
        .filter_map(|d| d["id"].as_str().map(str::to_string))
        .collect();
    assert_eq!(
        ids,
        vec!["call_todo".to_string()],
        "the resolved call is filtered; the outstanding one is not"
    );
}

/// A streamed call is often announced before any arguments exist — this
/// repository's own Responses parser builds exactly that shape. Emitting
/// `"{}"` for the announcement would put a literal `{}` at the head of the
/// fragment sequence, and a client concatenating deltas by index would then
/// parse `{}{"city":"Oslo"}`.
#[tokio::test]
async fn an_argumentless_announcement_does_not_inject_empty_braces() {
    let agent = AnnounceThenStreamAgent::new(
        "a1",
        "call_1",
        "get_weather",
        vec![r#"{"city":"#, r#""Oslo"}"#],
    )
    .arc();
    let router = OpenAiRouter::for_agent("assistant", agent).into_router();
    let body = json!({
        "model": "assistant",
        "stream": true,
        "messages": [{ "role": "user", "content": "ping" }],
    });
    let (_, text) = post_raw(
        router,
        "/v1/chat/completions",
        serde_json::to_string(&body).unwrap(),
    )
    .await;

    let deltas: Vec<serde_json::Value> = parse_sse_json(&text)
        .into_iter()
        .filter_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array().cloned())
        .flatten()
        .collect();
    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas[0]["id"], "call_1");
    assert_eq!(
        deltas[0]["function"]["name"], "get_weather",
        "the name from the announcement survives the argumentless fragments"
    );

    let arguments = deltas[0]["function"]["arguments"]
        .as_str()
        .expect("arguments string");
    assert_eq!(
        arguments, r#"{"city":"Oslo"}"#,
        "the argumentless announcement contributes nothing, not `{{}}`"
    );
    serde_json::from_str::<serde_json::Value>(arguments)
        .expect("the reassembled arguments are valid JSON");
}

/// A client that executes an advertised call sends the whole turn back: the
/// assistant message with `tool_calls`, then one `role: "tool"` message per
/// result. Reading only `content` turned that result into an empty tool
/// message and lost the correlation, so the call could never be completed.
#[tokio::test]
async fn a_tool_result_round_trip_reaches_the_agent_as_call_and_result() {
    let agent = RecordingAgent::new("a1");
    let seen = agent.seen();
    let router = OpenAiRouter::for_agent("assistant", agent.arc()).into_router();

    let body = json!({
        "model": "assistant",
        "messages": [
            { "role": "user", "content": "weather in Oslo?" },
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "get_weather", "arguments": "{\"city\":\"Oslo\"}" },
                }],
            },
            { "role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\":7}" },
        ],
    });
    let (status, _) = post_json(router, "/v1/chat/completions", &body).await;
    assert_eq!(status, StatusCode::OK);

    let messages = seen.lock().unwrap().clone();
    assert_eq!(messages.len(), 3);

    let call = messages[1]
        .contents
        .iter()
        .find_map(|c| c.as_function_call())
        .expect("the assistant's call survived the round trip");
    assert_eq!(call.call_id, "call_1");
    assert_eq!(call.name, "get_weather");

    let result = messages[2]
        .contents
        .iter()
        .find_map(|c| c.as_function_result())
        .expect("the tool message became a function result, not text");
    assert_eq!(result.call_id, "call_1");
    assert_eq!(
        result.result,
        Some(json!({ "temp_c": 7 })),
        "a JSON-shaped result is stored as the value it represents"
    );
}

/// A tool result that is not JSON stays a string rather than being lost or
/// mangled into one.
#[tokio::test]
async fn a_non_json_tool_result_stays_a_string() {
    let agent = RecordingAgent::new("a1");
    let seen = agent.seen();
    let router = OpenAiRouter::for_agent("assistant", agent.arc()).into_router();
    let body = json!({
        "model": "assistant",
        "messages": [
            { "role": "tool", "tool_call_id": "call_1", "content": "7 degrees and raining" },
        ],
    });
    let (_, _) = post_json(router, "/v1/chat/completions", &body).await;

    let messages = seen.lock().unwrap().clone();
    let result = messages[0]
        .contents
        .iter()
        .find_map(|c| c.as_function_result())
        .expect("function result");
    assert_eq!(result.result, Some(json!("7 degrees and raining")));
}

/// An ordinary request is unaffected: no calls, no results, just text.
#[tokio::test]
async fn an_ordinary_request_still_reaches_the_agent_as_text() {
    let agent = RecordingAgent::new("a1");
    let seen = agent.seen();
    let router = OpenAiRouter::for_agent("assistant", agent.arc()).into_router();
    let body = json!({
        "model": "assistant",
        "messages": [{ "role": "user", "content": "ping" }],
    });
    let (_, _) = post_json(router, "/v1/chat/completions", &body).await;

    let messages = seen.lock().unwrap().clone();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].text(), "ping");
    assert!(messages[0]
        .contents
        .iter()
        .all(|c| c.as_function_call().is_none() && c.as_function_result().is_none()));
}
