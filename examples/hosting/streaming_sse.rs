//! Real, end-to-end token streaming over HTTP: a canned streaming agent
//! served by the DevUI-style `AgentHost` router, plus an in-process client
//! that POSTs `/v1/responses` with `"stream": true` and prints each
//! `response.output_text.delta` as it lands on the wire -- not after
//! buffering the whole reply.
//!
//! `agent-framework-examples` doesn't pull in an HTTP client crate (e.g.
//! `reqwest`), so the client below is a deliberately minimal HTTP/1.1 + SSE
//! reader hand-rolled over `tokio::net::TcpStream`: just enough to send the
//! request and decode the (possibly chunked) response body incrementally as
//! bytes arrive. A real client would just use `reqwest` or an SSE crate; the
//! point here is the framework's streaming behavior, not HTTP plumbing.
//!
//! Offline and self-terminating: binds an ephemeral local port, serves the
//! router on a background task, drives one streaming request to completion
//! while printing deltas as they arrive, then exits.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example streaming_sse
//! ```

use std::io::Write as _;
use std::net::SocketAddr;
use std::time::Duration;

use agent_framework::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// A canned model that streams its reply one word at a time, with a short
/// delay between each -- so a client watching the wire sees genuinely
/// incremental deltas instead of one chunk that merely looks like streaming.
#[derive(Clone)]
struct StreamingCannedClient;

#[async_trait]
impl ChatClient for StreamingCannedClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(reply_to(&messages)))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        let words: Vec<String> = reply_to(&messages)
            .split(' ')
            .map(|w| format!("{w} "))
            .collect();
        // `stream::unfold` awaits *inside* the generator, so each item is
        // only produced once its delay has actually elapsed -- unlike
        // `stream::iter` over a pre-built `Vec`, which would resolve
        // instantly and give the client nothing to observe arriving live.
        let stream = futures::stream::unfold(words.into_iter(), |mut remaining| async move {
            let word = remaining.next()?;
            tokio::time::sleep(Duration::from_millis(90)).await;
            Some((Ok(ChatResponseUpdate::text(word)), remaining))
        });
        Ok(stream.boxed())
    }
}

fn reply_to(messages: &[Message]) -> String {
    let last = messages.last().map(Message::text).unwrap_or_default();
    format!(
        "You asked: '{last}' -- this reply is streamed word by word straight off the wire, \
         with a short delay between each, so you can watch it arrive."
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = ChatAgent::builder(StreamingCannedClient)
        .name("assistant")
        .instructions("You are a streaming demo assistant.")
        .build();
    let host = AgentHost::new().agent("assistant", agent);

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(io_err)?;
    let addr = listener.local_addr().map_err(io_err)?;
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, host.into_router()).await;
    });

    println!("serving on http://{addr} (in-process demo; no need to curl it yourself)");
    println!("POST /v1/responses {{\"model\":\"assistant\",\"input\":\"...\",\"stream\":true}}\n");

    let request_body = json!({
        "model": "assistant",
        "input": "Tell me something about Rust.",
        "stream": true,
    });

    print!("assistant: ");
    std::io::stdout().flush().ok();

    let (full_text, delta_count) = tokio::time::timeout(
        Duration::from_secs(15),
        stream_sse_deltas(addr, &request_body),
    )
    .await
    .map_err(|_| Error::Configuration("timed out waiting for the SSE stream".into()))??;

    println!(
        "\n\n{delta_count} SSE delta(s) received, {} character(s) total.",
        full_text.len()
    );
    if delta_count < 3 {
        return Err(Error::Configuration(format!(
            "expected several incremental SSE deltas but only saw {delta_count}; \
             streaming isn't behaving as expected"
        )));
    }

    // The server only needs to live for the one request above.
    server.abort();
    Ok(())
}

fn io_err(e: std::io::Error) -> Error {
    Error::Configuration(format!("streaming_sse: {e}"))
}

/// POST `/v1/responses` on `addr` with `body`, read the SSE response
/// incrementally over a raw TCP socket, print each
/// `response.output_text.delta` payload as soon as it's decoded off the
/// wire, and return `(full_text, delta_count)`.
///
/// This hand-rolls just enough HTTP/1.1 to work against this crate's own
/// server: a request line + headers, then a response body that is either
/// `Transfer-Encoding: chunked` or -- since the request sends
/// `Connection: close` -- simply everything up to EOF.
async fn stream_sse_deltas(addr: SocketAddr, body: &Value) -> Result<(String, usize)> {
    let payload = body.to_string();
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\n\
         Host: {addr}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Accept: text/event-stream\r\n\
         Connection: close\r\n\r\n\
         {payload}",
        payload.len(),
    );

    let mut socket = TcpStream::connect(addr).await.map_err(io_err)?;
    socket.write_all(request.as_bytes()).await.map_err(io_err)?;

    let mut raw = Vec::new(); // accumulates bytes until the header block is complete
    let mut header_len: Option<usize> = None;
    let mut chunked = false;
    let mut decoder = ChunkDecoder::default();
    let mut decoded: Vec<u8> = Vec::new(); // de-chunked body bytes seen so far
    let mut scanned = 0usize; // how much of `decoded` has been scanned for lines
    let mut full_text = String::new();
    let mut delta_count = 0usize;
    let mut read_buf = [0u8; 4096];

    loop {
        let n = socket.read(&mut read_buf).await.map_err(io_err)?;
        if n == 0 {
            break; // server closed the connection
        }

        let new_body_bytes: Vec<u8> = match header_len {
            Some(_) => read_buf[..n].to_vec(),
            None => {
                raw.extend_from_slice(&read_buf[..n]);
                match find_subslice(&raw, b"\r\n\r\n") {
                    Some(pos) => {
                        let headers = String::from_utf8_lossy(&raw[..pos]).to_lowercase();
                        chunked = headers.contains("transfer-encoding: chunked");
                        header_len = Some(pos + 4);
                        raw.split_off(pos + 4) // body bytes that arrived alongside the headers
                    }
                    None => continue, // headers not complete yet; read more
                }
            }
        };

        let piece = if chunked {
            decoder.feed(&new_body_bytes)
        } else {
            new_body_bytes
        };
        decoded.extend_from_slice(&piece);

        // Only complete (`\n`-terminated) lines are converted to `str`, so a
        // multi-byte UTF-8 character split across two socket reads is never
        // decoded from a truncated byte slice.
        while let Some(rel) = decoded[scanned..].iter().position(|&b| b == b'\n') {
            let line = String::from_utf8_lossy(&decoded[scanned..scanned + rel]);
            let line = line.trim_end_matches('\r');
            let data = line.strip_prefix("data: ").map(str::to_string);
            scanned += rel + 1;
            let Some(data) = data else { continue };
            if data == "[DONE]" {
                return Ok((full_text, delta_count));
            }
            let Ok(event) = serde_json::from_str::<Value>(&data) else {
                continue;
            };
            if event.get("type").and_then(Value::as_str) == Some("response.output_text.delta") {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    print!("{delta}");
                    std::io::stdout().flush().ok();
                    full_text.push_str(delta);
                    delta_count += 1;
                }
            }
        }
    }

    Ok((full_text, delta_count))
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Minimal incremental HTTP/1.1 chunked-transfer-encoding decoder (no
/// trailers, no chunk extensions beyond `;`) -- just enough to unwrap the
/// framing hyper adds around a streamed response body.
#[derive(Default)]
struct ChunkDecoder {
    carry: Vec<u8>,
    remaining: usize,
}

impl ChunkDecoder {
    /// Feed newly read bytes; returns any newly decoded body bytes.
    fn feed(&mut self, bytes: &[u8]) -> Vec<u8> {
        self.carry.extend_from_slice(bytes);
        let mut out = Vec::new();
        loop {
            if self.remaining == 0 {
                let Some(pos) = find_subslice(&self.carry, b"\r\n") else {
                    break;
                };
                let size_line = String::from_utf8_lossy(&self.carry[..pos]);
                let size_hex = size_line.split(';').next().unwrap_or("").trim();
                let size = usize::from_str_radix(size_hex, 16).unwrap_or(0);
                self.carry.drain(..pos + 2);
                if size == 0 {
                    self.carry.clear();
                    break; // terminal 0-length chunk
                }
                self.remaining = size;
            } else {
                if self.carry.len() < self.remaining + 2 {
                    break; // wait for the rest of this chunk plus its trailing CRLF
                }
                out.extend_from_slice(&self.carry[..self.remaining]);
                self.carry.drain(..self.remaining + 2);
                self.remaining = 0;
            }
        }
        out
    }
}
