//! Securing a hosted app: `HostingSecurity` adds bearer-token auth and an
//! anti-DNS-rebinding `Host` allowlist to an axum router.
//!
//! Both guards are **opt-in**, which is the thing worth knowing: a plain
//! `AgentHost::new().agent(..).into_router()` is unauthenticated and
//! unguarded, exactly as it always has been. There are three ways to add
//! them, and they differ in what they cover:
//!
//! | Call | Covers |
//! | --- | --- |
//! | `AgentHost::into_router()` | the DevUI routes, guarded only if you called `with_bearer_token` / `with_allowed_hosts` |
//! | `AgentHost::into_secure_router()` | the same, plus the loopback `Host` allowlist applied by default |
//! | `HostingSecurity::apply(router)` | **whatever router you hand it** |
//!
//! The last one is the one to reach for in a real deployment. `AgentHost`'s
//! own middleware wraps only the routes it built, so an `OpenAiRouter`,
//! `A2ARouter`, or `AgUiRouter` you `merge` or `nest` afterwards would sit
//! *outside* the guard -- an unauthenticated execution endpoint next to an
//! authenticated one. Compose the whole app first, then wrap it once.
//!
//! Why the `Host` allowlist exists: a DevUI-style server bound to localhost
//! is still reachable from a browser on the same machine. An attacker's page
//! can point a DNS name they control at `127.0.0.1` and have the victim's
//! browser POST to your agent endpoint. Checking the `Host` header against an
//! allowlist rejects that, since the browser sends the attacker's name.
//! Requests with no `Host` header at all are allowed through (HTTP/1.0
//! clients, in-process test calls).
//!
//! Offline and self-terminating: binds an ephemeral local port, serves a
//! guarded router on a background task, drives six requests against it, and
//! exits. The HTTP client is hand-rolled over `tokio::net::TcpStream` (the
//! examples crate pulls in no HTTP client), which is also what lets it send a
//! deliberately wrong `Host` header.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example hosting_security
//! ```

use std::net::SocketAddr;
use std::time::Duration;

use agent_framework::hosting::security::HostingSecurity;
use agent_framework::hosting::AgentHost;
use agent_framework::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const TOKEN: &str = "s3cret-deploy-token";

/// A canned model, so the example needs no credentials.
#[derive(Clone)]
struct CannedClient;

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        let last = messages.last().map(Message::text).unwrap_or_default();
        Ok(ChatResponse::from_text(format!("You said: {last}")))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let updates = response.messages.into_iter().map(|m| {
            Ok(ChatResponseUpdate {
                contents: m.contents,
                role: Some(m.role),
                ..Default::default()
            })
        });
        Ok(futures::stream::iter(updates).boxed())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(CannedClient)
        .name("assistant")
        .instructions("You are a helpful assistant.")
        .build();

    // Build the whole app first -- here just the DevUI surface, but in a real
    // deployment this is where you'd merge the OpenAI-compatible and A2A
    // routers too (see `openai_compat_server.rs` and `a2a_server.rs`).
    let app = AgentHost::new().agent("assistant", agent).into_router();

    // ...then guard it once, in one place. Layers run outermost-last-added-
    // first, so the host check runs *before* the token comparison: a
    // rebinding attempt is rejected without the token ever being examined.
    let security = HostingSecurity::new()
        .with_bearer_token(TOKEN)
        .with_default_localhost_hosts();
    println!("security configured: {}", security.is_configured());
    let app = security.apply(app);

    // Serve on an ephemeral port, in the background.
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|e| Error::Configuration(format!("bind failed: {e}")))?;
    let addr: SocketAddr = listener
        .local_addr()
        .map_err(|e| Error::Configuration(e.to_string()))?;
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    println!("serving a guarded DevUI router on http://{addr}\n");

    println!("== bearer-token auth ==\n");

    probe(addr, "GET", "/v1/entities", Some("127.0.0.1"), None, None).await?;
    probe(
        addr,
        "GET",
        "/v1/entities",
        Some("127.0.0.1"),
        Some("Bearer wrong-token"),
        None,
    )
    .await?;
    probe(
        addr,
        "GET",
        "/v1/entities",
        Some("127.0.0.1"),
        Some(&format!("Bearer {TOKEN}")),
        None,
    )
    .await?;

    println!("\n== the Host allowlist (anti-DNS-rebinding) ==\n");

    // A correct token is not enough: the Host header has to be on the list.
    probe(
        addr,
        "GET",
        "/v1/entities",
        Some("evil.example.com"),
        Some(&format!("Bearer {TOKEN}")),
        None,
    )
    .await?;
    // `localhost` is on the default list, and the port is ignored.
    probe(
        addr,
        "GET",
        "/v1/entities",
        Some("localhost:9999"),
        Some(&format!("Bearer {TOKEN}")),
        None,
    )
    .await?;

    println!("\n== a real, authenticated run ==\n");

    probe(
        addr,
        "POST",
        "/v1/responses",
        Some("127.0.0.1"),
        Some(&format!("Bearer {TOKEN}")),
        Some(r#"{"model":"assistant","input":"Hello!"}"#),
    )
    .await?;

    println!(
        "\nnote: rejections come back in the OpenAI error shape\n\
         (`{{\"error\": {{\"message\": ..., \"code\": ...}}}}`), so a client written\n\
         against the OpenAI API can surface them without special-casing.\n\n\
         In production, terminate TLS in front of this and keep the token in a\n\
         `SecretString` (see the `settings_and_secrets` example) -- a bearer\n\
         token over plain HTTP is only as private as the network it crosses."
    );

    Ok(())
}

/// Send one HTTP/1.1 request and print a one-line summary of the response.
///
/// Hand-rolled rather than pulled from a crate for two reasons: the examples
/// crate has no HTTP client dependency, and sending a *deliberately wrong*
/// `Host` header is awkward with most of them.
async fn probe(
    addr: SocketAddr,
    method: &str,
    path: &str,
    host: Option<&str>,
    authorization: Option<&str>,
    body: Option<&str>,
) -> Result<()> {
    let mut stream = TcpStream::connect(addr)
        .await
        .map_err(|e| Error::service(format!("connect failed: {e}")))?;

    let mut request = format!("{method} {path} HTTP/1.1\r\n");
    if let Some(host) = host {
        request.push_str(&format!("Host: {host}\r\n"));
    }
    if let Some(auth) = authorization {
        request.push_str(&format!("Authorization: {auth}\r\n"));
    }
    match body {
        Some(body) => {
            request.push_str("Content-Type: application/json\r\n");
            request.push_str(&format!("Content-Length: {}\r\n", body.len()));
            request.push_str("Connection: close\r\n\r\n");
            request.push_str(body);
        }
        None => request.push_str("Connection: close\r\n\r\n"),
    }

    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|e| Error::service(format!("write failed: {e}")))?;

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .await
        .map_err(|e| Error::service(format!("read failed: {e}")))?;
    let text = String::from_utf8_lossy(&raw);

    let status = text.lines().next().unwrap_or("(no status line)").trim();
    let payload = text.split("\r\n\r\n").nth(1).unwrap_or("").trim();
    let summary: String = payload.chars().take(110).collect();

    let label = format!(
        "{method} {path} host={} auth={}",
        host.unwrap_or("(none)"),
        match authorization {
            None => "(none)",
            Some(a) if a.ends_with(TOKEN) => "correct",
            Some(_) => "wrong",
        }
    );
    println!("  {label:<58} {status}");
    if !summary.is_empty() {
        println!("      {summary}");
    }
    Ok(())
}
