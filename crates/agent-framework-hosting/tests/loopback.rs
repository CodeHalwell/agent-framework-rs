//! Optional smoke test: drive the router over a real loopback TCP socket via
//! `axum::serve` (the exact mechanism [`AgentHost::serve`] wraps).

mod common;

use agent_framework_hosting::AgentHost;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use common::MockAgent;

#[tokio::test]
async fn serve_health_over_loopback() {
    let host = AgentHost::new().agent("assistant", MockAgent::new("a1").arc());

    // Bind an ephemeral port ourselves so we can address it; `AgentHost::serve`
    // is the one-liner `TcpListener::bind(addr) + axum::serve(listener, router)`.
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let router = host.into_router();
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });

    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut buf = String::new();
    stream.read_to_string(&mut buf).await.unwrap();

    assert!(buf.contains("200 OK"), "response: {buf}");
    assert!(buf.contains("\"status\":\"healthy\""));

    server.abort();
}
