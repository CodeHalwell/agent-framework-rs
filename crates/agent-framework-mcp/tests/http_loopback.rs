//! Nice-to-have loopback test: a bare `std::net::TcpListener` thread speaks
//! just enough HTTP/1.1 to serve one canned `application/json` response,
//! exercising the real `reqwest` POST path end-to-end without any external
//! network access or additional dependencies.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use agent_framework_mcp::{McpStreamableHttpTransport, McpTransport as _};
use serde_json::{json, Value};

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Accept one connection on `listener`, bounded by a generous retry loop so a
/// misbehaving client can't hang the test suite forever.
fn accept_with_timeout(listener: &TcpListener) -> TcpStream {
    listener.set_nonblocking(true).expect("set nonblocking");
    for _ in 0..500 {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).expect("set blocking");
                return stream;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => panic!("accept failed: {e}"),
        }
    }
    panic!("timed out waiting for a client connection");
}

/// Read one HTTP/1.1 request's headers + `Content-Length` body from `stream`.
fn read_http_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).expect("read request headers");
        assert!(n > 0, "connection closed before headers completed");
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
    };
    let header_str = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let content_length: usize = header_str
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    let body_start = header_end + 4;
    while buf.len() < body_start + content_length {
        let n = stream.read(&mut chunk).expect("read request body");
        assert!(n > 0, "connection closed before body completed");
        buf.extend_from_slice(&chunk[..n]);
    }
    buf[body_start..body_start + content_length].to_vec()
}

fn write_json_response(stream: &mut TcpStream, body: &Value) {
    let payload = body.to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    stream
        .write_all(response.as_bytes())
        .expect("write response");
    stream.flush().expect("flush response");
}

#[tokio::test]
async fn http_transport_round_trips_over_a_real_loopback_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener local addr");

    let server = std::thread::spawn(move || {
        let mut stream = accept_with_timeout(&listener);
        let body = read_http_request(&mut stream);
        let request: Value = serde_json::from_slice(&body).expect("parse JSON-RPC request body");
        assert_eq!(request.get("method").and_then(Value::as_str), Some("ping"));
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        write_json_response(
            &mut stream,
            &json!({"jsonrpc": "2.0", "id": id, "result": {}}),
        );
    });

    let url = format!("http://{addr}/mcp");
    let transport = McpStreamableHttpTransport::new(url, Default::default(), None).unwrap();

    let result = tokio::time::timeout(Duration::from_secs(10), transport.call("ping", json!({})))
        .await
        .expect("client call timed out")
        .expect("client call failed");
    assert_eq!(result, json!({}));

    server.join().expect("loopback server thread panicked");
}

/// Read one request and return `(header block, body)`.
fn read_request_parts(stream: &mut TcpStream) -> (String, Vec<u8>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).expect("read request headers");
        assert!(n > 0, "connection closed before headers completed");
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
    };
    let header_str = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let content_length: usize = header_str
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    let body_start = header_end + 4;
    while buf.len() < body_start + content_length {
        let n = stream.read(&mut chunk).expect("read request body");
        assert!(n > 0, "connection closed before body completed");
        buf.extend_from_slice(&chunk[..n]);
    }
    (
        header_str,
        buf[body_start..body_start + content_length].to_vec(),
    )
}

fn write_redirect(stream: &mut TcpStream, location: &str) {
    let response = format!(
        "HTTP/1.1 307 Temporary Redirect\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(response.as_bytes()).expect("write");
    stream.flush().expect("flush");
}

#[tokio::test]
async fn a_cross_origin_redirect_does_not_carry_the_configured_headers() {
    // The whole point of scoping: an MCP server that redirects elsewhere must
    // not hand that host the caller's credentials. `reqwest`'s own redirect
    // handling strips three well-known header names on a cross-host hop and
    // passes everything else — `X-Api-Key` included — straight through.
    let victim = TcpListener::bind("127.0.0.1:0").expect("bind");
    let attacker = TcpListener::bind("127.0.0.1:0").expect("bind");
    let victim_addr = victim.local_addr().unwrap();
    let attacker_addr = attacker.local_addr().unwrap();
    // A different port on the same host is a different origin, which is what
    // two loopback listeners can express — and is exactly the case a
    // host-only comparison (`reqwest`'s) would get wrong.
    let attacker_url = format!("http://{attacker_addr}/evil");

    let redirector = std::thread::spawn(move || {
        let mut stream = accept_with_timeout(&victim);
        let (headers, _) = read_request_parts(&mut stream);
        write_redirect(&mut stream, &attacker_url);
        headers
    });
    let receiver = std::thread::spawn(move || {
        let mut stream = accept_with_timeout(&attacker);
        let (headers, body) = read_request_parts(&mut stream);
        write_json_response(
            &mut stream,
            &json!({"jsonrpc": "2.0", "id": 1, "result": {}}),
        );
        (headers, body)
    });

    let headers = McpStreamableHttpTransport::header_map(&[
        ("Authorization".into(), "Bearer super-secret".into()),
        ("X-Api-Key".into(), "also-secret".into()),
    ])
    .unwrap();
    let transport = McpStreamableHttpTransport::new(
        format!("http://{victim_addr}/mcp"),
        headers,
        Some(Duration::from_secs(10)),
    )
    .unwrap();

    let result = tokio::time::timeout(Duration::from_secs(10), transport.call("ping", json!({})))
        .await
        .expect("client call timed out")
        .expect("client call failed");
    assert_eq!(result, json!({}));

    let first = redirector.join().expect("redirector panicked");
    let (second, body) = receiver.join().expect("receiver panicked");
    let first = first.to_ascii_lowercase();
    let second_lower = second.to_ascii_lowercase();

    // The configured origin got both credentials...
    assert!(
        first.contains("authorization: bearer super-secret"),
        "{first}"
    );
    assert!(first.contains("x-api-key: also-secret"), "{first}");
    // ...and the redirect target got neither.
    assert!(
        !second_lower.contains("super-secret"),
        "Authorization leaked across origins: {second}"
    );
    assert!(
        !second_lower.contains("also-secret"),
        "X-Api-Key leaked across origins: {second}"
    );
    // The method and body survive the hop: a redirect that degraded the POST
    // to a GET would drop the JSON-RPC request entirely.
    assert!(second.starts_with("POST /evil"), "{second}");
    let request: Value = serde_json::from_slice(&body).expect("body is still the JSON-RPC request");
    assert_eq!(request["method"], "ping");
}

#[tokio::test]
async fn a_same_origin_redirect_keeps_the_configured_headers() {
    // The negative control: scoping must not break a server that redirects
    // within its own origin (a trailing-slash or path move), which would
    // otherwise start failing authentication.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().unwrap();

    let server = std::thread::spawn(move || {
        let mut first = accept_with_timeout(&listener);
        let (first_headers, _) = read_request_parts(&mut first);
        write_redirect(&mut first, "/mcp/v2");
        let mut second = accept_with_timeout(&listener);
        let (second_headers, body) = read_request_parts(&mut second);
        write_json_response(
            &mut second,
            &json!({"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}),
        );
        (first_headers, second_headers, body)
    });

    let headers =
        McpStreamableHttpTransport::header_map(&[("X-Api-Key".into(), "still-needed".into())])
            .unwrap();
    let transport = McpStreamableHttpTransport::new(
        format!("http://{addr}/mcp"),
        headers,
        Some(Duration::from_secs(10)),
    )
    .unwrap();

    let result = tokio::time::timeout(Duration::from_secs(10), transport.call("ping", json!({})))
        .await
        .expect("client call timed out")
        .expect("client call failed");
    assert_eq!(result, json!({"ok": true}));

    let (first, second, body) = server.join().expect("server panicked");
    assert!(first
        .to_ascii_lowercase()
        .contains("x-api-key: still-needed"));
    assert!(
        second
            .to_ascii_lowercase()
            .contains("x-api-key: still-needed"),
        "a same-origin redirect must keep the headers: {second}"
    );
    assert!(second.starts_with("POST /mcp/v2"), "{second}");
    let request: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(request["method"], "ping");
}

#[tokio::test]
async fn a_non_http_url_is_refused_at_construction() {
    for url in [
        "file:///etc/passwd",
        "ws://host/mcp",
        "not-a-url",
        "http:///",
    ] {
        let err = match McpStreamableHttpTransport::new(url, Default::default(), None) {
            Ok(_) => panic!("{url} should be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("MCP URL") || err.contains("invalid MCP URL"),
            "{url}: {err}"
        );
    }
    assert!(McpStreamableHttpTransport::new("https://h/mcp", Default::default(), None).is_ok());
}

#[tokio::test]
async fn a_session_teardown_follows_a_same_origin_redirect() {
    // `close()` is best effort, but "best effort" means the request reaches
    // the endpoint. With a client that does not follow redirects, a bare send
    // reads the 3xx as a delivered teardown and the server session stays open
    // forever.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().unwrap();

    let server = std::thread::spawn(move || {
        // 1: the JSON-RPC call that establishes the session.
        let mut first = accept_with_timeout(&listener);
        let (_, _) = read_request_parts(&mut first);
        let payload = json!({"jsonrpc": "2.0", "id": 1, "result": {}}).to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: sess-42\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            payload.len(),
            payload
        );
        first.write_all(response.as_bytes()).expect("write");
        first.flush().expect("flush");

        // 2: the DELETE, answered with a same-origin redirect.
        let mut second = accept_with_timeout(&listener);
        let (redirected, _) = read_request_parts(&mut second);
        write_redirect(&mut second, "/mcp/v2");

        // 3: the DELETE again, at the redirect target.
        let mut third = accept_with_timeout(&listener);
        let (followed, _) = read_request_parts(&mut third);
        let ok = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        third.write_all(ok.as_bytes()).expect("write");
        third.flush().expect("flush");
        (redirected, followed)
    });

    let transport = McpStreamableHttpTransport::new(
        format!("http://{addr}/mcp"),
        Default::default(),
        Some(Duration::from_secs(10)),
    )
    .unwrap();
    tokio::time::timeout(Duration::from_secs(10), transport.call("ping", json!({})))
        .await
        .expect("client call timed out")
        .expect("client call failed");
    assert_eq!(transport.session_id().await.as_deref(), Some("sess-42"));

    tokio::time::timeout(Duration::from_secs(10), transport.close())
        .await
        .expect("close timed out")
        .expect("close failed");

    let (redirected, followed) = server.join().expect("server panicked");
    assert!(redirected.starts_with("DELETE /mcp"), "{redirected}");
    assert!(
        followed.starts_with("DELETE /mcp/v2"),
        "the teardown must reach the redirect target: {followed}"
    );
    assert!(
        followed
            .to_ascii_lowercase()
            .contains("mcp-session-id: sess-42"),
        "and must still carry the session it is tearing down: {followed}"
    );
}
