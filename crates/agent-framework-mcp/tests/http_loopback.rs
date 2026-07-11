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
    let transport = McpStreamableHttpTransport::new(url, Default::default(), None);

    let result = tokio::time::timeout(Duration::from_secs(10), transport.call("ping", json!({})))
        .await
        .expect("client call timed out")
        .expect("client call failed");
    assert_eq!(result, json!({}));

    server.join().expect("loopback server thread panicked");
}
