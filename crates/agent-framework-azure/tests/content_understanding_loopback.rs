//! `ContentUnderstandingContextProvider` against a fake Content
//! Understanding service, driven through a real `Agent`.
#![cfg(feature = "content-understanding")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_azure::content_understanding::ContentUnderstandingContextProvider;
use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::prelude::*;
use agent_framework_core::types::DataContent;
use async_trait::async_trait;
use serde_json::{json, Value};

/// Serves analyze → Operation-Location → results. The operation reports
/// `Running` until `ready_after` polls have been made.
fn serve(ready_after: usize) -> (String, Arc<Mutex<Vec<(String, String)>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let log: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let (seen, me) = (log.clone(), base.clone());
    let polls = Arc::new(AtomicUsize::new(0));
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut buf = Vec::new();
            let mut chunk = [0u8; 8192];
            let head_end = loop {
                let n = stream.read(&mut chunk).unwrap();
                buf.extend_from_slice(&chunk[..n]);
                if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break p;
                }
                if n == 0 {
                    break buf.len();
                }
            };
            let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
            let len: usize = head
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse().unwrap())
                })
                .unwrap_or(0);
            while buf.len() < head_end + 4 + len {
                let n = stream.read(&mut chunk).unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            let line = head.lines().next().unwrap().to_string();
            let key = head
                .lines()
                .find_map(|l| {
                    l.strip_prefix("Ocp-Apim-Subscription-Key: ")
                        .or_else(|| l.strip_prefix("ocp-apim-subscription-key: "))
                })
                .unwrap_or_default()
                .to_string();
            seen.lock().unwrap().push((line.clone(), key));
            let (status, extra, body) =
                if line.starts_with("POST") && line.contains(":analyzeBinary") {
                    (
                        202,
                        format!(
                            "Operation-Location: {me}/contentunderstanding/analyzerResults/op1\r\n"
                        ),
                        json!({}),
                    )
                } else if line.starts_with("GET") && line.contains("analyzerResults/op1") {
                    if polls.fetch_add(1, Ordering::SeqCst) + 1 < ready_after {
                        (200, String::new(), json!({"status": "Running"}))
                    } else {
                        (
                            200,
                            String::new(),
                            json!({"status": "Succeeded", "result": {"contents": [{
                                "markdown": "# Invoice\nTotal due: 42",
                                "fields": {"Total": {"type": "number", "valueNumber": 42}}
                            }]}}),
                        )
                    }
                } else {
                    (
                        404,
                        String::new(),
                        json!({"error": {"message": "no route"}}),
                    )
                };
            let text = body.to_string();
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\n{extra}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
                text.len()
            );
        }
    });
    (base, log)
}

/// Records what the model is sent.
#[derive(Clone, Default)]
struct Recorder(Arc<Mutex<Vec<Vec<Message>>>>);

#[async_trait]
impl ChatClient for Recorder {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _o: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatResponse> {
        self.0.lock().unwrap().push(messages);
        Ok(ChatResponse::from_text("ok"))
    }
    async fn get_streaming_response(
        &self,
        _m: Vec<Message>,
        _o: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatStream> {
        unimplemented!()
    }
}

fn pdf_message() -> Message {
    let pdf = DataContent::from_bytes(b"%PDF-1.7 fake", "application/pdf");
    let named = DataContent {
        uri: pdf.uri.replacen(";base64", ";name=invoice.pdf;base64", 1),
        media_type: pdf.media_type,
    };
    Message::with_contents(
        Role::user(),
        vec![Content::text("what is owed?"), Content::Data(named)],
    )
}

fn has_data(messages: &[Message]) -> bool {
    messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .any(|c| matches!(c, Content::Data(_)))
}

#[tokio::test]
async fn an_attachment_is_analyzed_injected_and_stripped() {
    let (base, log) = serve(2);
    let provider = ContentUnderstandingContextProvider::new(&base, "cu-key")
        .with_poll_interval(Duration::from_millis(5));
    let recorder = Recorder::default();
    let agent = Agent::builder(recorder.clone())
        .context_provider(Arc::new(provider))
        .build();
    let mut session = agent.create_session();
    agent
        .run(vec![pdf_message()], Some(&mut session))
        .await
        .unwrap();

    let sent = recorder.0.lock().unwrap()[0].clone();
    assert!(!has_data(&sent), "the raw file never reaches the model");
    let texts: Vec<String> = sent.iter().map(Message::text).collect();
    let doc = texts
        .iter()
        .find(|t| t.contains("Total due: 42"))
        .expect("document injected");
    assert!(
        doc.contains("source: \"invoice.pdf\"") && doc.contains("\"Total\": 42"),
        "{doc}"
    );
    assert!(texts
        .iter()
        .any(|t| t.contains("The user just uploaded 'invoice.pdf'")));
    assert!(texts.iter().any(|t| t == "what is owed?"));
    let requests = log.lock().unwrap().clone();
    assert!(requests[0].0.contains(
        "/contentunderstanding/analyzers/prebuilt-documentSearch:analyzeBinary?api-version="
    ));
    assert_eq!(requests[0].1, "cu-key");

    // The same name again is rejected rather than analyzed twice.
    agent
        .run(vec![pdf_message()], Some(&mut session))
        .await
        .unwrap();
    let second: Vec<String> = recorder.0.lock().unwrap()[1]
        .iter()
        .map(Message::text)
        .collect();
    assert!(second
        .iter()
        .any(|t| t.contains("already uploaded earlier in this session")));
    assert_eq!(
        log.lock()
            .unwrap()
            .iter()
            .filter(|r| r.0.contains(":analyzeBinary"))
            .count(),
        1
    );
}

#[tokio::test]
async fn a_slow_analysis_is_resumed_on_the_next_turn() {
    let (base, _log) = serve(3);
    let provider = ContentUnderstandingContextProvider::new(&base, "k")
        .with_poll_interval(Duration::from_millis(5))
        .with_max_wait(Some(Duration::ZERO));
    let recorder = Recorder::default();
    let agent = Agent::builder(recorder.clone())
        .context_provider(Arc::new(provider))
        .build();
    let mut session = agent.create_session();
    agent
        .run(vec![pdf_message()], Some(&mut session))
        .await
        .unwrap();
    let first: Vec<String> = recorder.0.lock().unwrap()[0]
        .iter()
        .map(Message::text)
        .collect();
    assert!(
        first.iter().any(|t| t.contains("is being analyzed")),
        "{first:?}"
    );

    agent
        .run(vec![Message::user("and now?")], Some(&mut session))
        .await
        .unwrap();
    let second: Vec<String> = recorder.0.lock().unwrap()[1]
        .iter()
        .map(Message::text)
        .collect();
    assert!(
        second.iter().any(|t| t.contains("Total due: 42")),
        "{second:?}"
    );
    assert!(second
        .iter()
        .any(|t| t.contains("analysis is now complete")));
    let state: Value = session.state.get("azure_contentunderstanding").unwrap();
    assert_eq!(state["documents"]["invoice.pdf"]["status"], "ready");
}
