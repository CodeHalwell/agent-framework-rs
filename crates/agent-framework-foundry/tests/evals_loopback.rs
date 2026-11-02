//! `FoundryEvals` against a fake Foundry Evals API on a bare TcpListener:
//! the eval and run bodies, polling, and paged output items.
#![cfg(feature = "evals")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::evaluation::{EvalItem, Evaluator};
use agent_framework_core::tools::FunctionTool;
use agent_framework_core::types::Message;
use agent_framework_foundry::evals::FoundryEvals;
use serde_json::{json, Value};

type Log = Arc<Mutex<Vec<(String, String, Value)>>>;

fn serve() -> (String, Log) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/api/projects/p", listener.local_addr().unwrap());
    let log: Log = Arc::default();
    let seen = log.clone();
    std::thread::spawn(move || {
        let mut polls = 0;
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
            let body: Value = serde_json::from_slice(&buf[head_end + 4..head_end + 4 + len])
                .unwrap_or(Value::Null);
            let mut parts = head.lines().next().unwrap().split(' ');
            let (method, path) = (
                parts.next().unwrap().to_string(),
                parts.next().unwrap().to_string(),
            );
            seen.lock()
                .unwrap()
                .push((method.clone(), path.clone(), body));
            let reply = match (
                method.as_str(),
                path.trim_start_matches("/api/projects/p/openai/v1"),
            ) {
                ("POST", "/evals") => json!({"id": "eval_1"}),
                ("POST", "/evals/eval_1/runs") => json!({"id": "run_1", "status": "queued"}),
                ("GET", "/evals/eval_1/runs/run_1") => {
                    polls += 1;
                    if polls < 2 {
                        json!({"id": "run_1", "status": "in_progress"})
                    } else {
                        json!({
                            "id": "run_1", "status": "completed", "report_url": "https://portal/r",
                            "result_counts": {"passed": 1, "failed": 1, "errored": 0, "total": 2},
                            "per_testing_criteria_results": [{"testing_criteria": "relevance", "passed": 1, "failed": 1}]
                        })
                    }
                }
                ("GET", "/evals/eval_1/runs/run_1/output_items") => json!({
                    "data": [{"id": "oi_1", "status": "pass", "results": [{"name": "relevance", "score": 5, "passed": true, "sample": null}],
                              "sample": {"input": [{"role": "user", "content": "q1"}], "output": [{"role": "assistant", "content": "a1"}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "cached_tokens": 0}}}],
                    "has_more": true, "last_id": "oi_1"
                }),
                ("GET", "/evals/eval_1/runs/run_1/output_items?after=oi_1") => json!({
                    "data": [{"id": "oi_2", "status": "fail", "results": [{"name": "relevance", "score": 1, "passed": false,
                              "sample": {"properties": {"dimension_scores": [{"id": "tone", "score": 1, "applicable": true, "weight": 1, "reason": "rude"}]}}}]}],
                    "has_more": false
                }),
                _ => json!({"error": "no route"}),
            };
            let text = reply.to_string();
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
                text.len()
            );
        }
    });
    (base, log)
}

#[tokio::test]
async fn a_dataset_run_is_created_polled_and_paged() {
    let (base, log) = serve();
    let evals = FoundryEvals::new(&base, "key", "gpt-4o")
        .with_poll_interval(Duration::from_millis(10))
        .with_evaluators(["relevance", "task_adherence", "tool_call_accuracy"]);
    let tool = FunctionTool::new("lookup", "Look up.", json!({"type": "object"}), |_| async {
        Ok(json!(1))
    })
    .into_definition();
    let mut item = EvalItem::new(vec![Message::user("q1"), Message::assistant("a1")]);
    item.tools = Some(vec![tool]);
    item.expected_output = Some("a1".into());
    let results = evals.evaluate(&[item], "smoke").await.unwrap();

    assert_eq!(results.status, "completed");
    assert_eq!((results.passed(), results.failed()), (1, 1));
    assert_eq!(results.report_url.as_deref(), Some("https://portal/r"));
    assert_eq!(results.per_evaluator["relevance"]["failed"], 1);
    assert_eq!(results.items.len(), 2);
    assert_eq!(results.items[0].input_text.as_deref(), Some("q1"));
    assert_eq!(
        results.items[0].token_usage.as_ref().unwrap()["total_tokens"],
        5
    );
    let dims = results.items[1].scores[0].dimensions.as_ref().unwrap();
    assert_eq!(dims[0].id, "tone");
    assert!(results
        .raise_for_status(None)
        .unwrap_err()
        .0
        .contains("See https://portal/r"));

    let log = log.lock().unwrap().clone();
    let eval_body = &log[0].2;
    assert_eq!(eval_body["data_source_config"]["type"], "custom");
    let props = &eval_body["data_source_config"]["item_schema"]["properties"];
    assert!(props.get("tool_definitions").is_some() && props.get("ground_truth").is_some());
    assert_eq!(
        eval_body["testing_criteria"][2]["evaluator_name"],
        "builtin.tool_call_accuracy"
    );
    let run_body = &log[1].2;
    let item = &run_body["data_source"]["source"]["content"][0]["item"];
    assert_eq!(item["query"], "q1");
    assert_eq!(item["response_messages"][0]["content"][0]["text"], "a1");
    assert_eq!(item["tool_definitions"][0]["name"], "lookup");
}

#[tokio::test]
async fn a_run_that_never_finishes_times_out() {
    let (base, _log) = serve();
    let evals = FoundryEvals::new(&base, "key", "gpt-4o")
        .with_poll_interval(Duration::from_millis(5))
        .with_timeout(Duration::ZERO);
    let results = evals
        .evaluate(
            &[EvalItem::new(vec![
                Message::user("q"),
                Message::assistant("a"),
            ])],
            "t",
        )
        .await
        .unwrap();
    assert_eq!(results.status, "timeout");
}
