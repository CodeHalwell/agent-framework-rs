//! A2A hosting tests: agent card, JSON-RPC `message/send` / `tasks/get` /
//! `tasks/cancel`, and malformed-request error paths.

mod common;

use agent_framework_hosting::a2a::A2ARouter;
use axum::http::StatusCode;
use serde_json::json;

use common::{get_json, post_json, post_raw, MockAgent};

fn router() -> axum::Router {
    A2ARouter::for_agent(
        "invoice-agent",
        MockAgent::new("a1").named("Invoice Agent").arc(),
        "http://localhost:8080/a2a",
    )
    .into_router()
}

#[tokio::test]
async fn agent_card_shape() {
    let (status, card) = get_json(router(), "/.well-known/agent-card.json").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(card["name"], "invoice-agent");
    assert_eq!(card["url"], "http://localhost:8080/a2a");
    assert_eq!(card["preferredTransport"], "JSONRPC");
    assert_eq!(card["protocolVersion"], "0.3.0");
    // Capabilities are camelCase and streaming is off.
    assert_eq!(card["capabilities"]["streaming"], false);
    assert_eq!(card["capabilities"]["pushNotifications"], false);
    assert_eq!(card["capabilities"]["stateTransitionHistory"], false);
    assert_eq!(card["defaultInputModes"], json!(["text"]));
    assert_eq!(card["defaultOutputModes"], json!(["text"]));
    // One skill derived from the agent metadata.
    assert_eq!(card["skills"][0]["id"], "invoice-agent");
}

fn send_message(text: &str) -> serde_json::Value {
    json!({
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": "m1",
                "parts": [{ "kind": "text", "text": text }],
            }
        }
    })
}

#[tokio::test]
async fn message_send_returns_completed_task() {
    let (status, resp) = post_json(router(), "/", &send_message("Show invoices")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], "1");

    let task = &resp["result"];
    assert_eq!(task["kind"], "task");
    assert!(task["id"].is_string());
    assert!(task["contextId"].is_string());
    assert_eq!(task["status"]["state"], "completed");

    // Artifact carries the agent's text reply.
    assert_eq!(task["artifacts"][0]["parts"][0]["kind"], "text");
    assert_eq!(
        task["artifacts"][0]["parts"][0]["text"],
        "echo: Show invoices"
    );

    // History has the inbound user message and the agent reply.
    assert_eq!(task["history"].as_array().unwrap().len(), 2);
    assert_eq!(task["history"][0]["role"], "user");
    assert_eq!(task["history"][1]["role"], "agent");
}

#[tokio::test]
async fn tasks_get_roundtrip() {
    let app = router();

    // Send a message, capture the task id.
    let (_, sent) = post_json(app.clone(), "/", &send_message("Hello")).await;
    let task_id = sent["result"]["id"].as_str().unwrap().to_string();

    // Retrieve it via tasks/get.
    let get = json!({
        "jsonrpc": "2.0", "id": "2", "method": "tasks/get",
        "params": { "id": task_id },
    });
    let (status, resp) = post_json(app, "/", &get).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["result"]["id"], task_id);
    assert_eq!(resp["result"]["status"]["state"], "completed");
}

#[tokio::test]
async fn tasks_get_unknown_is_task_not_found() {
    let get = json!({
        "jsonrpc": "2.0", "id": "3", "method": "tasks/get",
        "params": { "id": "does-not-exist" },
    });
    let (status, resp) = post_json(router(), "/", &get).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["error"]["code"], -32001);
}

#[tokio::test]
async fn tasks_cancel_terminal_task_not_cancelable() {
    let app = router();
    let (_, sent) = post_json(app.clone(), "/", &send_message("Hello")).await;
    let task_id = sent["result"]["id"].as_str().unwrap().to_string();

    let cancel = json!({
        "jsonrpc": "2.0", "id": "4", "method": "tasks/cancel",
        "params": { "id": task_id },
    });
    let (_, resp) = post_json(app, "/", &cancel).await;
    // Completed tasks are terminal → TaskNotCancelableError.
    assert_eq!(resp["error"]["code"], -32002);
}

#[tokio::test]
async fn message_stream_unsupported() {
    let req = json!({
        "jsonrpc": "2.0", "id": "5", "method": "message/stream",
        "params": { "message": { "kind": "message", "role": "user", "messageId": "m", "parts": [] } },
    });
    let (_, resp) = post_json(router(), "/", &req).await;
    assert_eq!(resp["error"]["code"], -32004);
}

#[tokio::test]
async fn malformed_json_is_parse_error() {
    let (status, text) = post_raw(router(), "/", "{ not json ".to_string()).await;
    assert_eq!(status, StatusCode::OK);
    let resp: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(resp["error"]["code"], -32700);
    assert!(resp["id"].is_null());
}

#[tokio::test]
async fn missing_method_is_invalid_request() {
    let req = json!({ "jsonrpc": "2.0", "id": "6", "params": {} });
    let (_, resp) = post_json(router(), "/", &req).await;
    assert_eq!(resp["error"]["code"], -32600);
    assert_eq!(resp["id"], "6");
}

#[tokio::test]
async fn unknown_method_is_method_not_found() {
    let req = json!({ "jsonrpc": "2.0", "id": "7", "method": "foo/bar", "params": {} });
    let (_, resp) = post_json(router(), "/", &req).await;
    assert_eq!(resp["error"]["code"], -32601);
}

#[tokio::test]
async fn message_send_missing_message_is_invalid_params() {
    let req = json!({ "jsonrpc": "2.0", "id": "8", "method": "message/send", "params": {} });
    let (_, resp) = post_json(router(), "/", &req).await;
    assert_eq!(resp["error"]["code"], -32602);
}

#[tokio::test]
async fn message_send_continues_conversation_per_context() {
    use agent_framework_core::agent::Agent;
    use agent_framework_core::error::Result as CoreResult;
    use agent_framework_core::threads::AgentThread;
    use agent_framework_core::types::{AgentRunResponse, ChatMessage};

    /// Replies with how many messages of history it saw on the thread.
    struct HistoryAgent;
    #[async_trait::async_trait]
    impl Agent for HistoryAgent {
        async fn run(
            &self,
            messages: Vec<ChatMessage>,
            thread: Option<&mut AgentThread>,
        ) -> CoreResult<AgentRunResponse> {
            let thread = thread.expect("host must supply a thread");
            let prior = thread.list_messages().await?.len();
            let reply = ChatMessage::assistant(format!("prior:{prior}"));
            thread.on_new_messages(messages).await?;
            thread.on_new_messages(vec![reply.clone()]).await?;
            Ok(AgentRunResponse {
                messages: vec![reply],
                ..Default::default()
            })
        }
        fn id(&self) -> &str {
            "history"
        }
        fn name(&self) -> Option<&str> {
            Some("history")
        }
    }

    let agent: std::sync::Arc<dyn Agent> = std::sync::Arc::new(HistoryAgent);
    let router = A2ARouter::for_agent("history", agent, "http://localhost/").into_router();

    // Turn one: no context id yet; the host mints one.
    let (status, resp) = post_json(router.clone(), "/", &send_message("one")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        resp["result"]["artifacts"][0]["parts"][0]["text"],
        "prior:0"
    );
    let ctx = resp["result"]["contextId"].as_str().unwrap().to_string();

    // Turn two on the same context: the agent must see turn one's history.
    let follow_up = json!({
        "jsonrpc": "2.0",
        "id": "2",
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": "m2",
                "contextId": ctx,
                "parts": [{ "kind": "text", "text": "two" }],
            }
        }
    });
    let (status, resp) = post_json(router.clone(), "/", &follow_up).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        resp["result"]["artifacts"][0]["parts"][0]["text"],
        "prior:2"
    );

    // A different context starts fresh.
    let (_, resp) = post_json(router, "/", &send_message("three")).await;
    assert_eq!(
        resp["result"]["artifacts"][0]["parts"][0]["text"],
        "prior:0"
    );
}
