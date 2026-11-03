//! Container-file citations announced in an MCP tool result's `_meta`
//! become `container_file_citation` annotations on the output text that
//! names the files (upstream `_container_file_citations_from_function_result`).

mod common;

use std::sync::Arc;

use agent_framework_core::client::ChatClient;
use agent_framework_core::prelude::*;
use agent_framework_core::types::{ChatResponse, FunctionCallContent};
use agent_framework_foundry_hosting::ResponsesHostServer;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Json, Router};
use common::*;
use serde_json::{json, Value};

/// An MCP server whose `make_chart` result announces its files in `_meta`.
async fn mcp(Json(body): Json<Value>) -> axum::response::Response {
    let Some(id) = body.get("id").cloned() else {
        return StatusCode::ACCEPTED.into_response();
    };
    let result = match body["method"].as_str() {
        Some("initialize") => json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "files", "version": "1"},
        }),
        Some("tools/list") => json!({"tools": [
            {"name": "make_chart", "description": "Make a chart.", "inputSchema": {"type": "object"}},
        ]}),
        Some("tools/call") => json!({
            "content": [{
                "type": "text",
                "text": "saved",
                "_meta": {"container_file_citations": [
                    {"container_id": "cntr_2", "file_id": "cfile_notes", "filename": "notes.txt"},
                    {"file_id": "cfile_bad", "filename": "missing-container.txt"},
                ]},
            }],
            "_meta": {
                "container_id": "cntr_1",
                // A JSON-encoded list, with the result's container as fallback.
                "container_file_citations": json!([
                    {"file_id": "cfile_chart", "filename": "chart.png"},
                    {"file_id": "cfile_unused", "filename": "unused.csv"},
                ]).to_string(),
            },
        }),
        _ => {
            return Json(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "Method not found"}}))
                .into_response()
        }
    };
    Json(json!({"jsonrpc": "2.0", "id": id, "result": result})).into_response()
}

/// Calls `make_chart`, then names the files it made.
struct Charts;

#[async_trait::async_trait]
impl ChatClient for Charts {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatResponse> {
        let called = messages
            .iter()
            .flat_map(|m| &m.contents)
            .any(|c| matches!(c, Content::FunctionResult(_)));
        let content = if called {
            Content::text("📊 See chart.png and notes.txt (not old_chart.png or chart.png.bak).")
        } else {
            Content::FunctionCall(FunctionCallContent::new(
                "c1",
                "make_chart",
                Some(agent_framework_core::types::FunctionArguments::Raw(
                    "{}".into(),
                )),
            ))
        };
        Ok(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), vec![content])],
            ..Default::default()
        })
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> agent_framework_core::error::Result<agent_framework_core::client::ChatStream> {
        let response = self.get_response(messages, options).await?;
        let updates: Vec<_> = response.messages[0]
            .contents
            .iter()
            .cloned()
            .map(|c| {
                Ok(agent_framework_core::types::ChatResponseUpdate {
                    contents: vec![c],
                    role: Some(Role::assistant()),
                    ..Default::default()
                })
            })
            .collect();
        Ok(Box::pin(futures::stream::iter(updates)))
    }
}

#[tokio::test]
async fn mcp_result_container_files_become_text_annotations() {
    let app = Router::new().route(
        "/mcp",
        axum::routing::post(mcp).delete(|| async { StatusCode::OK }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let tools =
        agent_framework_mcp::McpStreamableHttpTool::new("files", format!("http://{addr}/mcp"))
            .tool_definitions()
            .await
            .unwrap();
    let mut builder = Agent::builder(Charts).name("charts");
    for tool in tools {
        builder = builder.tool(tool);
    }
    let agent: Arc<dyn SupportsAgentRun> = Arc::new(builder.build());
    let router = ResponsesHostServer::agent(agent).into_router();

    for stream in [false, true] {
        let (status, _, body) = send(
            &router,
            post("/responses", json!({"input": "chart it", "stream": stream})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let response: Value = if stream {
            sse_events(&body)
                .into_iter()
                .find(|(k, _)| k == "response.completed")
                .expect("a completed event")
                .1["response"]
                .clone()
        } else {
            serde_json::from_str(&body).unwrap()
        };
        let message = response["output"]
            .as_array()
            .unwrap()
            .iter()
            .find(|i| i["type"] == "message")
            .unwrap_or_else(|| panic!("a message: {response}"));
        let text = message["content"][0]["text"].as_str().unwrap();
        let at = |needle: &str| text[..text.find(needle).unwrap()].chars().count();
        assert_eq!(
            message["content"][0]["annotations"],
            json!([
                {"type": "container_file_citation", "container_id": "cntr_1", "file_id": "cfile_chart",
                 "filename": "chart.png", "start_index": at("chart.png"), "end_index": at("chart.png") + 9},
                {"type": "container_file_citation", "container_id": "cntr_2", "file_id": "cfile_notes",
                 "filename": "notes.txt", "start_index": at("notes.txt"), "end_index": at("notes.txt") + 9},
            ]),
            "stream={stream}: {response}"
        );
    }
}
