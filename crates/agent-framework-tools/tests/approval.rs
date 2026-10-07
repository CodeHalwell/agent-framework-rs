//! The shell function inside the real function-invocation loop: with the
//! default approval mode nothing runs until a human approves the exact call.
#![cfg(unix)]

mod common;

use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::types::{Content, FunctionArguments, FunctionCallContent, Role};
use agent_framework_tools::shell::{LocalShellTool, ShellMode};
use async_trait::async_trait;
use common::TempDir;
use futures::StreamExt;

#[derive(Clone)]
struct ScriptedClient {
    responses: Arc<Mutex<Vec<ChatResponse>>>,
    seen: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl ScriptedClient {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl ChatClient for ScriptedClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(messages);
        let mut queue = self.responses.lock().unwrap();
        Ok(if queue.is_empty() {
            ChatResponse::from_text("done")
        } else {
            queue.remove(0)
        })
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<agent_framework_core::client::ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let updates: Vec<Result<ChatResponseUpdate>> = resp
            .messages
            .into_iter()
            .map(|m| {
                Ok(ChatResponseUpdate {
                    contents: m.contents,
                    role: Some(m.role),
                    ..Default::default()
                })
            })
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }
}

fn shell_call(command: &str) -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_shell",
                "run_shell",
                Some(FunctionArguments::Raw(
                    serde_json::json!({ "command": command }).to_string(),
                )),
            ))],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

async fn request_approval(
    approved: bool,
) -> (TempDir, std::path::PathBuf, ScriptedClient, ChatResponse) {
    let dir = TempDir::new("approval");
    let marker = dir.path().join("ran");
    let command = format!("touch '{}' && echo created", marker.display());
    let tool = LocalShellTool::builder()
        .mode(ShellMode::Stateless)
        .build()
        .unwrap();
    let scripted = ScriptedClient::new(vec![
        shell_call(&command),
        ChatResponse::from_text("finished"),
    ]);
    let client = FunctionInvokingChatClient::new(scripted.clone());
    let options = ChatOptions::new().with_tool(tool.as_function());

    let first = client
        .get_response(vec![Message::user("make the marker")], options.clone())
        .await
        .unwrap();
    let requests = first.user_input_requests();
    assert_eq!(requests.len(), 1, "the shell call must wait for approval");
    assert_eq!(requests[0].function_call.name, "run_shell");
    assert!(!marker.exists(), "the command ran before approval");

    let response = requests[0].create_response(approved);
    let mut conversation = vec![Message::user("make the marker")];
    conversation.extend(first.messages.clone());
    conversation.push(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(response)],
    ));
    let second = client.get_response(conversation, options).await.unwrap();
    (dir, marker, scripted, second)
}

#[tokio::test]
async fn approved_call_runs_once_and_returns_the_model_text() {
    let (_dir, marker, scripted, second) = request_approval(true).await;
    assert!(marker.exists(), "approved command did not run");
    assert_eq!(second.text(), "finished");
    let seen = scripted.seen.lock().unwrap();
    let results: Vec<String> = seen
        .last()
        .unwrap()
        .iter()
        .flat_map(|m| m.contents.iter())
        .filter_map(|c| match c {
            Content::FunctionResult(r) => r.result.as_ref().map(|v| v.to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 1);
    assert!(results[0].contains("created"), "{results:?}");
    assert!(results[0].contains("exit_code: 0"), "{results:?}");
}

#[tokio::test]
async fn rejected_call_never_runs() {
    let (_dir, marker, _scripted, second) = request_approval(false).await;
    assert!(!marker.exists(), "rejected command ran");
    assert_eq!(second.text(), "finished");
}
