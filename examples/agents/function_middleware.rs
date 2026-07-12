//! Function middleware: wraps every local tool call the function-invocation
//! loop executes. A middleware receives an owned `FunctionInvocationContext`
//! (the tool name and its parsed arguments) and a `Next` continuation: it can
//! rewrite `ctx.arguments` before calling `next.run(ctx)`, then observe (or
//! override) `ctx.result` afterward.
//!
//! Runs fully offline against a canned client that emits one function call
//! (`add`), then a final text answer -- the same scripted round-trip pattern
//! used by the core crate's own tool-loop tests.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example function_middleware
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::json;

/// Rewrites `add`'s `a` argument (multiplying it by 10) before execution, and
/// prints the tool's real result afterward.
struct ArgRewriteMiddleware;

#[async_trait]
impl Middleware<FunctionInvocationContext> for ArgRewriteMiddleware {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        println!(
            "  [function-middleware] {} called with {}",
            ctx.function_name, ctx.arguments
        );
        if ctx.function_name == "add" {
            if let Some(obj) = ctx.arguments.as_object_mut() {
                if let Some(a) = obj.get("a").and_then(serde_json::Value::as_i64) {
                    obj.insert("a".to_string(), json!(a * 10));
                }
            }
            println!(
                "  [function-middleware] rewritten arguments: {}",
                ctx.arguments
            );
        }

        let ctx = next.run(ctx).await?;
        println!("  [function-middleware] result: {:?}", ctx.result);
        Ok(ctx)
    }
}

/// A scripted client: first response asks to call `add`, second is the final
/// answer. Mirrors the round-trip the automatic function-invocation loop
/// drives against a real model.
#[derive(Clone)]
struct MockClient {
    responses: Arc<std::sync::Mutex<Vec<ChatResponse>>>,
}

impl MockClient {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(std::sync::Mutex::new(responses)),
        }
    }
}

#[async_trait]
impl ChatClient for MockClient {
    async fn get_response(
        &self,
        _messages: Vec<ChatMessage>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut resps = self.responses.lock().unwrap();
        if resps.is_empty() {
            Ok(ChatResponse::from_text("(script exhausted)"))
        } else {
            Ok(resps.remove(0))
        }
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<ChatMessage>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let call = FunctionCallContent::new(
        "call_1",
        "add",
        Some(FunctionArguments::Raw(json!({"a": 2, "b": 3}).to_string())),
    );
    let ask = ChatResponse {
        messages: vec![ChatMessage::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    };
    let answer = ChatResponse::from_text("Done -- check the tool result above.");
    let client = MockClient::new(vec![ask, answer]);

    let add = AiFunction::new(
        "add",
        "Add two integers.",
        json!({
            "type": "object",
            "properties": { "a": {"type":"integer"}, "b": {"type":"integer"} },
            "required": ["a", "b"]
        }),
        |args| async move {
            let a = args["a"].as_i64().unwrap_or(0);
            let b = args["b"].as_i64().unwrap_or(0);
            Ok(json!(a + b))
        },
    )
    .into_definition();

    let agent = ChatAgent::builder(client)
        .name("calculator")
        .tool(add)
        .function_middleware(Arc::new(ArgRewriteMiddleware))
        .build();

    println!("asking the agent to add 2 and 3 (middleware rewrites 'a' to 20 first)...");
    let response = agent.run_once("Please add 2 and 3.").await?;
    println!("final assistant message (scripted): {}", response.text());

    Ok(())
}
