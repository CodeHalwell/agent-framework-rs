//! `LiveToolList`: change the agent's tool list *during* a run.
//!
//! Every run of the function-invocation loop carries a live, mutable copy of
//! the tool list, reachable from function middleware (and from a tool itself)
//! as `FunctionInvocationContext::tools`. Adding to it makes a tool visible
//! to the model on the **next** iteration; removing from it takes one away.
//! The in-flight batch of calls is unaffected either way.
//!
//! Two patterns are shown:
//!
//! - **Progressive disclosure.** Start with one `login` tool. Once it
//!   succeeds, the account-management tools appear. This keeps the tool
//!   schema the model sees small (and stops it inventing an authenticated
//!   call before there is a session).
//! - **One-shot tools.** Remove a tool after its first successful call, so a
//!   `send_invoice` cannot fire twice in one run however insistently the
//!   model asks.
//!
//! Compare with `AgentBuilder::tool_source` (see `mcp/mcp_first_class_tools.rs`),
//! which resolves a *fresh* list at the start of each run: `ToolSource` is
//! for "what tools exist right now", `LiveToolList` for "what this run has
//! unlocked so far".
//!
//! Runs fully offline against a scripted client.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example dynamic_tools
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::json;

/// A canned model working through a fixed wish-list of calls, reporting the
/// tool list it can see on each turn -- so the list growing and shrinking
/// mid-run is visible in the output.
///
/// A wish it cannot currently see is skipped (with a note) rather than
/// ending the run, which is roughly how a real model behaves: it only ever
/// sees the tools it was handed this iteration, so it moves on to whatever
/// else it can do.
#[derive(Clone)]
struct ScriptedClient {
    /// The tools it would like to call, in order.
    wishlist: Vec<&'static str>,
    /// How far through the wish-list it has got.
    cursor: Arc<AtomicUsize>,
    turn: Arc<AtomicUsize>,
}

impl ScriptedClient {
    fn new(wishlist: Vec<&'static str>) -> Self {
        Self {
            wishlist,
            cursor: Arc::new(AtomicUsize::new(0)),
            turn: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl ChatClient for ScriptedClient {
    async fn get_response(&self, _m: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        let turn = self.turn.fetch_add(1, Ordering::SeqCst);
        let mut visible: Vec<&str> = options.tools.iter().map(|t| t.name.as_str()).collect();
        visible.sort_unstable();
        println!("    turn {turn}: model can see {visible:?}");

        // Advance to the first wish that is actually available right now.
        loop {
            let i = self.cursor.fetch_add(1, Ordering::SeqCst);
            let Some(want) = self.wishlist.get(i) else {
                return Ok(ChatResponse::from_text(
                    "nothing left on my list that I can call -- done.",
                ));
            };
            if !visible.contains(want) {
                println!("      (wanted `{want}`, not available to me -- skipping)");
                continue;
            }
            println!("      -> calling `{want}`");
            return Ok(ChatResponse {
                messages: vec![Message::with_contents(
                    Role::assistant(),
                    vec![Content::FunctionCall(FunctionCallContent::new(
                        format!("call-{turn}"),
                        *want,
                        Some(FunctionArguments::Raw(json!({}).to_string())),
                    ))],
                )],
                finish_reason: Some(FinishReason::tool_calls()),
                ..Default::default()
            });
        }
    }

    async fn get_streaming_response(
        &self,
        _m: Vec<Message>,
        _o: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

fn tool(name: &'static str, description: &'static str) -> ToolDefinition {
    FunctionTool::new(
        name,
        description,
        json!({ "type": "object", "properties": {} }),
        move |_args| async move { Ok(json!({ "ok": true })) },
    )
    .into_definition()
}

/// Unlocks the account tools once `login` has been called successfully.
struct UnlockAfterLogin;

#[async_trait]
impl Middleware<FunctionInvocationContext> for UnlockAfterLogin {
    async fn process(
        &self,
        ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        let was_login = ctx.function_name == "login";
        // Run the call first: only a *successful* login should unlock.
        let ctx = next.run(ctx).await?;

        if was_login && ctx.result.is_some() {
            if let Some(tools) = &ctx.tools {
                // `add_tools` rejects a name already in the list, so guard
                // against a second login unlocking twice.
                if !tools.contains("view_balance") {
                    tools.add_tools([
                        tool("view_balance", "Show the signed-in account's balance."),
                        tool("send_invoice", "Send an invoice to a customer."),
                    ])?;
                    println!("    [middleware] login succeeded -> unlocked 2 account tool(s)");
                }
            }
        }
        Ok(ctx)
    }
}

/// Removes a tool once it has run, so it cannot fire twice in one run.
struct OneShot {
    tool_name: &'static str,
}

#[async_trait]
impl Middleware<FunctionInvocationContext> for OneShot {
    async fn process(
        &self,
        ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        let matched = ctx.function_name == self.tool_name;
        let ctx = next.run(ctx).await?;
        if matched && ctx.result.is_some() {
            if let Some(tools) = &ctx.tools {
                tools.remove_tools([self.tool_name]);
                println!(
                    "    [middleware] `{}` is one-shot -> removed",
                    self.tool_name
                );
            }
        }
        Ok(ctx)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("== progressive disclosure: account tools appear only after login ==\n");

    // The model tries `view_balance` before logging in (it cannot see it),
    // then logs in, then tries again (now it can).
    let client = ScriptedClient::new(vec!["view_balance", "login", "view_balance"]);
    let agent = Agent::builder(client)
        .name("banker")
        .tool(tool("login", "Sign in to the account."))
        .function_middleware(Arc::new(UnlockAfterLogin))
        .build();
    let response = agent.run_once("check my balance").await?;
    println!("\n  final: {}\n", response.text());

    println!("== one-shot: a tool that cannot fire twice in a run ==\n");

    // The model tries to send the invoice three times; only the first lands.
    let client = ScriptedClient::new(vec!["send_invoice", "send_invoice", "send_invoice"]);
    let agent = Agent::builder(client)
        .name("billing")
        .tool(tool("send_invoice", "Send an invoice to a customer."))
        .function_middleware(Arc::new(OneShot {
            tool_name: "send_invoice",
        }))
        .build();
    let response = agent.run_once("send the invoice").await?;
    println!("\n  final: {}", response.text());

    println!(
        "\nnote: mutations land on the *next* model iteration, not the in-flight\n\
         batch of calls -- so a parallel batch of three `send_invoice` calls would\n\
         all still run. Use `ApprovalMode::AlwaysRequire` (see the `approvals`\n\
         example) when the guarantee has to hold within a batch too."
    );

    Ok(())
}
