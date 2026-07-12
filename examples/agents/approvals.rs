//! Human-in-the-loop tool approval: a tool marked `ApprovalMode::AlwaysRequire`
//! makes the function-invocation loop pause instead of executing it -- the
//! caller inspects the pending request, approves (or rejects) it, and
//! resubmits on the same thread to continue.
//!
//! The flow (not the tool itself) is the point of this example: run -> look
//! at `user_input_requests()` -> approve via `FunctionApprovalResponseContent`
//! -> repeat until a final answer comes back.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example approvals
//! ```

use agent_framework::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIClient::from_env("gpt-4o-mini")?;

    // A tool that always requires a human sign-off before it runs.
    // `ApprovalMode` lives on the tool itself; the function-invocation loop
    // checks it for every call before executing.
    let delete_file = AiFunction::new(
        "delete_file",
        "Permanently delete a file by name.",
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }),
        |args| async move {
            let path = args["path"].as_str().unwrap_or("unknown").to_string();
            Ok(json!({ "deleted": path }))
        },
    )
    .with_approval_mode(ApprovalMode::AlwaysRequire)
    .into_definition();

    let agent = ChatAgent::builder(client)
        .name("file-assistant")
        .instructions("You help manage files. Use tools when the user asks for a file action.")
        .tool(delete_file)
        .build();

    let mut thread = agent.get_new_thread();
    let mut input = vec![ChatMessage::user("Please delete scratch.txt")];

    // Drive the approve/resubmit loop until a final, non-approval answer
    // comes back (bounded so a misbehaving model can't spin forever).
    for _ in 0..5 {
        let response = agent.run(input.clone(), Some(&mut thread)).await?;
        let requests = response.user_input_requests();

        if requests.is_empty() {
            println!("{}", response.text());
            return Ok(());
        }

        for req in &requests {
            println!(
                "approval requested: {}({:?})",
                req.function_call.name, req.function_call.arguments
            );
        }

        // Approve every pending call. In a real application this is where
        // you'd prompt a human (CLI, chat UI, ticket queue, ...); rejecting
        // instead is `req.create_response(false)`.
        let approvals: Vec<Content> = requests
            .iter()
            .map(|req| Content::FunctionApprovalResponse(req.create_response(true)))
            .collect();
        input = vec![ChatMessage::with_contents(Role::user(), approvals)];
    }

    println!("gave up after 5 rounds without a final answer");
    Ok(())
}
