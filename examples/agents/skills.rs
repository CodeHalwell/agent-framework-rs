//! Skills: progressive-disclosure capability packages (`Skill` +
//! `SkillsProvider`, from `agent_framework_core::skills`). A `Skill` bundles
//! a short always-visible `description`, a longer `instructions` body, and
//! named `resources`. `SkillsProvider` is a `ContextProvider` that injects a
//! compact catalog (name + description only) into every run, plus two
//! framework-generated tools -- `load_skill` and `read_skill_resource` --
//! that let the model pull in a skill's full detail only when it decides the
//! skill is relevant, instead of paying that token cost on every turn. Once
//! loaded, a skill's full instructions are injected on every later run of
//! the same provider.
//!
//! Runs fully offline against a scripted client that "decides" to load a
//! skill and read one of its resources, recording what instructions and
//! tools each model call actually saw -- no API key or network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example skills
//! ```

use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::json;

/// A scripted stand-in for a model: returns queued responses in order and
/// records the system prompt each call carried (the agent's instructions plus
/// everything the `SkillsProvider` injected, delivered as a leading system
/// message), so progressive disclosure is observable from the outside.
#[derive(Clone)]
struct ScriptedClient {
    responses: Arc<Mutex<Vec<ChatResponse>>>,
    seen_instructions: Arc<Mutex<Vec<String>>>,
}

impl ScriptedClient {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            seen_instructions: Arc::new(Mutex::new(Vec::new())),
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
        let system_prompt: String = messages
            .iter()
            .filter(|m| m.role == Role::system())
            .map(|m| m.text())
            .collect::<Vec<_>>()
            .join("\n");
        self.seen_instructions.lock().unwrap().push(system_prompt);
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            Ok(ChatResponse::from_text("(script exhausted)"))
        } else {
            Ok(responses.remove(0))
        }
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

/// An assistant message that calls `name` with `args` (the framework's
/// function-invocation loop executes it and calls the model again).
fn tool_call(call_id: &str, name: &str, args: serde_json::Value) -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                call_id,
                name,
                Some(FunctionArguments::Raw(args.to_string())),
            ))],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Two skills: only their one-line descriptions are visible up front.
    let skills = vec![
        Skill::new("code-review", "Review code changes for defects and style")
            .with_instructions(
                "Check error handling first, then naming, then test coverage. \
                 Consult the 'checklist' resource for the full rubric.",
            )
            .with_resource(
                "checklist",
                "1. unwrap()/panic! in library code  2. missing error context  3. dead code",
            ),
        Skill::new("release-notes", "Draft release notes from a changelog")
            .with_instructions("Group changes by area; lead with breaking changes."),
    ];

    // The script plays the model's side of one review request: it decides the
    // code-review skill is relevant, loads it, reads its checklist resource,
    // and then answers. A second turn shows the loaded skill persisting.
    let client = ScriptedClient::new(vec![
        tool_call("c1", "load_skill", json!({"skill_name": "code-review"})),
        tool_call(
            "c2",
            "read_skill_resource",
            json!({"skill_name": "code-review", "resource_name": "checklist"}),
        ),
        ChatResponse::from_text(
            "Review: `parse_config` swallows the I/O error -- return it with context instead.",
        ),
        ChatResponse::from_text("Second file looks fine per the same checklist."),
    ]);
    let seen = client.seen_instructions.clone();

    let agent = Agent::builder(client)
        .name("reviewer")
        .instructions("You are a meticulous engineering assistant.")
        .context_provider(Arc::new(SkillsProvider::new(skills)))
        .build();

    let mut session = agent.create_session();
    println!("-- turn 1: the model loads the skill it needs --");
    let r1 = agent
        .run(
            vec![Message::user("Please review src/config.rs for issues.")],
            Some(&mut session),
        )
        .await?;
    println!("assistant: {}\n", r1.text());

    println!("-- turn 2: the loaded skill persists across turns --");
    let r2 = agent
        .run(vec![Message::user("And src/cli.rs?")], Some(&mut session))
        .await?;
    println!("assistant: {}\n", r2.text());

    // What each model call actually saw, proving progressive disclosure:
    // the catalog is always present; full instructions appear only after
    // `load_skill` ran.
    let seen = seen.lock().unwrap();
    println!("-- instructions visible to each model call --");
    for (i, instructions) in seen.iter().enumerate() {
        let catalog = instructions.contains("code-review: Review code changes");
        let full = instructions.contains("Full instructions for skill 'code-review'");
        println!(
            "call {}: catalog entry: {catalog:<5}  full instructions: {full}",
            i + 1
        );
    }
    println!(
        "(calls 2-3 are turn 1's tool loop: there the model already holds the\n\
         instructions as the load_skill tool result; system-prompt injection\n\
         starts with the next run -- call 4)"
    );
    assert!(
        !seen[0].contains("Check error handling first"),
        "the full instructions must not be visible before load_skill"
    );
    assert!(
        seen.last().unwrap().contains("Check error handling first"),
        "once loaded, the full instructions are injected on every later call"
    );
    // The un-loaded skill stays summarized forever.
    assert!(!seen.last().unwrap().contains("Group changes by area"));

    println!(
        "\n(the 'release-notes' skill was never loaded, so its full instructions\n\
         never entered the context -- only its one-line catalog entry did)"
    );
    Ok(())
}
