//! Writing your own `ChatClient`: everything in the framework -- agents,
//! tools, workflows, orchestrations, hosting -- is built on this one trait,
//! so an in-house model, a gateway, or a fake for tests plugs into all of it
//! by implementing three methods.
//!
//! ```ignore
//! #[async_trait]
//! trait ChatClient: Send + Sync {
//!     async fn get_response(&self, messages: Vec<Message>, options: ChatOptions)
//!         -> Result<ChatResponse>;
//!     async fn get_streaming_response(&self, messages: Vec<Message>, options: ChatOptions)
//!         -> Result<ChatStream>;
//!     fn model(&self) -> Option<&str> { None }   // has a default
//! }
//! ```
//!
//! What a real implementation owes its callers, and what this example shows:
//!
//! 1. **Honour `ChatOptions`.** At minimum `model`, `temperature`,
//!    `max_tokens`, `tools`, `tool_choice`, and `response_format` -- silently
//!    dropping one is how a caller ends up debugging a temperature that never
//!    took effect. Ignore `options.session`: it is a framework-internal side
//!    channel and never goes on the wire.
//! 2. **Emit tool calls as `Content::FunctionCall`.** Do that and the
//!    function-invocation loop works for free -- you do not implement tool
//!    calling yourself.
//! 3. **Report `usage_details`.** Observability and cost tracking read it.
//! 4. **Classify errors.** Return `ServiceInvalidAuth` / `ServiceInvalidRequest`
//!    / `ServiceContentFilter` / `ServiceStatus` rather than a flat string,
//!    so `RetryingChatClient` retries what is transient and nothing else
//!    (see the `error_handling` example).
//! 5. **Stream for real.** `get_streaming_response` should yield chunks as
//!    they arrive, not split a buffered answer afterwards.
//!
//! The client below is deliberately not backed by a network service, so the
//! example runs offline; the shape is what matters. For a real HTTP client,
//! read `crates/agent-framework-ollama` -- it is the smallest one in the
//! workspace.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example custom_chat_client
//! ```

use std::time::Duration;

use agent_framework::prelude::*;
use agent_framework::types::{FunctionArguments, UsageContent, UsageDetails};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

/// A toy backend standing in for whatever you are actually calling.
///
/// It answers arithmetic by asking for the `add` tool when one is offered,
/// and otherwise echoes -- enough to show tool calling, options handling,
/// usage reporting, error classification, and streaming.
struct MyChatClient {
    model: String,
    /// Set to simulate the backend rejecting the credentials.
    fail_auth: bool,
}

impl MyChatClient {
    fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            fail_auth: false,
        }
    }

    /// Turn the framework's request into whatever your backend wants. A real
    /// client builds its JSON body here; this one just summarises.
    fn build_request(&self, messages: &[Message], options: &ChatOptions) -> String {
        // `options.model` wins over the client's default: a per-run override
        // must be able to reach a different deployment.
        let model = options.model.as_deref().unwrap_or(&self.model);
        let mut parts = vec![format!("model={model}")];
        if let Some(t) = options.temperature {
            parts.push(format!("temperature={t}"));
        }
        if let Some(m) = options.max_tokens {
            parts.push(format!("max_tokens={m}"));
        }
        if let Some(choice) = &options.tool_choice {
            parts.push(format!("tool_choice={}", choice.as_str()));
        }
        if !options.tools.is_empty() {
            let names: Vec<&str> = options.tools.iter().map(|t| t.name.as_str()).collect();
            parts.push(format!("tools={names:?}"));
        }
        if options.response_format.is_some() {
            parts.push("response_format=json_schema".to_string());
        }
        // Deliberately NOT forwarded: `options.session` is a framework-internal
        // side channel, and a provider converter must ignore it.
        parts.push(format!("messages={}", messages.len()));
        parts.join(" ")
    }

    /// What the backend "returns" for a given transcript.
    fn answer(&self, messages: &[Message], options: &ChatOptions) -> ChatResponse {
        let request = self.build_request(messages, options);
        println!("    [wire] {request}");

        // Already have a tool result? Then this is the second pass of the
        // function-invocation loop: produce a final answer.
        if let Some(result) = messages
            .iter()
            .flat_map(|m| m.contents.iter())
            .find_map(|c| match c {
                Content::FunctionResult(r) => r.result.clone(),
                _ => None,
            })
        {
            return with_usage(ChatResponse::from_text(format!("The answer is {result}.")));
        }

        let last = messages.last().map(Message::text).unwrap_or_default();

        // (2) Ask for a tool by emitting `Content::FunctionCall`. The
        // framework executes it and calls us again -- we implement nothing.
        if last.contains('+') && options.tools.iter().any(|t| t.name == "add") {
            let mut nums = last
                .split(|c: char| !c.is_ascii_digit())
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse::<i64>().ok());
            let (a, b) = (nums.next().unwrap_or(0), nums.next().unwrap_or(0));
            return with_usage(ChatResponse {
                messages: vec![Message::with_contents(
                    Role::assistant(),
                    vec![Content::FunctionCall(FunctionCallContent::new(
                        "call-1",
                        "add",
                        Some(FunctionArguments::Raw(json!({"a": a, "b": b}).to_string())),
                    ))],
                )],
                finish_reason: Some(FinishReason::tool_calls()),
                ..Default::default()
            });
        }

        with_usage(ChatResponse::from_text(format!(
            "You said: {last}. (I am a toy backend.)"
        )))
    }
}

/// (3) Report token usage. Observability spans and the `otel-metrics`
/// histograms read this; leave it `None` and cost tracking goes dark.
fn with_usage(mut response: ChatResponse) -> ChatResponse {
    let output = response.text().split_whitespace().count() as u64;
    response.usage_details = Some(UsageDetails {
        input_token_count: Some(12),
        output_token_count: Some(output),
        total_token_count: Some(12 + output),
        ..Default::default()
    });
    response.model = Some("my-model-v1".into());
    response
}

#[async_trait]
impl ChatClient for MyChatClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        // (4) Classify failures by kind, not as one opaque string.
        if self.fail_auth {
            return Err(Error::service_invalid_auth(
                "401 from my-backend: the API key was rejected",
            ));
        }
        Ok(self.answer(&messages, &options))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        if self.fail_auth {
            return Err(Error::service_invalid_auth(
                "401 from my-backend: the API key was rejected",
            ));
        }
        let response = self.answer(&messages, &options);

        // (5) A real client decodes chunks off the socket as they arrive.
        // `stream::unfold` awaits *inside* the generator so each item is only
        // produced once its (simulated) latency has elapsed -- unlike
        // `stream::iter` over a pre-built Vec, which resolves instantly and
        // only looks like streaming.
        let words: Vec<String> = response
            .text()
            .split(' ')
            .map(|w| format!("{w} "))
            .collect();
        let usage = response.usage_details.clone();
        let stream = futures::stream::unfold(
            (words.into_iter(), usage),
            |(mut remaining, usage)| async move {
                match remaining.next() {
                    Some(word) => {
                        tokio::time::sleep(Duration::from_millis(40)).await;
                        Some((Ok(ChatResponseUpdate::text(word)), (remaining, usage)))
                    }
                    // Usage lands on the final chunk, as most providers do.
                    None => usage.map(|u| {
                        let update = ChatResponseUpdate {
                            contents: vec![Content::Usage(UsageContent { details: u })],
                            finish_reason: Some(FinishReason::stop()),
                            ..Default::default()
                        };
                        (Ok(update), (remaining, None))
                    }),
                }
            },
        );
        Ok(stream.boxed())
    }

    /// The default model, used when `ChatOptions::model` is unset. The agent
    /// builder reads it to fill in `chat_options.model`.
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("== 1. used directly as a ChatClient ==\n");
    let client = MyChatClient::new("my-model-v1");
    let response = client
        .get_response(
            vec![Message::user("hello there")],
            ChatOptions::new().with_temperature(0.4).with_max_tokens(256),
        )
        .await?;
    println!("    {}", response.text());
    println!("    usage: {:?}", response.usage_details);

    println!("\n== 2. driving an Agent, with tool calling for free ==\n");

    let add = FunctionTool::new(
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

    let agent = Agent::builder(MyChatClient::new("my-model-v1"))
        .name("calculator")
        .instructions("Use the add tool for arithmetic.")
        .tool(add)
        .build();
    let response = agent.run_once("what is 17 + 25?").await?;
    println!("    {}", response.text());
    println!(
        "\n    Two round trips: the client asked for `add`, the framework ran it\n    \
         and called back with the result. The client implements no tool logic."
    );

    println!("\n== 3. streaming ==\n");
    use std::io::Write as _;
    let mut stream = MyChatClient::new("my-model-v1")
        .get_streaming_response(vec![Message::user("stream me something")], ChatOptions::new())
        .await?;
    print!("    ");
    while let Some(update) = stream.next().await {
        let update = update?;
        for content in &update.contents {
            match content {
                Content::Text(t) => {
                    print!("{}", t.text);
                    let _ = std::io::stdout().flush();
                }
                Content::Usage(u) => println!("\n    [usage chunk] {:?}", u.details),
                _ => {}
            }
        }
    }

    println!("\n== 4. composing with the framework's client wrappers ==\n");

    // Because it is just a `ChatClient`, every wrapper in the workspace
    // applies -- none of them know or care what backend is underneath.
    let failing = MyChatClient {
        model: "my-model-v1".into(),
        fail_auth: true,
    };
    let wrapped = ObservableChatClient::new(
        RetryingChatClient::new(failing).with_policy(RetryPolicy {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            jitter: 0.0,
            ..RetryPolicy::default()
        }),
        "my-backend",
    );
    match wrapped
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
    {
        Ok(r) => println!("    unexpected success: {}", r.text()),
        Err(e) => println!("    {e}"),
    }
    println!(
        "\n    Not retried, and correctly so: the client returned\n    \
         `ServiceInvalidAuth`, which the default policy knows is not transient.\n    \
         Had it returned a bare `Error::Service(\"401 ...\")`, the policy would\n    \
         have had to guess from the message -- which is why step (4) matters."
    );

    println!(
        "\nnext: `ObservableChatClient` gives it OTel GenAI spans, `AgentHost`\n\
         serves it over HTTP, and it can be a participant in any workflow or\n\
         orchestration -- all without the backend knowing any of it exists."
    );

    Ok(())
}
