//! `A2AClient`: the protocol surface underneath `A2AAgent`.
//!
//! `A2AAgent` (see `a2a/a2a_client.rs`) is the convenient shape -- it hides
//! A2A behind the `SupportsAgentRun` trait, so a remote agent drops into a
//! workflow or orchestration like a local one. `A2AClient` is what it is
//! built on, and you want it directly when you care about the *task*: its id,
//! its state, its artifacts, cancelling it, or being called back when it
//! finishes.
//!
//! The full client surface:
//!
//! | Method | JSON-RPC |
//! | --- | --- |
//! | `get_agent_card()` | `GET /.well-known/agent-card.json` |
//! | `get_extended_card()` | `agent/getAuthenticatedExtendedCard` |
//! | `send_message(params)` | `message/send` |
//! | `send_message_stream(params)` | `message/stream` (SSE) |
//! | `get_task(id)` | `tasks/get` |
//! | `cancel_task(id)` | `tasks/cancel` |
//! | `set_push_notification_config(..)` | `tasks/pushNotificationConfig/set` |
//! | `get_push_notification_config(..)` | `tasks/pushNotificationConfig/get` |
//! | `resubscribe(id)` | `tasks/resubscribe` (SSE) |
//!
//! Note `send_message` returns a `SendMessageResult`, which is *either* a
//! `Message` (the server answered immediately) or a `Task` (it accepted work
//! and will report progress). Handling both is not optional -- which of the
//! two you get is the server's choice, not yours.
//!
//! Offline and self-terminating: serves this workspace's own `A2ARouter` on
//! an ephemeral port with a canned agent behind it, then drives the client
//! against it over a real socket.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example a2a_task_lifecycle
//! ```

use std::net::SocketAddr;
use std::time::Duration;

use agent_framework::a2a::{
    A2AClient, Message as A2AMessage, MessageSendConfiguration, MessageSendParams, Part,
    PushNotificationConfig, SendMessageResult, TextPart,
};
use agent_framework::hosting::a2a::A2ARouter;
use agent_framework::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;

/// A canned model, so no credentials are needed.
#[derive(Clone)]
struct CannedClient;

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        let last = messages.last().map(Message::text).unwrap_or_default();
        Ok(ChatResponse::from_text(format!(
            "Considered '{last}' and concluded: yes, probably."
        )))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let updates = response.messages.into_iter().map(|m| {
            Ok(ChatResponseUpdate {
                contents: m.contents,
                role: Some(m.role),
                ..Default::default()
            })
        });
        Ok(futures::stream::iter(updates).boxed())
    }
}

/// A user-role A2A message carrying one text part.
fn text_message(text: &str) -> A2AMessage {
    A2AMessage::user(vec![Part::Text(TextPart {
        text: text.to_string(),
        metadata: None,
    })])
}

#[tokio::main]
async fn main() -> Result<()> {
    // --- serve an A2A agent in-process ---------------------------------
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|e| Error::Configuration(format!("bind failed: {e}")))?;
    let addr: SocketAddr = listener
        .local_addr()
        .map_err(|e| Error::Configuration(e.to_string()))?;
    let base_url = format!("http://{addr}/");

    let agent = Agent::builder(CannedClient)
        .name("analyst")
        .description("Weighs a question and gives a short verdict.")
        .build();
    let router = A2ARouter::for_agent("analyst", agent, &base_url)
        .version("2.1.0")
        .into_router();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    println!("A2A server listening at {base_url}\n");

    // --- the client ----------------------------------------------------
    let client = A2AClient::from_url(&base_url)
        // Auth, when the server wants it. `with_header` covers anything else
        // (a tenant id, a trace header); both return Result because a bad
        // header value should fail here, not on every later request.
        .with_bearer_token("demo-token")?
        .with_timeout(Duration::from_secs(10))?;

    println!("== 1. discovery: the agent card ==\n");

    let card = client.get_agent_card().await?;
    println!("  name:         {}", card.name);
    println!("  description:  {}", card.description);
    println!("  version:      {}", card.version);
    println!("  protocol:     {:?}", card.protocol_version);
    println!("  transport:    {:?}", card.preferred_transport);
    println!(
        "  capabilities: streaming={} push={} history={}",
        card.capabilities.streaming,
        card.capabilities.push_notifications,
        card.capabilities.state_transition_history
    );
    println!(
        "  skills:       {:?}",
        card.skills.iter().map(|s| &s.id).collect::<Vec<_>>()
    );
    // The card is cached after the first fetch, so later calls are free.
    println!(
        "  cached:       {}",
        client.cached_agent_card().await.is_some()
    );
    println!("  rpc endpoint: {}", client.rpc_url().await);

    println!("\n== 2. message/send ==\n");

    // `configuration` is advisory: `blocking` asks the server to hold the
    // response until the task settles rather than returning `Submitted`
    // immediately. Not every server honours it.
    let params = MessageSendParams {
        message: text_message("Is a hot-water bottle a reasonable substitute for heating?"),
        configuration: Some(MessageSendConfiguration {
            accepted_output_modes: Some(vec!["text".to_string()]),
            blocking: Some(true),
            history_length: Some(10),
        }),
        metadata: None,
    };

    // Either shape is valid. Branching on it is the whole reason this returns
    // an enum rather than a `Task`.
    let task_id = match client.send_message(params).await? {
        SendMessageResult::Message(message) => {
            println!("  server answered inline (no task):");
            for part in &message.parts {
                if let Part::Text(t) = part {
                    println!("    {}", t.text);
                }
            }
            None
        }
        SendMessageResult::Task(task) => {
            println!("  server created a task:");
            println!("    id:       {}", task.id);
            println!("    context:  {}", task.context_id);
            println!(
                "    state:    {:?} (terminal={})",
                task.status.state,
                task.status.state.is_terminal()
            );
            if let Some(message) = &task.status.message {
                for part in &message.parts {
                    if let Part::Text(t) = part {
                        println!("    reply:    {}", t.text);
                    }
                }
            }
            println!(
                "    history:  {} message(s)",
                task.history.as_ref().map_or(0, Vec::len)
            );
            // The answer arrives as an *artifact*, not as status.message: A2A
            // separates "how the task is going" from "what it produced".
            for artifact in task.artifacts.iter().flatten() {
                for part in &artifact.parts {
                    if let Part::Text(t) = part {
                        println!(
                            "    artifact `{}`: {}",
                            artifact.name.as_deref().unwrap_or(&artifact.artifact_id),
                            t.text
                        );
                    }
                }
            }
            Some(task.id)
        }
    };

    println!("\n== 3. tasks/get: polling a task by id ==\n");

    if let Some(id) = &task_id {
        let task = client.get_task(id).await?;
        println!("  {} -> {:?}", task.id, task.status.state);
        println!(
            "  artifacts: {}",
            task.artifacts.as_ref().map_or(0, Vec::len)
        );
        println!(
            "\n  `TaskState::is_terminal()` is the poll-loop predicate: keep\n  \
             calling tasks/get while it is false. Completed / Canceled / Failed /\n  \
             Rejected end the lifecycle; Submitted, Working, InputRequired and\n  \
             AuthRequired do not."
        );
    }

    println!("\n== 4. tasks/cancel ==\n");

    // Cancelling a task that has already completed is a normal thing for a
    // client to attempt (it raced the server) and servers reject it.
    if let Some(id) = &task_id {
        match client.cancel_task(id).await {
            Ok(task) => println!("  cancelled -> {:?}", task.status.state),
            Err(e) => println!("  refused, as expected for a finished task:\n    {e}"),
        }
    }
    // A task id that never existed.
    match client.cancel_task("no-such-task").await {
        Ok(task) => println!("  unexpected: {:?}", task.status.state),
        Err(e) => println!("  unknown id:\n    {e}"),
    }

    println!("\n== 5. what a fuller server also offers ==\n");

    // These four are implemented on the client but not yet on this
    // workspace's *server* (see the roadmap in the root README), so they
    // return errors here. The calls are what matters -- point the same client
    // at a server that implements them and they work unchanged.

    // Push notifications: instead of polling, hand the server a callback URL
    // and it POSTs there when the task changes state. The right choice for a
    // long-running task you do not want to hold a connection open for.
    let config = PushNotificationConfig::new("https://my-service.example.com/a2a-callback")
        .with_token("callback-shared-secret");
    match client
        .set_push_notification_config(task_id.as_deref().unwrap_or("task-1"), config)
        .await
    {
        Ok(set) => println!(
            "  push config registered: {:?}",
            set.push_notification_config.id
        ),
        Err(e) => println!("  set_push_notification_config -> {e}"),
    }

    // Streaming: `message/stream` returns an SSE stream of task status and
    // artifact updates instead of one response.
    match client
        .send_message_stream(MessageSendParams::new(text_message("stream this")))
        .await
    {
        Ok(mut stream) => {
            let mut seen = 0;
            while let Some(event) = stream.next().await {
                seen += 1;
                println!("  stream event: {event:?}");
            }
            if seen == 0 {
                println!(
                    "  send_message_stream       -> stream opened but ended with no \
                     events\n                               (this server does not implement \
                     message/stream)"
                );
            }
        }
        Err(e) => println!("  send_message_stream       -> {e}"),
    }

    // Resubscribe: rejoin the event stream of a task you were already
    // watching, after a dropped connection.
    match client
        .resubscribe(task_id.as_deref().unwrap_or("task-1"))
        .await
    {
        Ok(mut stream) => match stream.next().await {
            Some(event) => println!("  resubscribe               -> first event: {event:?}"),
            None => println!(
                "  resubscribe               -> stream opened but ended immediately \
                 (not implemented here)"
            ),
        },
        Err(e) => println!("  resubscribe               -> {e}"),
    }

    // The extended card: a fuller card served only to an authenticated
    // caller, typically listing skills not advertised publicly.
    match client.get_extended_card().await {
        Ok(card) => println!("  extended card: {} skill(s)", card.skills.len()),
        Err(e) => println!("  get_extended_card         -> {e}"),
    }

    println!(
        "\nnote: reach for `A2AAgent` when you want a remote agent to behave like\n\
         a local one; reach for `A2AClient` when the task itself is the thing you\n\
         are managing -- a long-running job you poll, cancel, or get called back\n\
         about. `A2AAgent::client()` hands you this client from an agent you\n\
         already built, so it is not an either/or."
    );

    Ok(())
}
