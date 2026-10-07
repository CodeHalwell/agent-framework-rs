//! End-to-end tests of the bundle on a real `Agent` with a scripted client.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use super::*;
use crate::agent::{Agent, AgentRunOptions, SupportsAgentRun};
use crate::client::{ChatClient, ChatStream};
use crate::error::{Error, Result};
use crate::middleware::{AgentContext, FunctionInvocationContext, Middleware, Next};
use crate::tools::FunctionTool;
use crate::types::{
    AgentResponse, ChatOptions, ChatResponse, ChatResponseUpdate, Content, FinishReason,
    FunctionArguments, FunctionCallContent, Message,
};

/// Returns scripted responses in order (repeating the last) and records
/// every request it receives.
struct Scripted {
    replies: Vec<ChatResponse>,
    calls: AtomicUsize,
    requests: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl Scripted {
    fn new(replies: Vec<ChatResponse>) -> (Self, Arc<Mutex<Vec<Vec<Message>>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                replies,
                calls: AtomicUsize::new(0),
                requests: requests.clone(),
            },
            requests,
        )
    }
}

#[async_trait]
impl ChatClient for Scripted {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.requests.lock().unwrap().push(messages);
        let i = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .replies
            .get(i)
            .or(self.replies.last())
            .cloned()
            .unwrap_or_default())
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let updates: Vec<Result<ChatResponseUpdate>> = response
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

    fn model(&self) -> Option<&str> {
        Some("scripted-model")
    }
}

fn text_reply(text: &str) -> ChatResponse {
    ChatResponse::from_text(text)
}

fn tool_call_reply(call_id: &str, name: &str, args: Value) -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            "assistant",
            vec![Content::FunctionCall(FunctionCallContent::new(
                call_id,
                name,
                Some(FunctionArguments::Object(
                    args.as_object().unwrap().clone().into_iter().collect(),
                )),
            ))],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

type Rule = Arc<dyn Fn(&InterceptionContext) -> Verdict + Send + Sync>;

/// Records every context it sees and answers with `rule`.
struct Recorder {
    seen: Arc<Mutex<Vec<InterceptionContext>>>,
    rule: Rule,
}

#[async_trait]
impl Interceptor for Recorder {
    async fn intercept(&self, context: InterceptionContext) -> Result<Verdict> {
        let verdict = (self.rule)(&context);
        self.seen.lock().unwrap().push(context);
        Ok(verdict)
    }
}

fn recorder(
    rule: impl Fn(&InterceptionContext) -> Verdict + Send + Sync + 'static,
) -> (Arc<dyn Interceptor>, Arc<Mutex<Vec<InterceptionContext>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    (
        Arc::new(Recorder {
            seen: seen.clone(),
            rule: Arc::new(rule),
        }),
        seen,
    )
}

fn points(seen: &Mutex<Vec<InterceptionContext>>) -> Vec<&'static str> {
    seen.lock()
        .unwrap()
        .iter()
        .map(|c| c.point().as_str())
        .collect()
}

fn hooks(interceptor: Arc<dyn Interceptor>) -> AgentHooks {
    AgentHooks::new(AgentHooksOptions::new().interceptor(interceptor)).unwrap()
}

/// A tool that counts its invocations and echoes its arguments.
fn echo_tool(count: Arc<AtomicUsize>) -> crate::tools::ToolDefinition {
    FunctionTool::new(
        "lookup",
        "Look something up",
        json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        move |args: Value| {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(json!({"found": args["q"]}))
            }
        },
    )
    .into_definition()
}

fn blocked_reason(err: &Error) -> Option<&str> {
    err.interception_blocked().and_then(|b| b.reason())
}

#[tokio::test]
async fn emits_every_point_in_order_for_a_plain_run() {
    let (client, _) = Scripted::new(vec![text_reply("hi there")]);
    let (interceptor, seen) = recorder(|_| Verdict::allow());
    let agent = hooks(interceptor).build_agent(Agent::builder(client).name("helper"));
    let response = agent.run(vec![Message::user("hello")], None).await.unwrap();
    assert_eq!(response.text(), "hi there");
    assert_eq!(
        points(&seen),
        vec![
            "agent_startup",
            "input",
            "pre_model_call",
            "post_model_call",
            "output",
            "agent_shutdown"
        ]
    );
    let seen = seen.lock().unwrap();
    let sequences: Vec<u64> = seen.iter().map(InterceptionContext::sequence).collect();
    assert_eq!(sequences, vec![0, 1, 2, 3, 4, 5]);
    let session = seen[0].session_id().to_string();
    assert!(seen.iter().all(|c| c.session_id() == session));
    assert_eq!(seen[0].as_json()["agent"]["framework"], "agent-framework");
    assert_eq!(seen[0].as_json()["agent"]["name"], "helper");
    assert_eq!(
        seen[1].target(),
        &json!({"content": "hello", "role": "user"})
    );
    assert_eq!(seen[2].as_json()["model"]["id"], "scripted-model");
    assert_eq!(seen[3].target()["content"], "hi there");
    assert_eq!(seen[3].target()["tool_calls"], json!([]));
    assert_eq!(seen[4].target(), &json!({"content": "hi there"}));
    assert_eq!(seen[5].target(), &json!({"reason": "completed"}));
}

#[tokio::test]
async fn each_run_is_its_own_session() {
    let (client, _) = Scripted::new(vec![text_reply("ok")]);
    let (interceptor, seen) = recorder(|_| Verdict::allow());
    let agent = hooks(interceptor).build_agent(Agent::builder(client));
    agent.run(vec![Message::user("a")], None).await.unwrap();
    agent.run(vec![Message::user("b")], None).await.unwrap();
    let seen = seen.lock().unwrap();
    assert_ne!(seen[0].session_id(), seen[6].session_id());
    assert_eq!(seen[6].sequence(), 0);
}

#[tokio::test]
async fn input_transform_reaches_the_model_and_history() {
    let (client, requests) = Scripted::new(vec![text_reply("noted")]);
    let (interceptor, _) = recorder(|ctx| {
        if ctx.point() == InterceptionPoint::Input {
            Verdict::transform("$target.content", json!("my card is [redacted]"))
        } else {
            Verdict::allow()
        }
    });
    let agent = hooks(interceptor).build_agent(Agent::builder(client));
    let mut session = agent.create_session();
    agent
        .run(vec![Message::user("my card is 4111")], Some(&mut session))
        .await
        .unwrap();
    agent
        .run(vec![Message::user("second turn")], Some(&mut session))
        .await
        .unwrap();
    let requests = requests.lock().unwrap();
    let first: Vec<String> = requests[0].iter().map(Message::text).collect();
    assert_eq!(first, vec!["my card is [redacted]"]);
    // The persisted history holds the transformed input.
    let second: Vec<String> = requests[1].iter().map(Message::text).collect();
    assert!(second.contains(&"my card is [redacted]".to_string()));
    assert!(!second.iter().any(|t| t.contains("4111")));
}

#[tokio::test]
async fn input_deny_blocks_before_the_model_and_persists_nothing() {
    let (client, requests) = Scripted::new(vec![text_reply("never")]);
    let (interceptor, seen) = recorder(|ctx| {
        if ctx.point() == InterceptionPoint::Input
            && ctx.target()["content"].as_str() == Some("jailbreak")
        {
            Verdict::deny("prompt_injection").with_message("nope")
        } else {
            Verdict::allow()
        }
    });
    let agent = hooks(interceptor).build_agent(Agent::builder(client));
    let mut session = agent.create_session();
    let err = agent
        .run(vec![Message::user("jailbreak")], Some(&mut session))
        .await
        .unwrap_err();
    assert_eq!(blocked_reason(&err), Some("prompt_injection"));
    assert_eq!(
        err.interception_blocked().unwrap().point(),
        InterceptionPoint::Input
    );
    assert!(requests.lock().unwrap().is_empty());
    assert_eq!(
        points(&seen),
        vec!["agent_startup", "input", "agent_shutdown"]
    );
    assert_eq!(
        seen.lock().unwrap()[2].target(),
        &json!({"reason": "error"})
    );
    agent
        .run(vec![Message::user("hello")], Some(&mut session))
        .await
        .unwrap();
    let requests = requests.lock().unwrap();
    assert!(!requests[0].iter().any(|m| m.text() == "jailbreak"));
}

#[tokio::test]
async fn startup_deny_processes_no_input() {
    let (client, requests) = Scripted::new(vec![text_reply("never")]);
    let (interceptor, seen) = recorder(|ctx| {
        if ctx.point() == InterceptionPoint::AgentStartup {
            Verdict::deny("agent_disabled")
        } else {
            Verdict::allow()
        }
    });
    let agent = hooks(interceptor).build_agent(Agent::builder(client));
    let err = agent.run(vec![Message::user("x")], None).await.unwrap_err();
    assert_eq!(blocked_reason(&err), Some("agent_disabled"));
    assert!(requests.lock().unwrap().is_empty());
    assert_eq!(points(&seen), vec!["agent_startup", "agent_shutdown"]);
}

#[tokio::test]
async fn pre_model_call_transform_and_deny() {
    let (client, requests) = Scripted::new(vec![text_reply("ok")]);
    let (interceptor, _) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::PreModelCall => {
            // Rewrite the last message's content.
            let last = ctx.target().as_array().unwrap().len() - 1;
            Verdict::transform(format!("$target[{last}].content"), json!("rewritten"))
        }
        _ => Verdict::allow(),
    });
    let agent = hooks(interceptor).build_agent(Agent::builder(client).instructions("sys"));
    agent
        .run(vec![Message::user("original")], None)
        .await
        .unwrap();
    let texts: Vec<String> = requests.lock().unwrap()[0]
        .iter()
        .map(Message::text)
        .collect();
    assert_eq!(texts, vec!["sys", "rewritten"]);

    let (client, requests) = Scripted::new(vec![text_reply("ok")]);
    let (interceptor, _) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::PreModelCall => Verdict::deny("egress_blocked"),
        _ => Verdict::allow(),
    });
    let agent = hooks(interceptor).build_agent(Agent::builder(client));
    let err = agent.run(vec![Message::user("x")], None).await.unwrap_err();
    assert_eq!(blocked_reason(&err), Some("egress_blocked"));
    assert!(requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn post_model_call_deny_keeps_tool_calls_from_running() {
    let count = Arc::new(AtomicUsize::new(0));
    let (client, _) = Scripted::new(vec![
        tool_call_reply("c1", "lookup", json!({"q": "x"})),
        text_reply("done"),
    ]);
    let (interceptor, _) = recorder(|ctx| {
        if ctx.point() == InterceptionPoint::PostModelCall
            && !ctx.target()["tool_calls"].as_array().unwrap().is_empty()
        {
            Verdict::deny("tool_use_forbidden")
        } else {
            Verdict::allow()
        }
    });
    let agent =
        hooks(interceptor).build_agent(Agent::builder(client).tool(echo_tool(count.clone())));
    let err = agent
        .run(vec![Message::user("go")], None)
        .await
        .unwrap_err();
    assert_eq!(blocked_reason(&err), Some("tool_use_forbidden"));
    assert_eq!(count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn tool_seam_brackets_and_transforms_calls() {
    let count = Arc::new(AtomicUsize::new(0));
    let (client, requests) = Scripted::new(vec![
        tool_call_reply("call-7", "lookup", json!({"q": "secret"})),
        text_reply("done"),
    ]);
    let (interceptor, seen) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::PreToolCall => Verdict::transform("$target.q", json!("public")),
        InterceptionPoint::PostToolCall => Verdict::transform("$target.found", json!("[filtered]")),
        _ => Verdict::allow(),
    });
    let agent =
        hooks(interceptor).build_agent(Agent::builder(client).tool(echo_tool(count.clone())));
    let response = agent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(response.text(), "done");
    assert_eq!(count.load(Ordering::SeqCst), 1);
    {
        let seen = seen.lock().unwrap();
        let pre = seen
            .iter()
            .find(|c| c.point() == InterceptionPoint::PreToolCall)
            .unwrap();
        assert_eq!(pre.as_json()["tool_call"]["id"], "call-7");
        assert_eq!(pre.as_json()["tool_call"]["name"], "lookup");
        let post = seen
            .iter()
            .find(|c| c.point() == InterceptionPoint::PostToolCall)
            .unwrap();
        // post_tool_call reflects the arguments actually passed (§4.2).
        assert_eq!(post.as_json()["tool_call"]["args"], json!({"q": "public"}));
        assert_eq!(post.target(), &json!({"found": "public"}));
        assert_eq!(post.as_json()["tool_result"]["is_error"], false);
    }
    // The model saw the transformed result.
    let requests = requests.lock().unwrap();
    let result = requests[1]
        .iter()
        .flat_map(|m| m.contents.iter())
        .find_map(Content::as_function_result)
        .unwrap();
    assert_eq!(result.result, Some(json!({"found": "[filtered]"})));
}

#[tokio::test]
async fn pre_tool_call_deny_blocks_the_call_and_the_loop_continues() {
    let count = Arc::new(AtomicUsize::new(0));
    let (client, requests) = Scripted::new(vec![
        tool_call_reply("c1", "lookup", json!({"q": "x"})),
        text_reply("handled"),
    ]);
    let (interceptor, seen) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::PreToolCall => Verdict::deny("tool_denied").with_message("not now"),
        _ => Verdict::allow(),
    });
    let agent =
        hooks(interceptor).build_agent(Agent::builder(client).tool(echo_tool(count.clone())));
    let response = agent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(response.text(), "handled");
    assert_eq!(count.load(Ordering::SeqCst), 0);
    // §6.2: no post_tool_call for a blocked pre_tool_call.
    assert!(!points(&seen).contains(&"post_tool_call"));
    // §6.2: the loop continues as if the call failed: an error result whose
    // blocked-call payload reaches the model under the default config.
    let requests = requests.lock().unwrap();
    let result = requests[1]
        .iter()
        .flat_map(|m| m.contents.iter())
        .find_map(Content::as_function_result)
        .unwrap();
    assert!(result.is_error());
    assert_eq!(result.result, None);
    let payload: Value = serde_json::from_str(result.exception.as_deref().unwrap()).unwrap();
    assert_eq!(
        payload,
        json!({
            "error": "Tool call blocked by agent-hooks at pre_tool_call.",
            "reason": "tool_denied",
            "message": "not now"
        })
    );
}

/// Calls `lookup` until tools are switched off, then answers in text.
struct KeepsCalling {
    requests: Arc<AtomicUsize>,
}

#[async_trait]
impl ChatClient for KeepsCalling {
    async fn get_response(&self, _: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        let n = self.requests.fetch_add(1, Ordering::SeqCst);
        Ok(
            if options.tool_choice == Some(crate::types::ToolMode::None) {
                text_reply("gave up")
            } else {
                tool_call_reply(&format!("c{n}"), "lookup", json!({"q": "x"}))
            },
        )
    }

    async fn get_streaming_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatStream> {
        unreachable!("not streamed")
    }
}

#[tokio::test]
async fn denied_tool_calls_count_as_consecutive_errors() {
    let count = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(AtomicUsize::new(0));
    let client = KeepsCalling {
        requests: requests.clone(),
    };
    let (interceptor, _) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::PreToolCall => Verdict::deny("tool_denied"),
        _ => Verdict::allow(),
    });
    let config = crate::tools::FunctionInvocationConfig {
        max_consecutive_errors_per_request: 1,
        ..Default::default()
    };
    let agent = hooks(interceptor).build_agent(
        Agent::builder(client)
            .tool(echo_tool(count.clone()))
            .function_invocation_config(config),
    );
    let response = agent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(response.text(), "gave up");
    assert_eq!(count.load(Ordering::SeqCst), 0);
    // Two denied iterations exceed the limit of one; the third request has
    // tools switched off.
    assert_eq!(requests.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn post_tool_call_deny_discards_the_result() {
    let count = Arc::new(AtomicUsize::new(0));
    let (client, requests) = Scripted::new(vec![
        tool_call_reply("c1", "lookup", json!({"q": "x"})),
        text_reply("handled"),
    ]);
    let (interceptor, _) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::PostToolCall => Verdict::deny("leaky_result"),
        _ => Verdict::allow(),
    });
    let agent =
        hooks(interceptor).build_agent(Agent::builder(client).tool(echo_tool(count.clone())));
    agent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let requests = requests.lock().unwrap();
    let result = requests[1]
        .iter()
        .flat_map(|m| m.contents.iter())
        .find_map(Content::as_function_result)
        .unwrap();
    // §6.1: discarded as if it had errored.
    assert_eq!(result.result, None);
    let exception = result.exception.as_deref().unwrap();
    assert!(exception.contains("leaky_result"), "{exception}");
    assert!(!exception.contains("found"), "{exception}");
}

#[tokio::test]
async fn tool_seam_host_error_halts_the_whole_run() {
    let count = Arc::new(AtomicUsize::new(0));
    let (client, requests) = Scripted::new(vec![
        tool_call_reply("c1", "lookup", json!({"q": "x"})),
        text_reply("should not happen"),
    ]);
    let interceptor: Arc<dyn Interceptor> = Arc::new(interceptor_fn(|ctx| async move {
        if ctx.point() == InterceptionPoint::PreToolCall {
            Err(Error::other("policy engine down"))
        } else {
            Ok(Verdict::allow())
        }
    }));
    let (observer, seen) = recorder(|_| Verdict::allow());
    let hooks = AgentHooks::new(
        AgentHooksOptions::new()
            .interceptor(interceptor)
            .interceptor(observer),
    )
    .unwrap();
    let mut session = None;
    let agent = hooks.build_agent(Agent::builder(client).tool(echo_tool(count.clone())));
    let err = agent
        .run(vec![Message::user("go")], session.as_mut())
        .await
        .unwrap_err();
    assert_eq!(blocked_reason(&err), Some(host_error::INTERCEPTOR_FAILED));
    assert_eq!(count.load(Ordering::SeqCst), 0);
    // The model was never called again after the halt.
    assert_eq!(requests.lock().unwrap().len(), 1);
    // The session trail still closes.
    assert_eq!(points(&seen).last(), Some(&"agent_shutdown"));
}

#[tokio::test]
async fn output_deny_and_transform() {
    let (client, requests) = Scripted::new(vec![text_reply("the password is hunter2")]);
    let (interceptor, _) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::Output => Verdict::deny("data_leak"),
        _ => Verdict::allow(),
    });
    let agent = hooks(interceptor).build_agent(Agent::builder(client));
    let mut session = agent.create_session();
    let err = agent
        .run(vec![Message::user("pw?")], Some(&mut session))
        .await
        .unwrap_err();
    assert_eq!(blocked_reason(&err), Some("data_leak"));
    // Nothing from the denied turn was persisted: a later run on the same
    // session (with an allow-all agent sharing it) sees no history.
    let (client2, requests2) = Scripted::new(vec![text_reply("ok")]);
    let (allow, _) = recorder(|_| Verdict::allow());
    let agent2 = hooks(allow).build_agent(Agent::builder(client2));
    agent2
        .run(vec![Message::user("next")], Some(&mut session))
        .await
        .unwrap();
    let texts: Vec<String> = requests2.lock().unwrap()[0]
        .iter()
        .map(Message::text)
        .collect();
    assert_eq!(texts, vec!["next"]);
    assert_eq!(requests.lock().unwrap().len(), 1);

    let (client, requests) = Scripted::new(vec![text_reply("the password is hunter2")]);
    let (interceptor, _) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::Output => Verdict::transform("$target.content", json!("[withheld]")),
        _ => Verdict::allow(),
    });
    let agent = hooks(interceptor).build_agent(Agent::builder(client));
    let mut session = agent.create_session();
    let response = agent
        .run(vec![Message::user("pw?")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(response.text(), "[withheld]");
    agent
        .run(vec![Message::user("again")], Some(&mut session))
        .await
        .unwrap();
    let wire = serde_json::to_string(&requests.lock().unwrap()[1]).unwrap();
    assert!(wire.contains("[withheld]"));
    assert!(!wire.contains("hunter2"), "{wire}");
}

/// A user middleware that short-circuits with its own result; installed
/// before the bundle, it still ends up inside the enforcement boundary.
struct Substitute;

#[async_trait]
impl Middleware<AgentContext> for Substitute {
    async fn process(
        &self,
        mut ctx: AgentContext,
        _next: Next<AgentContext>,
    ) -> Result<AgentContext> {
        ctx.result = Some(AgentResponse {
            messages: vec![Message::assistant("substituted secret")],
            ..Default::default()
        });
        ctx.terminate = true;
        Ok(ctx)
    }
}

#[tokio::test]
async fn substituted_results_still_pass_output() {
    let (client, _) = Scripted::new(vec![text_reply("model")]);
    let (interceptor, seen) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::Output if ctx.target().to_string().contains("secret") => {
            Verdict::deny("substitution_blocked")
        }
        _ => Verdict::allow(),
    });
    let agent =
        hooks(interceptor).build_agent(Agent::builder(client).middleware(Arc::new(Substitute)));
    let err = agent.run(vec![Message::user("x")], None).await.unwrap_err();
    assert_eq!(blocked_reason(&err), Some("substitution_blocked"));
    assert!(points(&seen).contains(&"output"));
}

#[tokio::test]
async fn streaming_is_buffered_behind_the_output_verdict() {
    let (client, _) = Scripted::new(vec![text_reply("streamed secret")]);
    let (interceptor, _) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::Output => Verdict::deny("data_leak"),
        _ => Verdict::allow(),
    });
    let agent = hooks(interceptor).build_agent(Agent::builder(client));
    let err = match SupportsAgentRun::run_stream(&agent, vec![Message::user("x")], None, None).await
    {
        Ok(_) => panic!("a denied stream must not open"),
        Err(e) => e,
    };
    assert_eq!(blocked_reason(&err), Some("data_leak"));

    let (client, _) = Scripted::new(vec![text_reply("streamed secret")]);
    let (interceptor, seen) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::Output => Verdict::transform("$target.content", json!("streamed [x]")),
        _ => Verdict::allow(),
    });
    let agent = hooks(interceptor).build_agent(Agent::builder(client));
    let stream = SupportsAgentRun::run_stream(&agent, vec![Message::user("x")], None, None)
        .await
        .unwrap();
    // The run (and its shutdown) completed before the stream was handed out.
    assert_eq!(points(&seen).last(), Some(&"agent_shutdown"));
    let updates: Vec<_> = stream.collect().await;
    let text: String = updates.into_iter().map(|u| u.unwrap().text()).collect();
    assert_eq!(text, "streamed [x]");
}

#[tokio::test]
async fn run_tools_appear_in_startup_and_pre_model_call() {
    let count = Arc::new(AtomicUsize::new(0));
    let (client, _) = Scripted::new(vec![text_reply("ok")]);
    let (interceptor, seen) = recorder(|_| Verdict::allow());
    let agent = hooks(interceptor).build_agent(Agent::builder(client).tool(echo_tool(count)));
    let extra = FunctionTool::new("extra", "", json!({"type": "object"}), |_| async {
        Ok(Value::Null)
    })
    .into_definition();
    agent
        .run_with_options(
            vec![Message::user("x")],
            None,
            AgentRunOptions::new().with_tool(extra),
        )
        .await
        .unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(
        seen[0].target(),
        &json!({"tools_registered": ["lookup", "extra"]})
    );
    assert_eq!(
        seen[2].as_json()["tools"],
        json!([{"name": "lookup", "description": "Look something up"}, {"name": "extra"}])
    );
}

#[tokio::test]
async fn host_owned_sessions_span_runs() {
    let (client, _) = Scripted::new(vec![text_reply("ok")]);
    let (interceptor, seen) = recorder(|_| Verdict::allow());
    let emitter = Arc::new(InterceptionEmitter::new().register(interceptor, None));
    let builder = Arc::new(InterceptionContextBuilder::new(
        "host-agent",
        "agent-framework",
        "host-session",
    ));
    emitter.emit(builder.agent_startup(vec![])).await.unwrap();
    let agent = AgentHooks::from_emitter(emitter.clone(), builder.clone())
        .build_agent(Agent::builder(client));
    agent.run(vec![Message::user("a")], None).await.unwrap();
    agent.run(vec![Message::user("b")], None).await.unwrap();
    emitter
        .emit(builder.agent_shutdown(ShutdownReason::Completed))
        .await
        .unwrap();
    let points = points(&seen);
    assert_eq!(points.first(), Some(&"agent_startup"));
    assert_eq!(points.last(), Some(&"agent_shutdown"));
    assert_eq!(points.iter().filter(|p| **p == "agent_startup").count(), 1);
    assert_eq!(points.iter().filter(|p| **p == "output").count(), 2);
    let seen = seen.lock().unwrap();
    assert!(seen.iter().all(|c| c.session_id() == "host-session"));
    assert!(seen.windows(2).all(|w| w[0].sequence() < w[1].sequence()));
}

#[tokio::test]
async fn evaluate_only_records_but_proceeds() {
    let (client, _) = Scripted::new(vec![text_reply("fine")]);
    let records = Arc::new(Mutex::new(Vec::new()));
    let sink_records = records.clone();
    let (interceptor, _) = recorder(|_| Verdict::deny("would_block"));
    let hooks = AgentHooks::new(
        AgentHooksOptions::new()
            .interceptor(interceptor)
            .mode(EnforcementMode::EvaluateOnly)
            .record_sink(Arc::new(move |r: &InterceptionRecord| {
                sink_records.lock().unwrap().push(r.clone())
            })),
    )
    .unwrap();
    let agent = hooks.build_agent(Agent::builder(client));
    let response = agent.run(vec![Message::user("x")], None).await.unwrap();
    assert_eq!(response.text(), "fine");
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 6);
    assert!(records
        .iter()
        .all(|r| r.mode == EnforcementMode::EvaluateOnly
            && r.verdict.reason.as_deref() == Some("would_block")));
}

#[test]
fn a_bundle_needs_an_interceptor() {
    let err = AgentHooks::new(AgentHooksOptions::new()).unwrap_err();
    assert!(matches!(err, Error::Configuration(_)));
}

#[tokio::test]
async fn nested_guarded_agents_keep_their_own_runs() {
    // A guarded sub-agent used as a tool of a guarded parent: each bundle
    // binds to its own run state.
    let (sub_client, _) = Scripted::new(vec![text_reply("sub answer")]);
    let (sub_interceptor, sub_seen) = recorder(|_| Verdict::allow());
    let sub = Arc::new(hooks(sub_interceptor).build_agent(Agent::builder(sub_client)));
    let sub_tool = {
        let sub = sub.clone();
        FunctionTool::new("ask_sub", "", json!({"type": "object"}), move |_| {
            let sub = sub.clone();
            async move {
                let r = sub.run(vec![Message::user("inner")], None).await?;
                Ok(Value::String(r.text()))
            }
        })
        .into_definition()
    };
    let (client, _) = Scripted::new(vec![
        tool_call_reply("c1", "ask_sub", json!({})),
        text_reply("parent done"),
    ]);
    let (interceptor, seen) = recorder(|_| Verdict::allow());
    let parent = hooks(interceptor).build_agent(Agent::builder(client).tool(sub_tool));
    let response = parent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(response.text(), "parent done");
    assert_eq!(points(&sub_seen).len(), 6);
    let seen = seen.lock().unwrap();
    let post = seen
        .iter()
        .find(|c| c.point() == InterceptionPoint::PostToolCall)
        .unwrap();
    assert_eq!(post.target(), &json!("sub answer"));
    let session = seen[0].session_id();
    assert!(seen.iter().all(|c| c.session_id() == session));
}

/// A user function middleware that rewrites the tool arguments.
struct RewriteArgs(Value);

#[async_trait]
impl Middleware<FunctionInvocationContext> for RewriteArgs {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        ctx.arguments = self.0.clone();
        next.run(ctx).await
    }
}

#[tokio::test]
async fn pre_tool_call_judges_the_arguments_after_user_middleware() {
    // A deny keyed on the rewritten arguments must stop the call.
    let count = Arc::new(AtomicUsize::new(0));
    let (client, _) = Scripted::new(vec![
        tool_call_reply("c1", "lookup", json!({"q": "public"})),
        text_reply("handled"),
    ]);
    let (interceptor, _) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::PreToolCall if ctx.target()["q"] == "sensitive" => {
            Verdict::deny("sensitive_path")
        }
        _ => Verdict::allow(),
    });
    let agent = hooks(interceptor).build_agent(
        Agent::builder(client)
            .tool(echo_tool(count.clone()))
            .function_middleware(Arc::new(RewriteArgs(json!({"q": "sensitive"})))),
    );
    agent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 0);

    // An allowed call reports the arguments the tool actually received.
    let (client, _) = Scripted::new(vec![
        tool_call_reply("c1", "lookup", json!({"q": "original"})),
        text_reply("handled"),
    ]);
    let (interceptor, seen) = recorder(|_| Verdict::allow());
    let agent = hooks(interceptor).build_agent(
        Agent::builder(client)
            .tool(echo_tool(count.clone()))
            .function_middleware(Arc::new(RewriteArgs(json!({"q": "rewritten"})))),
    );
    agent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let seen = seen.lock().unwrap();
    let pre = seen
        .iter()
        .find(|c| c.point() == InterceptionPoint::PreToolCall)
        .unwrap();
    assert_eq!(pre.target(), &json!({"q": "rewritten"}));
    assert_eq!(pre.as_json()["tool_call"]["id"], "c1");
    let post = seen
        .iter()
        .find(|c| c.point() == InterceptionPoint::PostToolCall)
        .unwrap();
    assert_eq!(
        post.as_json()["tool_call"]["args"],
        json!({"q": "rewritten"})
    );
    assert_eq!(post.as_json()["tool_call"]["id"], "c1");
    assert_eq!(post.target(), &json!({"found": "rewritten"}));
}

#[tokio::test]
async fn post_tool_call_transform_rewrites_an_errored_result() {
    let failing = FunctionTool::new("lookup", "", json!({"type": "object"}), |_| async {
        Err::<Value, _>(Error::Tool("raw internal detail".into()))
    })
    .into_definition();
    let (client, requests) = Scripted::new(vec![
        tool_call_reply("c1", "lookup", json!({})),
        text_reply("handled"),
    ]);
    let (interceptor, seen) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::PostToolCall => Verdict::transform("$target", json!("sanitized")),
        _ => Verdict::allow(),
    });
    // The default config (no detailed errors): the transformed text is the
    // interceptor's, addressed to the model, so it still gets through.
    let agent = hooks(interceptor).build_agent(Agent::builder(client).tool(failing));
    agent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(
        seen.lock()
            .unwrap()
            .iter()
            .find(|c| c.point() == InterceptionPoint::PostToolCall)
            .unwrap()
            .as_json()["tool_result"]["is_error"],
        true
    );
    let requests = requests.lock().unwrap();
    let exception = requests[1]
        .iter()
        .flat_map(|m| m.contents.iter())
        .find_map(Content::as_function_result)
        .and_then(|r| r.exception.clone())
        .unwrap();
    assert_eq!(exception, "sanitized");
}

#[tokio::test]
async fn post_tool_call_judges_the_error_text_the_model_sees() {
    // A rejection is shown verbatim; another error only with detailed
    // errors on. post_tool_call must judge exactly that text.
    async fn judged(error: fn() -> Error, detailed: bool) -> (Value, String) {
        let failing = FunctionTool::new(
            "lookup",
            "",
            json!({"type": "object"}),
            move |_| async move { Err::<Value, _>(error()) },
        )
        .into_definition();
        let (client, requests) = Scripted::new(vec![
            tool_call_reply("c1", "lookup", json!({})),
            text_reply("handled"),
        ]);
        let (interceptor, seen) = recorder(|_| Verdict::allow());
        let config = crate::tools::FunctionInvocationConfig {
            include_detailed_errors: detailed,
            ..Default::default()
        };
        let agent = hooks(interceptor).build_agent(
            Agent::builder(client)
                .tool(failing)
                .function_invocation_config(config),
        );
        agent.run(vec![Message::user("go")], None).await.unwrap();
        let target = seen
            .lock()
            .unwrap()
            .iter()
            .find(|c| c.point() == InterceptionPoint::PostToolCall)
            .unwrap()
            .target()
            .clone();
        let exception = requests.lock().unwrap()[1]
            .iter()
            .flat_map(|m| m.contents.iter())
            .find_map(Content::as_function_result)
            .and_then(|r| r.exception.clone())
            .unwrap();
        (target, exception)
    }

    let (target, seen) = judged(|| Error::tool_rejected("secret 1234"), false).await;
    assert_eq!(seen, "secret 1234");
    assert_eq!(target, json!(seen));

    let (target, seen) = judged(|| Error::Tool("secret 1234".into()), true).await;
    assert!(seen.contains("secret 1234"), "{seen}");
    assert_eq!(target, json!(seen));

    let (target, seen) = judged(|| Error::Tool("secret 1234".into()), false).await;
    assert!(!seen.contains("secret"), "{seen}");
    assert_eq!(target, json!(seen));
}

#[tokio::test]
async fn post_tool_call_deny_discards_an_errored_result() {
    let failing = FunctionTool::new("lookup", "", json!({"type": "object"}), |_| async {
        Err::<Value, _>(Error::tool_rejected("secret 1234"))
    })
    .into_definition();
    let (client, requests) = Scripted::new(vec![
        tool_call_reply("c1", "lookup", json!({})),
        text_reply("handled"),
    ]);
    let (interceptor, _) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::PostToolCall => Verdict::deny("leaky_result"),
        _ => Verdict::allow(),
    });
    let agent = hooks(interceptor).build_agent(Agent::builder(client).tool(failing));
    agent.run(vec![Message::user("go")], None).await.unwrap();
    let requests = requests.lock().unwrap();
    let result = requests[1]
        .iter()
        .flat_map(|m| m.contents.iter())
        .find_map(Content::as_function_result)
        .unwrap();
    // §6.1: the denied error payload never reaches the model; the
    // blocked-call payload replaces it.
    assert_eq!(result.result, None);
    let exception = result.exception.as_deref().unwrap();
    assert!(exception.contains("leaky_result"), "{exception}");
    assert!(!exception.contains("secret"), "{exception}");
}

#[tokio::test]
async fn post_tool_call_brackets_a_dispatched_call_that_fails_closed() {
    let failing = FunctionTool::new("lookup", "", json!({"type": "object"}), |_| async {
        Err::<Value, _>(Error::middleware_failure("executor halted"))
    })
    .into_definition();
    let (client, _) = Scripted::new(vec![
        tool_call_reply("c1", "lookup", json!({})),
        text_reply("unreachable"),
    ]);
    let (interceptor, seen) = recorder(|_| Verdict::allow());
    let agent = hooks(interceptor).build_agent(Agent::builder(client).tool(failing));
    let err = agent
        .run(vec![Message::user("go")], None)
        .await
        .unwrap_err();
    // The halt still surfaces...
    assert!(err.is_middleware_failure(), "{err}");
    // ...but the dispatched call is still bracketed by post_tool_call.
    let seen = seen.lock().unwrap();
    let post = seen
        .iter()
        .find(|c| c.point() == InterceptionPoint::PostToolCall)
        .expect("post_tool_call for the dispatched call");
    assert_eq!(post.as_json()["tool_result"]["is_error"], true);
}

fn host_calls(response: &ChatResponse) -> Vec<(String, Value)> {
    response
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .filter_map(|c| match c {
            Content::FunctionCall(call) => Some((
                call.call_id.clone(),
                Value::Object(call.parse_arguments().unwrap().into_iter().collect()),
            )),
            _ => None,
        })
        .collect()
}

fn two_calls(first: &str, second: &str) -> ChatResponse {
    let call = |id: &str, q: i64| {
        Content::FunctionCall(FunctionCallContent::new(
            id,
            "lookup",
            Some(FunctionArguments::Object(
                [("q".to_string(), json!(q))].into_iter().collect(),
            )),
        ))
    };
    ChatResponse {
        messages: vec![Message::with_contents(
            "assistant",
            vec![Content::text("calling"), call(first, 1), call(second, 2)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

#[test]
fn post_model_call_transform_reorders_tool_calls() {
    let mut response = two_calls("a", "b");
    let before = codecs::response_to_wire(&response).unwrap();
    let mut after = before.clone();
    after["tool_calls"].as_array_mut().unwrap().swap(0, 1);
    assert!(codecs::response_write_back(&mut response, &before, &after).unwrap());
    assert_eq!(
        host_calls(&response),
        vec![("b".into(), json!({"q": 2})), ("a".into(), json!({"q": 1}))]
    );
    // The visible text keeps its place ahead of the calls.
    assert!(matches!(
        &response.messages[0].contents[0],
        Content::Text(_)
    ));
}

#[test]
fn post_model_call_transform_keeps_duplicate_call_ids_apart() {
    let mut response = two_calls("dup", "dup");
    let before = codecs::response_to_wire(&response).unwrap();
    let mut after = before.clone();
    after["tool_calls"][1]["args"] = json!({"q": 3});
    assert!(codecs::response_write_back(&mut response, &before, &after).unwrap());
    assert_eq!(
        host_calls(&response),
        vec![
            ("dup".into(), json!({"q": 1})),
            ("dup".into(), json!({"q": 3}))
        ]
    );
}

/// A user function middleware that relabels the invocation.
struct RenameTool(&'static str);

#[async_trait]
impl Middleware<FunctionInvocationContext> for RenameTool {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        ctx.function_name = self.0.to_string();
        next.run(ctx).await
    }
}

#[tokio::test]
async fn tool_seam_judges_the_selected_tool_not_a_rewritten_name() {
    // Renaming the invocation cannot launder a forbidden tool past policy:
    // the executor was fixed when the loop selected `lookup`.
    let count = Arc::new(AtomicUsize::new(0));
    let (client, _) = Scripted::new(vec![
        tool_call_reply("c1", "lookup", json!({"q": "x"})),
        text_reply("handled"),
    ]);
    let (interceptor, seen) = recorder(|ctx| match ctx.point() {
        InterceptionPoint::PreToolCall if ctx.as_json()["tool_call"]["name"] == "lookup" => {
            Verdict::deny("tool_forbidden")
        }
        _ => Verdict::allow(),
    });
    let agent = hooks(interceptor).build_agent(
        Agent::builder(client)
            .tool(echo_tool(count.clone()))
            .function_middleware(Arc::new(RenameTool("harmless"))),
    );
    agent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(points(&seen).contains(&"pre_tool_call"));

    // An allowed call names the selected tool at both halves.
    let (client, _) = Scripted::new(vec![
        tool_call_reply("c1", "lookup", json!({"q": "x"})),
        text_reply("handled"),
    ]);
    let (interceptor, seen) = recorder(|_| Verdict::allow());
    let agent = hooks(interceptor).build_agent(
        Agent::builder(client)
            .tool(echo_tool(count.clone()))
            .function_middleware(Arc::new(RenameTool("harmless"))),
    );
    agent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let seen = seen.lock().unwrap();
    for point in [
        InterceptionPoint::PreToolCall,
        InterceptionPoint::PostToolCall,
    ] {
        let ctx = seen.iter().find(|c| c.point() == point).unwrap();
        assert_eq!(ctx.as_json()["tool_call"]["name"], "lookup", "{point:?}");
    }
}

/// A user agent middleware that hides any inner failure behind a canned
/// response.
struct SwallowErrors;

#[async_trait]
impl Middleware<AgentContext> for SwallowErrors {
    async fn process(&self, ctx: AgentContext, next: Next<AgentContext>) -> Result<AgentContext> {
        let mut fallback = AgentContext::new(ctx.messages.clone(), ctx.is_streaming);
        match next.run(ctx).await {
            Ok(ctx) => Ok(ctx),
            Err(_) => {
                fallback.result = Some(AgentResponse {
                    messages: vec![Message::assistant("substituted")],
                    ..Default::default()
                });
                Ok(fallback)
            }
        }
    }
}

#[tokio::test]
async fn swallowed_model_call_denies_still_halt_the_run() {
    for point in [
        InterceptionPoint::PreModelCall,
        InterceptionPoint::PostModelCall,
    ] {
        let (client, _) = Scripted::new(vec![text_reply("model")]);
        let (interceptor, seen) = recorder(move |ctx| {
            if ctx.point() == point {
                Verdict::deny("model_blocked")
            } else {
                Verdict::allow()
            }
        });
        let agent = hooks(interceptor)
            .build_agent(Agent::builder(client).middleware(Arc::new(SwallowErrors)));
        let err = agent.run(vec![Message::user("x")], None).await.unwrap_err();
        assert_eq!(blocked_reason(&err), Some("model_blocked"), "{point:?}");
        assert!(!points(&seen).contains(&"output"), "{point:?}");
    }
}
