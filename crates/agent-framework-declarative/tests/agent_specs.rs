//! Agent-spec parsing, field mapping, agent building, and error tests.
//!
//! The `specs/*.yaml` fixtures are verbatim copies of upstream reference
//! samples, cited at each test:
//!   * `agent-samples/chatclient/GetWeather.yaml`
//!   * `agent-samples/openai/OpenAIChat.yaml`
//!   * `agent-samples/foundry/MicrosoftLearnAgent.yaml`

mod common;

use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::types::{Content, FunctionArguments, FunctionCallContent, Role};
use agent_framework_declarative::{
    AgentSpec, ApprovalModeSpec, ChatClientFactory, DeclarativeError, DeclarativeLoader,
    ToolRegistry,
};
use common::MockClient;
use serde_json::json;

const GET_WEATHER: &str = include_str!("specs/GetWeather.yaml");
const OPENAI_CHAT: &str = include_str!("specs/OpenAIChat.yaml");
const MS_LEARN: &str = include_str!("specs/MicrosoftLearnAgent.yaml");

/// `ChatAgent` does not implement `Debug`, so `Result::unwrap_err` is
/// unavailable; extract the error explicitly.
fn expect_load_err(loader: &DeclarativeLoader, yaml: &str) -> DeclarativeError {
    match loader.load_agent(yaml) {
        Ok(_) => panic!("expected load_agent to fail for spec:\n{yaml}"),
        Err(err) => err,
    }
}

// ---------------------------------------------------------------------------
// Real sample: agent-samples/chatclient/GetWeather.yaml
// ---------------------------------------------------------------------------

#[test]
fn parse_get_weather_sample_all_fields() {
    let spec = AgentSpec::from_yaml(GET_WEATHER).expect("parse GetWeather.yaml");

    assert_eq!(spec.kind, "Prompt");
    assert_eq!(spec.name.as_deref(), Some("Assistant"));
    assert_eq!(spec.description.as_deref(), Some("Helpful assistant"));
    assert_eq!(
        spec.instructions.as_deref(),
        Some("You are a helpful assistant. You answer questions using the tools provided.")
    );

    // model has only options (no id/provider) -> resolves via default factory.
    let model = spec.model.as_ref().expect("model");
    assert!(model.id.is_none());
    assert!(model.provider.is_none());
    let options = model.options.as_ref().expect("options");
    assert_eq!(options.temperature, Some(0.9));
    assert_eq!(options.top_p, Some(0.95));
    assert_eq!(options.allow_multiple_tool_calls, Some(true));
    assert_eq!(options.chat_tool_mode.as_deref(), Some("auto"));

    // function tool with a binding and a parameter schema.
    assert_eq!(spec.tools.len(), 1);
    let tool = &spec.tools[0];
    assert_eq!(tool.kind, "function");
    assert_eq!(tool.name.as_deref(), Some("GetWeather"));
    assert_eq!(
        tool.description.as_deref(),
        Some("Get the weather for a given location.")
    );
    let bindings = tool.bindings.as_ref().expect("bindings");
    assert_eq!(bindings.get("get_weather"), Some(&json!("get_weather")));

    let params = tool.parameters.as_ref().expect("parameters");
    let location = params.properties.get("location").expect("location prop");
    assert_eq!(location.kind.as_deref(), Some("string"));
    assert_eq!(location.required, Some(true));
    let unit = params.properties.get("unit").expect("unit prop");
    assert_eq!(unit.required, Some(false));
    assert_eq!(
        unit.enum_values.as_ref().expect("enum"),
        &vec![json!("celsius"), json!("fahrenheit")]
    );
}

#[tokio::test]
async fn get_weather_binds_and_executes_native_tool() {
    // A native tool the spec's function tool binds to (via `bindings.get_weather`).
    let calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let calls_for_tool = calls.clone();
    let get_weather = AiFunction::new(
        "get_weather",
        "native get weather",
        json!({"type": "object", "properties": {"location": {"type": "string"}}}),
        move |args: serde_json::Value| {
            let calls = calls_for_tool.clone();
            async move {
                let loc = args
                    .get("location")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                calls.lock().unwrap().push(loc.clone());
                Ok(json!(format!("The weather in {loc} is sunny.")))
            }
        },
    );

    // Mock: first turn requests the "GetWeather" tool (the spec name), then a
    // final answer. The bound native closure must run in between.
    let call = FunctionCallContent::new(
        "call_1",
        "GetWeather",
        Some(FunctionArguments::Raw(
            json!({"location": "Paris"}).to_string(),
        )),
    );
    let ask = ChatResponse {
        messages: vec![ChatMessage::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    };
    let answer = ChatResponse::from_text("It is sunny in Paris.");
    let mock = MockClient::new(vec![ask, answer]);

    let loader = DeclarativeLoader::new()
        .with_client_factory(
            ChatClientFactory::new().with_default(move |_model| Ok(Arc::new(mock.clone()) as _)),
        )
        .with_tool_registry(ToolRegistry::new().with(get_weather.into_definition()));

    let agent = loader.load_agent(GET_WEATHER).expect("build agent");
    let response = agent.run_once("Weather in Paris?").await.expect("run");

    assert_eq!(response.text(), "It is sunny in Paris.");
    assert_eq!(calls.lock().unwrap().as_slice(), &["Paris".to_string()]);
}

// ---------------------------------------------------------------------------
// Real sample: agent-samples/openai/OpenAIChat.yaml
// ---------------------------------------------------------------------------

#[test]
fn parse_openai_chat_sample_all_fields() {
    let spec = AgentSpec::from_yaml(OPENAI_CHAT).expect("parse OpenAIChat.yaml");

    assert_eq!(spec.kind, "Prompt");
    assert_eq!(spec.name.as_deref(), Some("Assistant"));

    let model = spec.model.as_ref().expect("model");
    assert_eq!(model.id.as_deref(), Some("gpt-4.1-mini"));
    assert_eq!(model.provider.as_deref(), Some("OpenAI"));
    assert_eq!(model.api_type.as_deref(), Some("Chat"));
    assert_eq!(model.provider_key().as_deref(), Some("OpenAI.Chat"));

    let options = model.options.as_ref().expect("options");
    assert_eq!(options.temperature, Some(0.9));
    assert_eq!(options.top_p, Some(0.95));

    // Connection: `kind: ApiKey`, `key: =Env.OPENAI_API_KEY` (PowerFx literal,
    // left untouched by this port).
    let conn = model.connection.as_ref().expect("connection");
    assert_eq!(conn.kind.as_deref(), Some("ApiKey"));
    assert_eq!(conn.resolved_key(), Some("=Env.OPENAI_API_KEY"));

    // outputSchema.properties use `type:` (aliased to `kind`).
    let out = spec.output_schema.as_ref().expect("outputSchema");
    for name in ["language", "answer", "type"] {
        let prop = out.properties.get(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(prop.kind.as_deref(), Some("string"));
        assert_eq!(prop.required, Some(true));
    }

    // The output schema maps to a JSON-schema response_format.
    let schema = out.to_json_schema();
    assert_eq!(schema["type"], json!("object"));
    let required = schema["required"].as_array().expect("required");
    assert_eq!(required.len(), 3);
}

#[tokio::test]
async fn openai_chat_resolves_provider_specific_factory() {
    let used_key: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let used_key_for_factory = used_key.clone();

    let loader = DeclarativeLoader::new().with_client_factory(ChatClientFactory::new().with(
        "OpenAI.Chat",
        move |model| {
            // The connection block is available to the factory.
            let key = model
                .connection
                .as_ref()
                .and_then(|c| c.resolved_key())
                .map(str::to_string);
            *used_key_for_factory.lock().unwrap() = key;
            Ok(Arc::new(MockClient::always("hi from openai")) as _)
        },
    ));

    let agent = loader.load_agent(OPENAI_CHAT).expect("build agent");
    let response = agent.run_once("Hello").await.expect("run");
    assert_eq!(response.text(), "hi from openai");
    assert_eq!(
        used_key.lock().unwrap().as_deref(),
        Some("=Env.OPENAI_API_KEY")
    );
}

// ---------------------------------------------------------------------------
// Real sample: agent-samples/foundry/MicrosoftLearnAgent.yaml
// ---------------------------------------------------------------------------

#[test]
fn parse_microsoft_learn_sample_mcp_tool() {
    let spec = AgentSpec::from_yaml(MS_LEARN).expect("parse MicrosoftLearnAgent.yaml");

    assert_eq!(spec.name.as_deref(), Some("MicrosoftLearnAgent"));
    let model = spec.model.as_ref().expect("model");
    assert_eq!(
        model.id.as_deref(),
        Some("=Env.AZURE_FOUNDRY_PROJECT_MODEL_ID")
    );
    let conn = model.connection.as_ref().expect("connection");
    assert_eq!(conn.kind.as_deref(), Some("remote"));
    assert_eq!(
        conn.endpoint.as_deref(),
        Some("=Env.AZURE_FOUNDRY_PROJECT_ENDPOINT")
    );

    assert_eq!(spec.tools.len(), 1);
    let tool = &spec.tools[0];
    assert_eq!(tool.kind, "mcp");
    assert_eq!(tool.name.as_deref(), Some("microsoft_learn"));
    assert_eq!(
        tool.url.as_deref(),
        Some("https://learn.microsoft.com/api/mcp")
    );
    assert_eq!(
        tool.allowed_tools.as_ref().expect("allowedTools"),
        &vec!["microsoft_docs_search".to_string()]
    );
    match tool.approval_mode.as_ref().expect("approvalMode") {
        ApprovalModeSpec::Detailed(d) => assert_eq!(d.kind.as_deref(), Some("never")),
        other => panic!("expected detailed approval mode, got {other:?}"),
    }
}

#[test]
fn microsoft_learn_builds_with_hosted_mcp_tool() {
    let loader = DeclarativeLoader::new().with_client_factory(
        ChatClientFactory::new().with_default(|_model| Ok(Arc::new(MockClient::always("ok")) as _)),
    );
    // Building must succeed: the mcp tool becomes a hosted-tool marker.
    let agent = loader.load_agent(MS_LEARN).expect("build agent");
    assert_eq!(agent.name(), Some("MicrosoftLearnAgent"));
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn unknown_top_level_field_is_actionable() {
    let yaml = "kind: Prompt\nname: A\nnonsense: true\n";
    let err = AgentSpec::from_yaml(yaml).unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, DeclarativeError::Parse(_)), "got {err:?}");
    assert!(
        msg.contains("nonsense"),
        "message should name the field: {msg}"
    );
}

#[test]
fn unsupported_agent_kind_is_actionable() {
    let loader = DeclarativeLoader::new().with_client_factory(
        ChatClientFactory::new().with_default(|_| Ok(Arc::new(MockClient::always("x")) as _)),
    );
    let err = expect_load_err(&loader, "kind: Banana\nname: A\n");
    match &err {
        DeclarativeError::UnsupportedKind {
            what,
            kind,
            expected,
        } => {
            assert_eq!(*what, "agent");
            assert_eq!(kind, "Banana");
            assert!(expected.contains(&"Prompt"));
        }
        other => panic!("expected UnsupportedKind, got {other:?}"),
    }
    assert!(err.to_string().contains("expected one of"));
}

#[test]
fn unsupported_tool_kind_is_actionable() {
    let loader = DeclarativeLoader::new().with_client_factory(
        ChatClientFactory::new().with_default(|_| Ok(Arc::new(MockClient::always("x")) as _)),
    );
    let yaml = "kind: Prompt\nname: A\ntools:\n  - kind: quantum\n    name: q\n";
    let err = expect_load_err(&loader, yaml);
    assert!(
        matches!(&err, DeclarativeError::UnsupportedKind { what, .. } if *what == "tool"),
        "got {err:?}"
    );
}

#[test]
fn missing_client_factory_is_actionable() {
    // No factory registered at all.
    let loader = DeclarativeLoader::new();
    let err = expect_load_err(
        &loader,
        "kind: Prompt\nname: A\nmodel:\n  id: gpt-4\n  provider: OpenAI\n",
    );
    assert!(
        matches!(err, DeclarativeError::NoClientFactory(_)),
        "got {err:?}"
    );
}

#[test]
fn unbound_function_tool_becomes_declaration_only() {
    // An inline function tool with no registry binding still parses and builds
    // (as a non-executable declaration); the agent constructs successfully.
    let loader = DeclarativeLoader::new().with_client_factory(
        ChatClientFactory::new().with_default(|_| Ok(Arc::new(MockClient::always("x")) as _)),
    );
    let yaml = "kind: Prompt\nname: A\ntools:\n  - kind: function\n    name: do_thing\n    description: does a thing\n";
    let agent = loader
        .load_agent(yaml)
        .expect("declaration-only tool builds");
    assert_eq!(agent.name(), Some("A"));
}
