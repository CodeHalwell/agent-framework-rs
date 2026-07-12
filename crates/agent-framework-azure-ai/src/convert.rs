//! Conversion between framework types and the Azure AI Foundry (persistent
//! agents) wire format, plus parsing of run/message/step JSON.
//!
//! The Azure AI Agents data plane is Assistants-API-shaped: agents (a.k.a.
//! assistants), threads, messages, and runs, with tool calls surfaced as a
//! run's `required_action` and answered via `submit_tool_outputs`. These
//! helpers are pure so they can be unit-tested against fixture JSON without a
//! network.

use agent_framework_core::error::{Error, Result};
use agent_framework_core::tools::{ToolDefinition, ToolKind};
use agent_framework_core::types::{
    ChatMessage, ChatOptions, Content, FinishReason, FunctionArguments, FunctionCallContent,
    FunctionResultContent, Role, ToolMode, UsageDetails,
};
use serde_json::{json, Map, Value};

/// The Entra ID scope (audience) for the Azure AI Foundry data plane.
pub const AI_FOUNDRY_SCOPE: &str = "https://ai.azure.com/.default";

// ---------------------------------------------------------------------------
// Tool-call id round-tripping
// ---------------------------------------------------------------------------

/// Encode a `(run_id, tool_call_id)` pair into the synthetic call id carried by
/// a [`FunctionCallContent`], mirroring the Python client's
/// `["<run_id>", "<tool_call_id>"]` JSON-array convention. The run id is needed
/// when the tool result is submitted back (`submit_tool_outputs` targets a run),
/// but only the bare `tool_call_id` is sent on the wire.
pub fn encode_call_id(run_id: &str, tool_call_id: &str) -> String {
    Value::Array(vec![json!(run_id), json!(tool_call_id)]).to_string()
}

/// Decode a synthetic call id produced by [`encode_call_id`] back into
/// `(run_id, tool_call_id)`. Returns `None` when the id is not the expected
/// two-element array of non-empty strings.
pub fn decode_call_id(call_id: &str) -> Option<(String, String)> {
    let v: Value = serde_json::from_str(call_id).ok()?;
    let arr = v.as_array()?;
    if arr.len() != 2 {
        return None;
    }
    let run = arr[0].as_str()?.to_string();
    let call = arr[1].as_str()?.to_string();
    if run.is_empty() || call.is_empty() {
        return None;
    }
    Some((run, call))
}

// ---------------------------------------------------------------------------
// Request-message preparation
// ---------------------------------------------------------------------------

/// The result of splitting a request's messages into the pieces the Azure AI
/// Agents API consumes separately.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PreparedMessages {
    /// System/developer text folded into run instructions (Azure AI has no
    /// system message role).
    pub instructions: Option<String>,
    /// Thread messages to create before the run: `{"role","content":[blocks]}`.
    pub messages: Vec<Value>,
    /// Decoded tool outputs `(tool_call_id, output)` to submit to an active run.
    pub tool_outputs: Vec<(String, String)>,
    /// Decoded tool approvals `(tool_call_id, approved)` to submit.
    pub tool_approvals: Vec<(String, bool)>,
    /// The run id shared by the tool results, if any were present.
    pub run_id: Option<String>,
}

/// Serialize a tool result the way `submit_tool_outputs` expects: a failed
/// invocation surfaces its error text (so the model can see the failure and
/// retry/repair), a bare string is sent verbatim, anything else is
/// JSON-encoded. Mirrors the OpenAI converters' `result_to_string`.
fn result_to_output(fr: &FunctionResultContent) -> String {
    if let Some(exc) = &fr.exception {
        return format!("error: {exc}");
    }
    match &fr.result {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => String::new(),
    }
}

/// Split request messages into instructions, thread messages, and tool results.
///
/// * `system`/`developer` messages become instructions.
/// * `assistant` messages become `{"role":"assistant"}` thread messages;
///   everything else becomes `{"role":"user"}`.
/// * text becomes a `{"type":"text","text":...}` block; image data/URI content
///   becomes an `{"type":"image_url","image_url":{"url":...}}` block.
/// * function results / approval responses are pulled out as tool results
///   (decoded from their synthetic `[run_id, call_id]` call ids) rather than
///   added as messages.
pub fn prepare_messages(messages: &[ChatMessage]) -> PreparedMessages {
    let mut out = PreparedMessages::default();
    let mut instructions: Vec<String> = Vec::new();

    for msg in messages {
        let role = msg.role.as_str();
        if role == Role::SYSTEM || role == "developer" {
            for content in &msg.contents {
                if let Content::Text(t) = content {
                    instructions.push(t.text.clone());
                }
            }
            continue;
        }

        let mut blocks: Vec<Value> = Vec::new();
        for content in &msg.contents {
            match content {
                Content::Text(t) => blocks.push(json!({"type": "text", "text": t.text})),
                Content::Uri(u) if u.media_type.starts_with("image") => {
                    blocks.push(json!({"type": "image_url", "image_url": {"url": u.uri}}))
                }
                Content::Data(d) if is_image(d.media_type.as_deref()) => {
                    blocks.push(json!({"type": "image_url", "image_url": {"url": d.uri}}))
                }
                Content::FunctionResult(fr) => {
                    if let Some((run_id, call_id)) = decode_call_id(&fr.call_id) {
                        out.run_id.get_or_insert(run_id);
                        out.tool_outputs.push((call_id, result_to_output(fr)));
                    }
                }
                Content::FunctionApprovalResponse(ar) => {
                    if let Some((run_id, call_id)) = decode_call_id(&ar.id) {
                        out.run_id.get_or_insert(run_id);
                        out.tool_approvals.push((call_id, ar.approved));
                    }
                }
                _ => {}
            }
        }

        if !blocks.is_empty() {
            let wire_role = if role == Role::ASSISTANT {
                "assistant"
            } else {
                "user"
            };
            out.messages
                .push(json!({"role": wire_role, "content": blocks}));
        }
    }

    if !instructions.is_empty() {
        out.instructions = Some(instructions.join("\n"));
    }
    out
}

fn is_image(media_type: Option<&str>) -> bool {
    media_type.map(|m| m.starts_with("image")).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Tool definitions & tool_choice
// ---------------------------------------------------------------------------

/// The result of preparing a request's tools for the Azure AI wire format.
///
/// Building this mirrors the Python client's `_prep_tools`
/// (`azure-ai/.../_chat_client.py:924-998`); `file_search_resources` and
/// `mcp_resources` mirror the `tool_resources` merging that happens around
/// it in `_create_run_options` (`_chat_client.py:807-842, 984-991`). They are
/// kept separate from `tools` (rather than assembled into a `tool_resources`
/// object here) so [`build_run_body`] can apply upstream's precedence when
/// merging with an existing agent's own `tool_resources`: a `file_search`
/// tool's resources only apply when nothing has already claimed
/// `tool_resources`, but `mcp` resources always overlay.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PreparedTools {
    /// The `tools` array entries (Assistants wire shape).
    pub tools: Vec<Value>,
    /// `tool_resources.file_search`, from the first
    /// [`ToolKind::HostedFileSearch`] tool whose parameters carry
    /// `vector_store_ids`.
    pub file_search_resources: Option<Value>,
    /// `tool_resources.mcp`: a JSON array with one entry per
    /// [`ToolKind::HostedMcp`] tool, built only when at least one is present
    /// (mirrors upstream's `if mcp_tools:` guard).
    pub mcp_resources: Option<Value>,
}

/// Map framework [`ToolDefinition`]s to Azure AI tool payloads (Assistants
/// shape) plus any `tool_resources` fields they require. Function tools
/// become `{"type":"function","function":{…}}`; hosted markers map to their
/// service tool types.
///
/// Mirrors the Python client's `_prep_tools` (`_chat_client.py:924-998`) and
/// its MCP `tool_resources`/`require_approval` handling
/// (`_chat_client.py:807-842`).
///
/// # Errors
/// A [`ToolKind::HostedWebSearch`] tool errors unless its `parameters` carry
/// either a Bing Grounding `connection_id`, or both a Bing Custom Search
/// `custom_connection_id` and `instance_name` — mirroring upstream's
/// `ServiceInitializationError` (`_chat_client.py:966-973`): the service
/// rejects a `bing_grounding`/`bing_custom_search` tool with no connection
/// attached, so upstream fails fast rather than emitting a broken tool, and
/// this does too.
pub fn tools_to_azure(tools: &[ToolDefinition]) -> Result<PreparedTools> {
    let mut out = PreparedTools::default();
    for tool in tools {
        append_tool(tool, &mut out)?;
    }
    Ok(out)
}

fn append_tool(tool: &ToolDefinition, out: &mut PreparedTools) -> Result<()> {
    match &tool.kind {
        ToolKind::Function => out.tools.push(json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        })),
        ToolKind::HostedCodeInterpreter => out.tools.push(json!({"type": "code_interpreter"})),
        ToolKind::HostedFileSearch { .. } => {
            out.tools.push(json!({"type": "file_search"}));
            // Mirrors `_prep_tools`'s `HostedFileSearchTool()` branch
            // (`_chat_client.py:984-991`): the marker itself carries no
            // store ids, so read them from `parameters` (the same
            // convention the OpenAI Responses client uses).
            if out.file_search_resources.is_none() {
                if let Some(ids) = tool.parameters.get("vector_store_ids") {
                    out.file_search_resources = Some(json!({ "vector_store_ids": ids }));
                }
            }
        }
        ToolKind::HostedWebSearch => out.tools.push(bing_tool_to_azure(tool)?),
        ToolKind::HostedMcp { url, allowed_tools } => {
            out.tools
                .push(mcp_tool_definition(tool, url, allowed_tools));
            let list = out
                .mcp_resources
                .get_or_insert_with(|| Value::Array(Vec::new()));
            list.as_array_mut()
                .expect("mcp_resources is always seeded as a JSON array")
                .push(mcp_tool_resource(tool));
        }
    }
    Ok(())
}

/// Build a `bing_grounding` or `bing_custom_search` tool entry from a
/// [`ToolKind::HostedWebSearch`] tool's `parameters`, mirroring the Bing
/// branch of `_prep_tools` (`_chat_client.py:933-974`) — including its two
/// *independent* `if`s (not `if`/`elif`): a fully-specified Bing Custom
/// Search pair (`custom_connection_id` + `instance_name`) overrides a plain
/// `connection_id`, but merely having a `custom_connection_id` (even without
/// `instance_name`) disqualifies the plain-grounding path too. Wire shape
/// verified against `azure.ai.agents.models` (`BingGroundingToolDefinition`/
/// `BingCustomSearchToolDefinition`, pinned `azure-ai-agents==1.2.0b5`):
/// `{"type":"bing_grounding","bing_grounding":{"search_configurations":[{...}]}}`.
/// `count`/`freshness`/`market`/`set_lang` are copied through only when
/// present, mirroring `_prep_tools`'s own `config_args`, which only forwards
/// them when the caller supplied them.
fn bing_tool_to_azure(tool: &ToolDefinition) -> Result<Value> {
    let params = &tool.parameters;
    let str_param = |key: &str| params.get(key).and_then(Value::as_str);

    let connection_id = str_param("connection_id");
    let custom_connection_id = str_param("custom_connection_id");
    let instance_name = str_param("instance_name");

    let mut config = Map::new();
    for key in ["count", "freshness", "market", "set_lang"] {
        if let Some(v) = params.get(key).filter(|v| !v.is_null()) {
            config.insert(key.to_string(), v.clone());
        }
    }

    let mut result = None;
    if let Some(connection_id) = connection_id {
        if custom_connection_id.is_none() && instance_name.is_none() {
            let mut search_config = Map::new();
            search_config.insert("connection_id".into(), json!(connection_id));
            search_config.extend(config.clone());
            result = Some(json!({
                "type": "bing_grounding",
                "bing_grounding": { "search_configurations": [Value::Object(search_config)] }
            }));
        }
    }
    if let (Some(custom_connection_id), Some(instance_name)) = (custom_connection_id, instance_name)
    {
        let mut search_config = Map::new();
        search_config.insert("connection_id".into(), json!(custom_connection_id));
        search_config.insert("instance_name".into(), json!(instance_name));
        search_config.extend(config);
        result = Some(json!({
            "type": "bing_custom_search",
            "bing_custom_search": { "search_configurations": [Value::Object(search_config)] }
        }));
    }

    result.ok_or_else(|| {
        Error::Configuration(
            "hosted web-search tool requires either 'connection_id' (Bing Grounding) or both \
             'custom_connection_id' and 'instance_name' (Bing Custom Search) in its parameters"
                .into(),
        )
    })
}

/// The `mcp` tools-array entry. Unlike `bing_grounding`/`file_search`, Azure's
/// `MCPToolDefinition` keeps `server_label`/`server_url`/`allowed_tools` at
/// the top level rather than nested under an `"mcp"` key — unchanged from
/// before this fix, verified against `_prep_tools`'s `HostedMCPTool()` branch
/// (`_chat_client.py:977-983`).
fn mcp_tool_definition(
    tool: &ToolDefinition,
    url: &str,
    allowed_tools: &Option<Vec<String>>,
) -> Value {
    let mut mcp = Map::new();
    mcp.insert("type".into(), json!("mcp"));
    mcp.insert("server_label".into(), json!(tool.name.replace(' ', "_")));
    mcp.insert("server_url".into(), json!(url));
    if let Some(allowed) = allowed_tools {
        mcp.insert("allowed_tools".into(), json!(allowed));
    }
    Value::Object(mcp)
}

/// The `tool_resources.mcp[]` entry for a hosted MCP tool: `headers` and
/// `approval_mode` read from the tool's `parameters`, mirroring the MCP
/// branch of `_create_run_options` (`_chat_client.py:807-837`). Every MCP
/// tool gets an entry (at minimum `{"server_label": ...}`), matching
/// upstream's unconditional `mcp_resources.append(mcp_resource)`.
///
/// `approval_mode` accepts either a string (`"always_require"` mapped to
/// `"always"`, anything else to `"never"`) or an object with an `"always"` or
/// `"never"` key naming specific tools (`{"always": [...]}`/
/// `{"never": [...]}`) — copied through as-is, mirroring
/// `mcp_tool.approval_mode["always_require_approval"]`/
/// `["never_require_approval"]`, which are already bare tool-name lists.
fn mcp_tool_resource(tool: &ToolDefinition) -> Value {
    let mut resource = Map::new();
    resource.insert("server_label".into(), json!(tool.name.replace(' ', "_")));

    if let Some(headers) = tool
        .parameters
        .get("headers")
        .filter(|h| h.as_object().is_some_and(|o| !o.is_empty()))
    {
        resource.insert("headers".into(), headers.clone());
    }

    match tool.parameters.get("approval_mode") {
        Some(Value::String(s)) => {
            let mapped = if s == "always_require" {
                "always"
            } else {
                "never"
            };
            resource.insert("require_approval".into(), json!(mapped));
        }
        Some(Value::Object(obj)) => {
            // Both keys may be present: the service's `MCPApprovalPerTool`
            // model carries `always` and `never` as optional siblings
            // (verified in `azure-ai-agents` 1.2.0b5). Upstream Python's
            // `elif` drops `never` whenever `always` is set — a deliberate
            // divergence here so a PerTool config with both lists survives.
            let mut approval = Map::new();
            if let Some(always) = obj.get("always") {
                approval.insert("always".into(), always.clone());
            }
            if let Some(never) = obj.get("never") {
                approval.insert("never".into(), never.clone());
            }
            if !approval.is_empty() {
                resource.insert("require_approval".into(), Value::Object(approval));
            }
        }
        _ => {}
    }

    Value::Object(resource)
}

/// Map a [`ToolMode`] to an Azure AI `tool_choice` value: `"none"`, `"auto"`,
/// `"required"`, or a specific `{"type":"function","function":{"name":…}}`.
pub fn tool_choice_to_azure(mode: &ToolMode) -> Value {
    match mode {
        ToolMode::None => json!("none"),
        ToolMode::Auto => json!("auto"),
        ToolMode::Required(None) => json!("required"),
        ToolMode::Required(Some(name)) => {
            json!({"type": "function", "function": {"name": name}})
        }
    }
}

// ---------------------------------------------------------------------------
// Existing-agent definition merge
// ---------------------------------------------------------------------------

/// Non-function tool entries from an existing agent's `tools` array
/// (Assistants wire shape) — replays a persistent agent's own hosted tools
/// onto every run, mirroring `_create_run_options`'s `agent_tools = [tool
/// for tool in agent_definition.tools if not isinstance(tool,
/// FunctionToolDefinition)]` (`_chat_client.py:794-798`). Function tools are
/// excluded because they're passed through `ChatOptions::tools` instead.
fn agent_definition_tools(agent: &Value) -> Vec<Value> {
    agent
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter(|t| t.get("type").and_then(Value::as_str) != Some("function"))
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

/// The existing agent's `tool_resources` object, if non-empty — the seed
/// `_create_run_options` starts `run_options["tool_resources"]` from
/// (`_chat_client.py:799-800`) before [`build_run_body`] layers
/// `file_search`/`mcp` resources on top.
fn agent_definition_tool_resources(agent: &Value) -> Option<Map<String, Value>> {
    agent
        .get("tool_resources")
        .and_then(Value::as_object)
        .filter(|o| !o.is_empty())
        .cloned()
}

/// The existing agent's `instructions`, if non-empty.
fn agent_definition_instructions(agent: &Value) -> Option<&str> {
    agent
        .get("instructions")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// Prepend an existing agent's own instructions ahead of the per-request
/// instructions, unless already present. Mirrors `_create_run_options`: `if
/// agent_definition.instructions and agent_definition.instructions not in
/// instructions: instructions.insert(0, agent_definition.instructions)`
/// (`_chat_client.py:911-917`) — joined here with `"\n"` rather than
/// upstream's bare concatenation, consistent with how this port already
/// joins instruction fragments elsewhere (see `prepare_messages`).
fn merge_agent_instructions(
    instructions: Option<&str>,
    agent_definition: Option<&Value>,
) -> Option<String> {
    let agent_instructions = agent_definition.and_then(agent_definition_instructions);
    match (agent_instructions, instructions) {
        (Some(a), Some(b)) if b.contains(a) => Some(b.to_string()),
        (Some(a), Some(b)) => Some(format!("{a}\n{b}")),
        (Some(a), None) => Some(a.to_string()),
        (None, Some(b)) => Some(b.to_string()),
        (None, None) => None,
    }
}

// ---------------------------------------------------------------------------
// Run / agent request bodies
// ---------------------------------------------------------------------------

/// Build the `POST …/threads/{id}/runs` body for `agent_id`, applying the
/// request `options` and `instructions`, and — when `agent_definition` is
/// supplied (an existing/persistent agent's fetched or just-created
/// definition) — merging in its own tools/instructions/tool_resources the
/// way `_create_run_options` does (`_chat_client.py:771-922`). `stream`
/// toggles SSE.
///
/// # Errors
/// Propagates [`tools_to_azure`]'s error for a misconfigured hosted
/// web-search tool.
pub fn build_run_body(
    agent_id: &str,
    model: Option<&str>,
    instructions: Option<&str>,
    options: &ChatOptions,
    agent_definition: Option<&Value>,
    stream: bool,
) -> Result<Value> {
    let mut body = Map::new();
    body.insert("assistant_id".into(), json!(agent_id));
    if let Some(m) = model {
        body.insert("model".into(), json!(m));
    }
    let merged_instructions = merge_agent_instructions(instructions, agent_definition);
    if let Some(instr) = &merged_instructions {
        body.insert("instructions".into(), json!(instr));
    }
    apply_common_run_fields(&mut body, options);

    // Tools + tool_resources. An existing agent's own (non-function) tools
    // and tool_resources always apply; the request's own tools are merged in
    // only when tool_choice allows (mirrors `_create_run_options`: the
    // agent-definition tools/resources are added unconditionally, but
    // `chat_options.tools` only under `if chat_options.tool_choice is not
    // None: if chat_options.tool_choice != "none" and chat_options.tools:`).
    let mut tools_list: Vec<Value> = agent_definition
        .map(agent_definition_tools)
        .unwrap_or_default();
    let mut tool_resources: Map<String, Value> = agent_definition
        .and_then(agent_definition_tool_resources)
        .unwrap_or_default();

    let tool_choice = options.tool_choice.clone().unwrap_or(ToolMode::None);
    if tool_choice != ToolMode::None && !options.tools.is_empty() {
        let prepared = tools_to_azure(&options.tools)?;
        tools_list.extend(prepared.tools);
        // A fresh `file_search`'s resources only apply when nothing has
        // already claimed `tool_resources` (mirrors `_prep_tools`'s `if
        // run_options is not None and "tool_resources" not in run_options`
        // guard, `_chat_client.py:990-991`).
        if tool_resources.is_empty() {
            if let Some(fs) = prepared.file_search_resources {
                tool_resources.insert("file_search".into(), fs);
            }
        }
        // `mcp` resources always overlay, creating `tool_resources` if
        // needed (`_chat_client.py:840-842`).
        if let Some(mcp) = prepared.mcp_resources {
            tool_resources.insert("mcp".into(), mcp);
        }
    }
    if !tools_list.is_empty() {
        body.insert("tools".into(), json!(tools_list));
    }
    if !tool_resources.is_empty() {
        body.insert("tool_resources".into(), json!(tool_resources));
    }
    if options.tool_choice.is_some() {
        body.insert("tool_choice".into(), tool_choice_to_azure(&tool_choice));
    }

    if stream {
        body.insert("stream".into(), json!(true));
    }
    Ok(Value::Object(body))
}

/// Build the `POST …/assistants` body to auto-create an agent.
///
/// # Errors
/// Propagates [`tools_to_azure`]'s error for a misconfigured hosted
/// web-search tool.
pub fn build_agent_body(
    model: &str,
    name: Option<&str>,
    description: Option<&str>,
    instructions: Option<&str>,
    options: &ChatOptions,
) -> Result<Value> {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    if let Some(n) = name {
        body.insert("name".into(), json!(n));
    }
    if let Some(d) = description {
        body.insert("description".into(), json!(d));
    }
    if let Some(instr) = instructions {
        body.insert("instructions".into(), json!(instr));
    }
    if !options.tools.is_empty() {
        let prepared = tools_to_azure(&options.tools)?;
        if !prepared.tools.is_empty() {
            body.insert("tools".into(), json!(prepared.tools));
        }
        // Mirrors `_get_agent_id_or_create`'s `if "tool_resources" in
        // run_options: args["tool_resources"] = run_options["tool_resources"]`
        // (`_chat_client.py:321-322`): a freshly auto-created agent gets the
        // same `file_search`/`mcp` resources its own hosted tools need.
        let mut tool_resources = Map::new();
        if let Some(fs) = prepared.file_search_resources {
            tool_resources.insert("file_search".into(), fs);
        }
        if let Some(mcp) = prepared.mcp_resources {
            tool_resources.insert("mcp".into(), mcp);
        }
        if !tool_resources.is_empty() {
            body.insert("tool_resources".into(), json!(tool_resources));
        }
    }
    if let Some(temp) = options.temperature {
        body.insert("temperature".into(), json!(temp));
    }
    if let Some(top_p) = options.top_p {
        body.insert("top_p".into(), json!(top_p));
    }
    if let Some(rf) = &options.response_format {
        body.insert("response_format".into(), response_format_to_azure(rf));
    }
    Ok(Value::Object(body))
}

fn apply_common_run_fields(body: &mut Map<String, Value>, options: &ChatOptions) {
    if let Some(t) = options.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(p) = options.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if let Some(m) = options.max_tokens {
        body.insert("max_completion_tokens".into(), json!(m));
    }
    if let Some(parallel) = options.allow_multiple_tool_calls {
        body.insert("parallel_tool_calls".into(), json!(parallel));
    }
    if let Some(rf) = &options.response_format {
        body.insert("response_format".into(), response_format_to_azure(rf));
    }
    if let Some(meta) = &options.metadata {
        body.insert("metadata".into(), json!(meta));
    }
}

fn response_format_to_azure(rf: &agent_framework_core::types::ResponseFormat) -> Value {
    // The core `ResponseFormat` already serializes to the OpenAI/Assistants
    // `response_format` object shape.
    serde_json::to_value(rf).unwrap_or(Value::Null)
}

/// Build the `submit_tool_outputs` body from decoded outputs/approvals.
pub fn build_submit_body(
    tool_outputs: &[(String, String)],
    tool_approvals: &[(String, bool)],
    stream: bool,
) -> Value {
    let mut body = Map::new();
    if !tool_outputs.is_empty() {
        let outputs: Vec<Value> = tool_outputs
            .iter()
            .map(|(id, output)| json!({"tool_call_id": id, "output": output}))
            .collect();
        body.insert("tool_outputs".into(), json!(outputs));
    }
    if !tool_approvals.is_empty() {
        let approvals: Vec<Value> = tool_approvals
            .iter()
            .map(|(id, approve)| json!({"tool_call_id": id, "approve": approve}))
            .collect();
        body.insert("tool_approvals".into(), json!(approvals));
    }
    if stream {
        body.insert("stream".into(), json!(true));
    }
    Value::Object(body)
}

// ---------------------------------------------------------------------------
// Response / run parsing
// ---------------------------------------------------------------------------

/// The `status` field of a run object (`"queued"`, `"in_progress"`,
/// `"requires_action"`, `"completed"`, `"failed"`, `"cancelled"`, `"expired"`).
pub fn run_status(run: &Value) -> Option<&str> {
    run.get("status").and_then(Value::as_str)
}

/// A run is terminal when it will not progress further without a new request.
pub fn is_terminal_status(status: &str) -> bool {
    matches!(
        status,
        "completed" | "failed" | "cancelled" | "expired" | "requires_action"
    )
}

/// Parse `usage: {prompt_tokens, completion_tokens, total_tokens}` into
/// [`UsageDetails`].
pub fn parse_usage(value: &Value) -> Option<UsageDetails> {
    let usage = value.get("usage").filter(|u| u.is_object())?;
    Some(UsageDetails {
        input_token_count: usage.get("prompt_tokens").and_then(Value::as_u64),
        output_token_count: usage.get("completion_tokens").and_then(Value::as_u64),
        total_token_count: usage.get("total_tokens").and_then(Value::as_u64),
        ..Default::default()
    })
}

/// Build [`FunctionCallContent`]/approval contents from a run's
/// `required_action`, encoding each call id as `[run_id, tool_call_id]`.
pub fn required_action_contents(run: &Value, run_id: &str) -> Vec<Content> {
    let Some(required) = run.get("required_action") else {
        return Vec::new();
    };
    let action_type = required.get("type").and_then(Value::as_str).unwrap_or("");
    let mut out = Vec::new();

    if action_type == "submit_tool_outputs" {
        let calls = required
            .get("submit_tool_outputs")
            .and_then(|s| s.get("tool_calls"))
            .and_then(Value::as_array);
        for call in calls.into_iter().flatten() {
            if call.get("type").and_then(Value::as_str) != Some("function") {
                continue;
            }
            let tool_call_id = call.get("id").and_then(Value::as_str).unwrap_or_default();
            let func = call.get("function");
            let name = func
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let args = func
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str)
                .unwrap_or("");
            out.push(Content::FunctionCall(FunctionCallContent::new(
                encode_call_id(run_id, tool_call_id),
                name,
                Some(FunctionArguments::Raw(args.to_string())),
            )));
        }
    } else if action_type == "submit_tool_approval" {
        let calls = required
            .get("submit_tool_approval")
            .and_then(|s| s.get("tool_calls"))
            .and_then(Value::as_array);
        for call in calls.into_iter().flatten() {
            let tool_call_id = call.get("id").and_then(Value::as_str).unwrap_or_default();
            let name = call.get("name").and_then(Value::as_str).unwrap_or_default();
            let args = call.get("arguments").and_then(Value::as_str).unwrap_or("");
            let encoded = encode_call_id(run_id, tool_call_id);
            out.push(Content::FunctionApprovalRequest(
                agent_framework_core::types::FunctionApprovalRequestContent {
                    id: encoded.clone(),
                    function_call: FunctionCallContent::new(
                        encoded,
                        name,
                        Some(FunctionArguments::Raw(args.to_string())),
                    ),
                },
            ));
        }
    }
    out
}

/// The finish reason for a terminal run status.
pub fn finish_reason_for(status: &str) -> Option<FinishReason> {
    match status {
        "completed" => Some(FinishReason::stop()),
        "requires_action" => Some(FinishReason::tool_calls()),
        "cancelled" | "expired" => Some(FinishReason::new(status)),
        _ => None,
    }
}

/// Extract assistant text from a `…/threads/{id}/messages` list response,
/// concatenating the text of every content block of assistant-role messages
/// (the list is expected ordered oldest→newest).
pub fn assistant_text_from_messages(list: &Value) -> String {
    let Some(data) = list.get("data").and_then(Value::as_array) else {
        return String::new();
    };
    let mut out = String::new();
    for msg in data {
        if msg.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        for block in msg
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if block.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text) = block
                    .get("text")
                    .and_then(|t| t.get("value"))
                    .and_then(Value::as_str)
                {
                    out.push_str(text);
                }
            }
        }
    }
    out
}

/// The last-error message on a failed run.
pub fn last_error_message(run: &Value) -> String {
    run.get("last_error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("run failed")
        .to_string()
}

/// The `code` field of a failed run's `last_error`, if present (e.g.
/// `"content_filter"`, `"rate_limit_exceeded"`, `"server_error"`).
pub fn last_error_code(run: &Value) -> Option<&str> {
    run.get("last_error")?.get("code")?.as_str()
}

/// Classify a failed run's `last_error` into an [`Error`], distinguishing a
/// content-filter refusal from every other failure reason.
///
/// Upstream (`agent_framework_azure_ai/_chat_client.py`, `case
/// AgentStreamEvent.THREAD_RUN_FAILED: raise
/// ServiceResponseException(event_data.last_error.message)`) does not
/// classify `last_error.code` at all — every run failure becomes the same
/// generic exception. This extends that with the one distinction Azure AI
/// Foundry's wire format carries that upstream's exception hierarchy already
/// has a dedicated class for: `last_error.code == "content_filter"` ->
/// [`Error::ServiceContentFilter`] (mirroring `ServiceContentFilterException`);
/// every other code (or none at all) stays the same plain [`Error::Service`]
/// Python constructs. Shared by both the polling
/// (`AzureAIAgentClient::response_from_run`'s `"failed"` arm) and streaming
/// (`crate::sse`'s `"thread.run.failed"` handling) run-failure paths.
pub fn classify_last_error(run: &Value) -> Error {
    let message = last_error_message(run);
    match last_error_code(run) {
        Some("content_filter") => Error::service_content_filter(message),
        _ => Error::service(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::{
        hosted_code_interpreter, hosted_file_search, hosted_mcp, hosted_web_search, FunctionTool,
    };
    use agent_framework_core::types::FunctionResultContent;

    #[test]
    fn call_id_round_trips() {
        let encoded = encode_call_id("run_1", "call_9");
        assert_eq!(
            decode_call_id(&encoded),
            Some(("run_1".into(), "call_9".into()))
        );
    }

    #[test]
    fn decode_rejects_malformed_call_ids() {
        assert_eq!(decode_call_id("not-json"), None);
        assert_eq!(decode_call_id("[\"run\"]"), None);
        assert_eq!(decode_call_id("[\"\", \"call\"]"), None);
    }

    #[test]
    fn system_messages_become_instructions() {
        let prepared =
            prepare_messages(&[ChatMessage::system("be terse"), ChatMessage::user("hi")]);
        assert_eq!(prepared.instructions.as_deref(), Some("be terse"));
        assert_eq!(prepared.messages.len(), 1);
        assert_eq!(prepared.messages[0]["role"], json!("user"));
        assert_eq!(prepared.messages[0]["content"][0]["text"], json!("hi"));
    }

    #[test]
    fn function_results_are_pulled_out_as_tool_outputs() {
        let call_id = encode_call_id("run_7", "call_3");
        let msg = ChatMessage::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent::new(
                call_id,
                Some(json!("sunny")),
            ))],
        );
        let prepared = prepare_messages(&[msg]);
        assert!(prepared.messages.is_empty());
        assert_eq!(prepared.run_id.as_deref(), Some("run_7"));
        assert_eq!(
            prepared.tool_outputs,
            vec![("call_3".into(), "sunny".into())]
        );
    }

    #[test]
    fn failed_tool_result_submits_its_error_text() {
        // A failed local tool (exception set, result None) must surface the
        // error to the model via submit_tool_outputs, not an empty string.
        let msg = ChatMessage::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent {
                call_id: encode_call_id("run_7", "call_3"),
                result: None,
                exception: Some("boom".into()),
            })],
        );
        let prepared = prepare_messages(&[msg]);
        assert_eq!(
            prepared.tool_outputs,
            vec![("call_3".into(), "error: boom".into())]
        );
    }

    #[test]
    fn run_body_includes_tools_only_when_choice_allows() {
        let tool = FunctionTool::new(
            "get_weather",
            "Get weather",
            json!({"type": "object", "properties": {}}),
            |_| async { Ok(json!("ok")) },
        )
        .into_definition();
        let options = ChatOptions::new()
            .with_tool(tool)
            .with_tool_choice(ToolMode::Auto)
            .with_temperature(0.5);
        let body =
            build_run_body("asst_1", Some("gpt-4o"), Some("sys"), &options, None, true).unwrap();
        assert_eq!(body["assistant_id"], json!("asst_1"));
        assert_eq!(body["model"], json!("gpt-4o"));
        assert_eq!(body["instructions"], json!("sys"));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["tool_choice"], json!("auto"));
        assert_eq!(body["tools"][0]["function"]["name"], json!("get_weather"));
        assert_eq!(body["temperature"], json!(0.5));
    }

    #[test]
    fn run_body_omits_tools_when_choice_none() {
        let tool = hosted_code_interpreter();
        let options = ChatOptions::new().with_tool(tool); // tool_choice unset
        let body = build_run_body("asst_1", None, None, &options, None, false).unwrap();
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn hosted_tools_map_to_service_types() {
        let tools = vec![hosted_code_interpreter(), hosted_file_search(Some(3))];
        let prepared = tools_to_azure(&tools).unwrap();
        assert_eq!(prepared.tools[0], json!({"type": "code_interpreter"}));
        assert_eq!(prepared.tools[1], json!({"type": "file_search"}));
        assert!(
            prepared.file_search_resources.is_none(),
            "no vector_store_ids were supplied via parameters"
        );
        assert!(prepared.mcp_resources.is_none());
    }

    // -- bing_grounding / bing_custom_search -------------------------------

    #[test]
    fn bing_grounding_requires_a_connection() {
        let tool = hosted_web_search(); // empty_schema() parameters: no connection_id
        let err = tools_to_azure(std::slice::from_ref(&tool)).unwrap_err();
        assert!(err.to_string().contains("connection_id"), "{err}");
    }

    #[test]
    fn bing_grounding_wire_shape_with_optional_config() {
        let mut tool = hosted_web_search();
        tool.parameters = json!({
            "connection_id": "conn-1",
            "count": 7,
            "freshness": "Week",
            "market": "en-US",
            "set_lang": "en",
        });
        let prepared = tools_to_azure(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(
            prepared.tools[0],
            json!({
                "type": "bing_grounding",
                "bing_grounding": {
                    "search_configurations": [{
                        "connection_id": "conn-1",
                        "count": 7,
                        "freshness": "Week",
                        "market": "en-US",
                        "set_lang": "en",
                    }]
                }
            })
        );
    }

    #[test]
    fn bing_grounding_omits_optional_fields_when_absent() {
        let mut tool = hosted_web_search();
        tool.parameters = json!({"connection_id": "conn-1"});
        let prepared = tools_to_azure(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(
            prepared.tools[0]["bing_grounding"]["search_configurations"][0],
            json!({"connection_id": "conn-1"})
        );
    }

    #[test]
    fn bing_custom_search_wire_shape() {
        let mut tool = hosted_web_search();
        tool.parameters = json!({
            "custom_connection_id": "custom-conn",
            "instance_name": "my-instance",
        });
        let prepared = tools_to_azure(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(
            prepared.tools[0],
            json!({
                "type": "bing_custom_search",
                "bing_custom_search": {
                    "search_configurations": [{
                        "connection_id": "custom-conn",
                        "instance_name": "my-instance",
                    }]
                }
            })
        );
    }

    #[test]
    fn bing_custom_search_overrides_a_plain_connection_id() {
        // All three present: upstream's independent `if`s mean the fully
        // specified custom pair wins over plain `connection_id`.
        let mut tool = hosted_web_search();
        tool.parameters = json!({
            "connection_id": "plain-conn",
            "custom_connection_id": "custom-conn",
            "instance_name": "my-instance",
        });
        let prepared = tools_to_azure(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(prepared.tools[0]["type"], json!("bing_custom_search"));
    }

    #[test]
    fn bing_grounding_disqualified_by_a_partial_custom_pair() {
        // `custom_connection_id` present but `instance_name` missing: matches
        // upstream's `not custom_connection_id` guard on the grounding
        // branch, so *neither* branch fires and this errors even though a
        // valid plain `connection_id` was also supplied.
        let mut tool = hosted_web_search();
        tool.parameters = json!({
            "connection_id": "plain-conn",
            "custom_connection_id": "custom-conn",
        });
        let err = tools_to_azure(std::slice::from_ref(&tool)).unwrap_err();
        assert!(err.to_string().contains("connection_id"), "{err}");
    }

    // -- file_search tool_resources -----------------------------------------

    #[test]
    fn file_search_vector_store_ids_produce_tool_resources() {
        let mut tool = hosted_file_search(None);
        tool.parameters = json!({"vector_store_ids": ["vs_1", "vs_2"]});
        let prepared = tools_to_azure(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(prepared.tools[0], json!({"type": "file_search"}));
        assert_eq!(
            prepared.file_search_resources,
            Some(json!({"vector_store_ids": ["vs_1", "vs_2"]}))
        );
    }

    // -- hosted_mcp headers / approval_mode ----------------------------------

    #[test]
    fn mcp_resource_always_includes_server_label_even_with_no_config() {
        let tool = hosted_mcp("my server", "https://mcp.example.com", None);
        let prepared = tools_to_azure(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(
            prepared.mcp_resources,
            Some(json!([{"server_label": "my_server"}]))
        );
    }

    #[test]
    fn mcp_resource_includes_headers_and_string_approval_mode() {
        let mut tool = hosted_mcp("my_server", "https://mcp.example.com", None);
        tool.parameters = json!({
            "headers": {"Authorization": "Bearer tok"},
            "approval_mode": "always_require",
        });
        let prepared = tools_to_azure(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(
            prepared.mcp_resources,
            Some(json!([{
                "server_label": "my_server",
                "headers": {"Authorization": "Bearer tok"},
                "require_approval": "always",
            }]))
        );
    }

    #[test]
    fn mcp_resource_never_require_string_mode() {
        let mut tool = hosted_mcp("my_server", "https://mcp.example.com", None);
        tool.parameters = json!({"approval_mode": "never_require"});
        let prepared = tools_to_azure(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(
            prepared.mcp_resources.unwrap()[0]["require_approval"],
            json!("never")
        );
    }

    #[test]
    fn mcp_resource_per_tool_approval_mode_object() {
        let mut tool = hosted_mcp("my_server", "https://mcp.example.com", None);
        tool.parameters = json!({"approval_mode": {"always": ["dangerous_tool"]}});
        let prepared = tools_to_azure(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(
            prepared.mcp_resources.unwrap()[0]["require_approval"],
            json!({"always": ["dangerous_tool"]})
        );

        let mut tool = hosted_mcp("my_server", "https://mcp.example.com", None);
        tool.parameters = json!({"approval_mode": {"never": ["safe_tool"]}});
        let prepared = tools_to_azure(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(
            prepared.mcp_resources.unwrap()[0]["require_approval"],
            json!({"never": ["safe_tool"]})
        );
    }

    #[test]
    fn mcp_resource_per_tool_approval_keeps_both_lists() {
        // The service's MCPApprovalPerTool model carries `always` and `never`
        // as optional siblings; a config with both must emit both (upstream
        // Python's elif drops `never` here — deliberate divergence).
        let mut tool = hosted_mcp("my_server", "https://mcp.example.com", None);
        tool.parameters =
            json!({"approval_mode": {"always": ["dangerous_tool"], "never": ["safe_tool"]}});
        let prepared = tools_to_azure(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(
            prepared.mcp_resources.unwrap()[0]["require_approval"],
            json!({"always": ["dangerous_tool"], "never": ["safe_tool"]})
        );
    }

    #[test]
    fn mcp_resources_collect_one_entry_per_tool() {
        let mut a = hosted_mcp("server a", "https://a.example.com", None);
        a.parameters = json!({"approval_mode": "always_require"});
        let b = hosted_mcp("server b", "https://b.example.com", None);
        let prepared = tools_to_azure(&[a, b]).unwrap();
        assert_eq!(prepared.tools.len(), 2);
        let mcp = prepared.mcp_resources.unwrap();
        assert_eq!(mcp.as_array().unwrap().len(), 2);
        assert_eq!(mcp[0]["server_label"], json!("server_a"));
        assert_eq!(mcp[1]["server_label"], json!("server_b"));
    }

    // -- existing-agent definition merge (build_run_body) --------------------

    #[test]
    fn agent_definition_tools_instructions_and_resources_are_merged() {
        let agent = json!({
            "instructions": "Be terse.",
            "tools": [
                {"type": "code_interpreter"},
                {"type": "function", "function": {"name": "ignored"}}
            ],
            "tool_resources": {"code_interpreter": {"file_ids": ["file_1"]}},
        });
        let options = ChatOptions::new();
        let body = build_run_body(
            "asst_1",
            None,
            Some("per-request instructions"),
            &options,
            Some(&agent),
            false,
        )
        .unwrap();
        // Only the non-function tool is replayed.
        assert_eq!(body["tools"], json!([{"type": "code_interpreter"}]));
        // Agent instructions are prepended ahead of the per-request ones.
        assert_eq!(
            body["instructions"],
            json!("Be terse.\nper-request instructions")
        );
        assert_eq!(
            body["tool_resources"],
            json!({"code_interpreter": {"file_ids": ["file_1"]}})
        );
    }

    #[test]
    fn agent_definition_instructions_not_duplicated_when_already_present() {
        let agent = json!({"instructions": "Be terse."});
        let body = build_run_body(
            "asst_1",
            None,
            Some("Be terse. Also say hi."),
            &ChatOptions::new(),
            Some(&agent),
            false,
        )
        .unwrap();
        assert_eq!(body["instructions"], json!("Be terse. Also say hi."));
    }

    #[test]
    fn agent_definition_tool_resources_win_over_a_fresh_file_search() {
        // The agent already has tool_resources; a fresh file_search tool's
        // own vector_store_ids are dropped (matches `_prep_tools`'s "only if
        // not already set" guard), but a fresh mcp tool's resources still
        // overlay (mcp always applies).
        let agent = json!({"tool_resources": {"code_interpreter": {"file_ids": ["f1"]}}});
        let mut fs = hosted_file_search(None);
        fs.parameters = json!({"vector_store_ids": ["vs_1"]});
        let mcp = hosted_mcp("srv", "https://mcp.example.com", None);
        let options = ChatOptions::new()
            .with_tool(fs)
            .with_tool(mcp)
            .with_tool_choice(ToolMode::Auto);
        let body = build_run_body("asst_1", None, None, &options, Some(&agent), false).unwrap();
        let resources = &body["tool_resources"];
        assert_eq!(resources["code_interpreter"], json!({"file_ids": ["f1"]}));
        assert!(
            resources.get("file_search").is_none(),
            "agent's own tool_resources should win: {resources}"
        );
        assert_eq!(resources["mcp"], json!([{"server_label": "srv"}]));
    }

    #[test]
    fn agent_definition_tools_apply_even_without_tool_choice() {
        // Upstream adds the agent's own tools unconditionally, unlike
        // `chat_options.tools` (gated on `tool_choice`).
        let agent = json!({"tools": [{"type": "code_interpreter"}]});
        let body = build_run_body(
            "asst_1",
            None,
            None,
            &ChatOptions::new(), // tool_choice unset
            Some(&agent),
            false,
        )
        .unwrap();
        assert_eq!(body["tools"], json!([{"type": "code_interpreter"}]));
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn no_agent_definition_leaves_run_body_unchanged() {
        let options = ChatOptions::new();
        let body = build_run_body("asst_1", None, Some("hi"), &options, None, false).unwrap();
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_resources").is_none());
        assert_eq!(body["instructions"], json!("hi"));
    }

    #[test]
    fn required_named_function_tool_choice() {
        let v = tool_choice_to_azure(&ToolMode::required_function("do_it"));
        assert_eq!(
            v,
            json!({"type": "function", "function": {"name": "do_it"}})
        );
    }

    #[test]
    fn required_action_builds_function_calls_with_encoded_ids() {
        let run = json!({
            "id": "run_42",
            "status": "requires_action",
            "required_action": {
                "type": "submit_tool_outputs",
                "submit_tool_outputs": {
                    "tool_calls": [
                        {"id": "call_a", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"loc\":\"NYC\"}"}}
                    ]
                }
            }
        });
        let contents = required_action_contents(&run, "run_42");
        assert_eq!(contents.len(), 1);
        let Content::FunctionCall(fc) = &contents[0] else {
            panic!("expected function call");
        };
        assert_eq!(fc.name, "get_weather");
        assert_eq!(
            decode_call_id(&fc.call_id),
            Some(("run_42".into(), "call_a".into()))
        );
    }

    #[test]
    fn usage_and_message_text_parsing() {
        let run =
            json!({"usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}});
        let usage = parse_usage(&run).unwrap();
        assert_eq!(usage.total_token_count, Some(14));

        let list = json!({"data": [
            {"role": "user", "content": [{"type": "text", "text": {"value": "hi"}}]},
            {"role": "assistant", "content": [{"type": "text", "text": {"value": "hello there"}}]},
        ]});
        assert_eq!(assistant_text_from_messages(&list), "hello there");
    }

    #[test]
    fn submit_body_shapes_outputs_and_approvals() {
        let body = build_submit_body(
            &[("call_1".into(), "42".into())],
            &[("call_2".into(), true)],
            true,
        );
        assert_eq!(
            body["tool_outputs"][0],
            json!({"tool_call_id": "call_1", "output": "42"})
        );
        assert_eq!(
            body["tool_approvals"][0],
            json!({"tool_call_id": "call_2", "approve": true})
        );
        assert_eq!(body["stream"], json!(true));
    }

    // -- last_error / classify_last_error -----------------------------------

    #[test]
    fn last_error_code_reads_the_code_field() {
        let run = json!({"status": "failed", "last_error": {"code": "content_filter", "message": "flagged"}});
        assert_eq!(last_error_code(&run), Some("content_filter"));

        let no_code = json!({"status": "failed", "last_error": {"message": "oops"}});
        assert_eq!(last_error_code(&no_code), None);

        let no_last_error = json!({"status": "failed"});
        assert_eq!(last_error_code(&no_last_error), None);
    }

    #[test]
    fn classify_last_error_maps_content_filter_code_to_content_filter_error() {
        let run = json!({
            "status": "failed",
            "last_error": {"code": "content_filter", "message": "The response was filtered"},
        });
        let err = classify_last_error(&run);
        assert!(matches!(err, Error::ServiceContentFilter { .. }), "{err:?}");
        assert!(err.to_string().contains("The response was filtered"));
    }

    #[test]
    fn classify_last_error_leaves_other_codes_as_generic_service_error() {
        // Matches upstream exactly: any non-content-filter `last_error`
        // (or one with no `code` at all) is just the plain exception Python
        // raises, not a more specific variant.
        for run in [
            json!({"status": "failed", "last_error": {"code": "rate_limit_exceeded", "message": "too many"}}),
            json!({"status": "failed", "last_error": {"code": "server_error", "message": "oops"}}),
            json!({"status": "failed", "last_error": {"message": "no code at all"}}),
            json!({"status": "failed"}),
        ] {
            let err = classify_last_error(&run);
            assert!(matches!(err, Error::Service(_)), "run {run:?}: got {err:?}");
            assert!(!matches!(err, Error::ServiceContentFilter { .. }));
        }
    }
}
