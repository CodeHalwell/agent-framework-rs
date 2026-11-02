//! Microsoft Foundry evaluations: an [`Evaluator`] that scores items with
//! Foundry's built-in (`builtin.*`) and generated rubric evaluators.
//!
//! Port of upstream `agent_framework_foundry._foundry_evals`
//! (`FoundryEvals`, `GeneratedEvaluatorRef`, `evaluate_traces`), behind this
//! crate's `evals` feature. It speaks the OpenAI-style Evals REST API on the
//! project (`{project}/openai/v1/evals`): create an eval with testing
//! criteria, start a run over the items as inline JSONL, poll until it
//! finishes, then page through the per-item output.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_core::evaluation::{evaluate_agent, Evaluator};
//! use agent_framework_foundry::evals::FoundryEvals;
//!
//! # async fn demo(agent: Arc<dyn agent_framework_core::agent::SupportsAgentRun>) -> agent_framework_core::error::Result<()> {
//! let evals = FoundryEvals::new("https://my-project.services.ai.azure.com/api/projects/p", "api-key", "gpt-4o")
//!     .with_evaluators([FoundryEvals::RELEVANCE, FoundryEvals::TOOL_CALL_ACCURACY]);
//! let results = evaluate_agent()
//!     .agent(agent)
//!     .queries(["What's the weather?"])
//!     .evaluator(Arc::new(evals) as Arc<dyn Evaluator>)
//!     .run()
//!     .await?;
//! results[0].raise_for_status(None)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Divergences
//!
//! - `evaluate_foundry_target` (running a Foundry-hosted target inside the
//!   eval) is not ported.
//! - Errors in the output-items listing are logged and yield no items, as
//!   upstream; errors creating or polling the run propagate.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use agent_framework_azure::TokenCredential;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::evaluation::{
    ConversationSplit, ConversationSplitter, EvalItem, EvalItemResult, EvalResults,
    EvalScoreResult, Evaluator, RubricScore,
};
use agent_framework_core::types::{Content, FunctionArguments, Message, Role};
use async_trait::async_trait;
use serde_json::{json, Value};

/// A rubric evaluator that already exists in the Foundry project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedEvaluatorRef {
    /// The evaluator's name in the project.
    pub name: String,
    /// The pinned version; `None` resolves to the latest at run time.
    pub version: Option<String>,
    /// The name shown in results (default: `name`).
    pub display_name: Option<String>,
}

impl GeneratedEvaluatorRef {
    /// A reference pinned to `version` (recommended, for reproducible runs).
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: Some(version.into()),
            display_name: None,
        }
    }
    /// A versionless reference (resolves to the latest version at run time).
    pub fn latest(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: None,
            display_name: None,
        }
    }
    /// Name the evaluator in results.
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }
}

/// An evaluator to run: a built-in name (`relevance` or `builtin.relevance`)
/// or a generated rubric evaluator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluatorSpec {
    /// A built-in evaluator.
    Builtin(String),
    /// A rubric evaluator in the project.
    Generated(GeneratedEvaluatorRef),
}

impl From<&str> for EvaluatorSpec {
    fn from(s: &str) -> Self {
        Self::Builtin(s.to_string())
    }
}
impl From<String> for EvaluatorSpec {
    fn from(s: String) -> Self {
        Self::Builtin(s)
    }
}
impl From<GeneratedEvaluatorRef> for EvaluatorSpec {
    fn from(r: GeneratedEvaluatorRef) -> Self {
        Self::Generated(r)
    }
}

const BUILTIN_EVALUATORS: &[&str] = &[
    "intent_resolution",
    "task_adherence",
    "task_completion",
    "task_navigation_efficiency",
    "tool_call_accuracy",
    "tool_selection",
    "tool_input_accuracy",
    "tool_output_utilization",
    "tool_call_success",
    "coherence",
    "fluency",
    "relevance",
    "groundedness",
    "response_completeness",
    "similarity",
    "violence",
    "sexual",
    "self_harm",
    "hate_unfairness",
];
const AGENT_EVALUATORS: &[&str] = &[
    "intent_resolution",
    "task_adherence",
    "task_completion",
    "task_navigation_efficiency",
    "tool_call_accuracy",
    "tool_selection",
    "tool_input_accuracy",
    "tool_output_utilization",
    "tool_call_success",
];
const TOOL_EVALUATORS: &[&str] = &[
    "tool_call_accuracy",
    "tool_selection",
    "tool_input_accuracy",
    "tool_output_utilization",
    "tool_call_success",
];
const TOOL_DEFINITION_EVALUATORS: &[&str] = &[
    "tool_call_accuracy",
    "tool_selection",
    "tool_input_accuracy",
    "tool_output_utilization",
    "tool_call_success",
    "intent_resolution",
    "task_adherence",
    "task_completion",
    "task_navigation_efficiency",
];
const DEFAULT_EVALUATORS: &[&str] = &["relevance", "coherence", "task_adherence"];

/// `relevance` → `builtin.relevance`. Unknown short names are refused; an
/// unknown `builtin.*` name passes with a warning.
pub fn resolve_evaluator(name: &str) -> Result<String> {
    if let Some(short) = name.strip_prefix("builtin.") {
        if !BUILTIN_EVALUATORS.contains(&short) {
            tracing::warn!(evaluator = %name, "evaluator is not in the known built-in list");
        }
        return Ok(name.to_string());
    }
    if BUILTIN_EVALUATORS.contains(&name) {
        Ok(format!("builtin.{name}"))
    } else {
        Err(Error::Configuration(format!(
            "Unknown evaluator '{name}'. Available: {BUILTIN_EVALUATORS:?}"
        )))
    }
}

fn short_of(qualified: &str) -> &str {
    qualified.strip_prefix("builtin.").unwrap_or(qualified)
}

/// One message in the Foundry Evals wire format (a function-result message
/// becomes one `tool` message per result).
pub fn convert_message(message: &Message) -> Vec<Value> {
    let mut items = Vec::new();
    let mut results = Vec::new();
    for content in &message.contents {
        match content {
            Content::Text(t) if !t.text.is_empty() => {
                items.push(json!({"type": "text", "text": t.text}));
            }
            Content::Uri(u) => {
                let mut image = json!({"type": "input_image", "image_url": u.uri});
                if !u.media_type.is_empty() {
                    image["detail"] = json!("auto");
                }
                items.push(image);
            }
            Content::FunctionCall(call) => {
                let arguments = match &call.arguments {
                    None => json!({}),
                    Some(FunctionArguments::Object(map)) => json!(map),
                    Some(FunctionArguments::Raw(raw)) => serde_json::from_str(raw)
                        .unwrap_or_else(|_| json!({"_raw_arguments": "[unparseable]"})),
                };
                items.push(json!({
                    "type": "tool_call",
                    "tool_call_id": call.call_id,
                    "name": call.name,
                    "arguments": arguments,
                }));
            }
            Content::FunctionResult(result) => {
                let value = match &result.result {
                    Some(Value::String(s)) => {
                        serde_json::from_str(s).unwrap_or(Value::String(s.clone()))
                    }
                    Some(other) => other.clone(),
                    None => Value::Null,
                };
                results.push((result.call_id.clone(), value));
            }
            _ => {}
        }
    }
    if !results.is_empty() {
        return results
            .into_iter()
            .map(|(id, value)| {
                json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": [{"type": "tool_result", "tool_result": value}],
                })
            })
            .collect();
    }
    if items.is_empty() {
        items.push(json!({"type": "text", "text": ""}));
    }
    vec![json!({"role": message.role.as_str(), "content": items})]
}

fn convert_messages(messages: &[Message]) -> Vec<Value> {
    messages.iter().flat_map(convert_message).collect()
}

/// The `testing_criteria` for an eval.
pub fn build_testing_criteria(
    evaluators: &[EvaluatorSpec],
    model: &str,
    include_data_mapping: bool,
    include_tool_definitions: bool,
) -> Result<Vec<Value>> {
    let mut criteria = Vec::new();
    for spec in evaluators {
        match spec {
            EvaluatorSpec::Generated(r) => {
                let mut entry = json!({
                    "type": "azure_ai_evaluator",
                    "name": r.display_name.as_deref().unwrap_or(&r.name),
                    "evaluator_name": r.name,
                    "initialization_parameters": {"deployment_name": model},
                });
                match &r.version {
                    Some(v) => entry["evaluator_version"] = json!(v),
                    None => tracing::warn!(
                        evaluator = %r.name,
                        "generated evaluator has no pinned version; the run resolves to the current one"
                    ),
                }
                if include_data_mapping {
                    let mut mapping = json!({
                        "query": "{{item.query_messages}}",
                        "response": "{{item.response_messages}}",
                    });
                    if include_tool_definitions {
                        mapping["tool_definitions"] = json!("{{item.tool_definitions}}");
                    }
                    entry["data_mapping"] = mapping;
                }
                criteria.push(entry);
            }
            EvaluatorSpec::Builtin(name) => {
                let qualified = resolve_evaluator(name)?;
                let short = short_of(&qualified);
                let mut entry = json!({
                    "type": "azure_ai_evaluator",
                    "name": short_of(name),
                    "evaluator_name": qualified,
                    "initialization_parameters": {"deployment_name": model},
                });
                if include_data_mapping {
                    let mut mapping = if AGENT_EVALUATORS.contains(&short) {
                        json!({"query": "{{item.query_messages}}", "response": "{{item.response_messages}}"})
                    } else {
                        json!({"query": "{{item.query}}", "response": "{{item.response}}"})
                    };
                    if short == "groundedness" {
                        mapping["context"] = json!("{{item.context}}");
                    }
                    if short == "similarity" {
                        mapping["ground_truth"] = json!("{{item.ground_truth}}");
                    }
                    if include_tool_definitions && TOOL_DEFINITION_EVALUATORS.contains(&short) {
                        mapping["tool_definitions"] = json!("{{item.tool_definitions}}");
                    }
                    entry["data_mapping"] = mapping;
                }
                criteria.push(entry);
            }
        }
    }
    Ok(criteria)
}

/// The `item_schema` for the inline JSONL items.
pub fn build_item_schema(has_context: bool, has_tools: bool, has_ground_truth: bool) -> Value {
    let mut properties = json!({
        "query": {"type": "string"},
        "response": {"type": "string"},
        "query_messages": {"type": "array"},
        "response_messages": {"type": "array"},
    });
    if has_context {
        properties["context"] = json!({"type": "string"});
    }
    if has_ground_truth {
        properties["ground_truth"] = json!({"type": "string"});
    }
    if has_tools {
        properties["tool_definitions"] = json!({"type": "array"});
    }
    json!({"type": "object", "properties": properties, "required": ["query", "response"]})
}

/// The evaluators to run: the configured ones, or the defaults (plus
/// `tool_call_accuracy` when items have tools); then tool-only evaluators
/// are dropped when no item has tools.
fn resolve_for_items(
    configured: Option<&[EvaluatorSpec]>,
    items: &[EvalItem],
) -> Result<Vec<EvaluatorSpec>> {
    let has_tools = items
        .iter()
        .any(|i| i.tools.as_ref().is_some_and(|t| !t.is_empty()));
    let mut specs: Vec<EvaluatorSpec> = match configured {
        Some(c) => c.to_vec(),
        None => {
            let mut d: Vec<EvaluatorSpec> =
                DEFAULT_EVALUATORS.iter().map(|s| (*s).into()).collect();
            if has_tools {
                d.push("tool_call_accuracy".into());
            }
            d
        }
    };
    if !has_tools {
        let before = specs.len();
        let mut kept = Vec::new();
        for spec in specs {
            let tool_only = match &spec {
                EvaluatorSpec::Generated(_) => false,
                EvaluatorSpec::Builtin(n) => {
                    TOOL_EVALUATORS.contains(&short_of(&resolve_evaluator(n)?))
                }
            };
            if !tool_only {
                kept.push(spec);
            }
        }
        if kept.is_empty() && before > 0 {
            return Err(Error::Configuration(
                "All requested evaluators require tool definitions, but no items have tools. Either add tool \
                 definitions to your items or choose evaluators that do not require tools."
                    .into(),
            ));
        }
        specs = kept;
    }
    Ok(specs)
}

/// Parse a rubric dimension breakdown from an evaluator's raw sample.
pub fn extract_rubric_scores(sample: &Value) -> Option<Vec<RubricScore>> {
    let mut containers = Vec::new();
    if let Some(props) = sample.get("properties") {
        containers.push(props);
    }
    containers.push(sample);
    for container in containers {
        for key in ["dimension_scores", "rubric_scores"] {
            let Some(Value::Array(raw)) = container.get(key) else {
                continue;
            };
            let parsed: Vec<RubricScore> = raw
                .iter()
                .filter_map(|e| {
                    let id = e.get("id")?;
                    let weight = e.get("weight")?.as_f64()?;
                    let applicable = e.get("applicable")?.as_bool()?;
                    Some(RubricScore {
                        id: match id {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        },
                        score: e.get("score").and_then(Value::as_f64).map(|s| s as i64),
                        applicable,
                        weight: weight as i64,
                        reason: e
                            .get("reason")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    })
                })
                .collect();
            if !parsed.is_empty() {
                return Some(parsed);
            }
        }
    }
    None
}

enum Auth {
    ApiKey(String),
    Credential(Arc<dyn TokenCredential>),
}

/// An [`Evaluator`] backed by Microsoft Foundry. See the [module docs](self).
#[derive(Clone)]
pub struct FoundryEvals {
    http: reqwest::Client,
    base: String,
    auth: Arc<Auth>,
    model: String,
    evaluators: Option<Vec<EvaluatorSpec>>,
    conversation_split: ConversationSplitter,
    poll_interval: Duration,
    timeout: Duration,
}

impl std::fmt::Debug for FoundryEvals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryEvals")
            .field("base", &self.base)
            .field("model", &self.model)
            .field("evaluators", &self.evaluators)
            .finish_non_exhaustive()
    }
}

macro_rules! evaluator_names {
    ($($konst:ident = $name:literal),* $(,)?) => {
        impl FoundryEvals {
            $(
                #[doc = concat!("The built-in `", $name, "` evaluator.")]
                pub const $konst: &'static str = $name;
            )*
        }
    };
}

evaluator_names!(
    INTENT_RESOLUTION = "intent_resolution",
    TASK_ADHERENCE = "task_adherence",
    TASK_COMPLETION = "task_completion",
    TASK_NAVIGATION_EFFICIENCY = "task_navigation_efficiency",
    TOOL_CALL_ACCURACY = "tool_call_accuracy",
    TOOL_SELECTION = "tool_selection",
    TOOL_INPUT_ACCURACY = "tool_input_accuracy",
    TOOL_OUTPUT_UTILIZATION = "tool_output_utilization",
    TOOL_CALL_SUCCESS = "tool_call_success",
    COHERENCE = "coherence",
    FLUENCY = "fluency",
    RELEVANCE = "relevance",
    GROUNDEDNESS = "groundedness",
    RESPONSE_COMPLETENESS = "response_completeness",
    SIMILARITY = "similarity",
    VIOLENCE = "violence",
    SEXUAL = "sexual",
    SELF_HARM = "self_harm",
    HATE_UNFAIRNESS = "hate_unfairness",
);

impl FoundryEvals {
    fn build(project_endpoint: &str, auth: Auth, model: impl Into<String>) -> Self {
        Self {
            http: crate::http_client(),
            base: format!("{}/openai/v1", project_endpoint.trim_end_matches('/')),
            auth: Arc::new(auth),
            model: model.into(),
            evaluators: None,
            conversation_split: ConversationSplit::LastTurn.splitter(),
            poll_interval: Duration::from_secs(5),
            timeout: Duration::from_secs(180),
        }
    }

    /// Evaluate in the project at `project_endpoint` with an API key, judging
    /// with model deployment `model`.
    pub fn new(
        project_endpoint: &str,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::build(project_endpoint, Auth::ApiKey(api_key.into()), model)
    }

    /// As [`new`](Self::new), with Entra ID (scoped to
    /// [`FOUNDRY_SCOPE`](crate::FOUNDRY_SCOPE)).
    pub fn with_token_credential(
        project_endpoint: &str,
        credential: Arc<dyn TokenCredential>,
        model: impl Into<String>,
    ) -> Self {
        Self::build(project_endpoint, Auth::Credential(credential), model)
    }

    /// Override the `/openai/v1` base URL.
    pub fn with_base_url(mut self, base: impl Into<String>) -> Self {
        self.base = base.into().trim_end_matches('/').to_string();
        self
    }

    /// The evaluators to run (default: relevance, coherence and
    /// task_adherence, plus tool_call_accuracy when items have tools).
    pub fn with_evaluators<S: Into<EvaluatorSpec>>(
        mut self,
        evaluators: impl IntoIterator<Item = S>,
    ) -> Self {
        self.evaluators = Some(evaluators.into_iter().map(Into::into).collect());
        self
    }

    /// The split used for items without their own (default last turn).
    pub fn with_conversation_split(mut self, split: ConversationSplitter) -> Self {
        self.conversation_split = split;
        self
    }

    /// How often to poll the run (default 5 s).
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// How long to wait for the run (default 180 s); after that the results
    /// report status `timeout`.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    async fn call(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value> {
        let mut request = self.http.request(method, format!("{}{path}", self.base));
        request = match &*self.auth {
            Auth::ApiKey(key) => request.header("api-key", key),
            Auth::Credential(c) => {
                request.bearer_auth(c.get_token_for_scope(crate::FOUNDRY_SCOPE).await?)
            }
        };
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request.send().await.map_err(|e| {
            Error::Service(format!(
                "Foundry evals request failed: {}",
                agent_framework_core::error::error_chain(&e)
            ))
        })?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Error::Service(format!(
                "Foundry evals returned {status}: {text}"
            )));
        }
        serde_json::from_str(&text)
            .map_err(|e| Error::Service(format!("Foundry evals returned invalid JSON: {e}")))
    }

    /// The JSONL item for one [`EvalItem`].
    pub fn item_payload(&self, item: &EvalItem) -> Value {
        let (query, response) = item.split_messages(Some(
            item.split_strategy
                .as_ref()
                .unwrap_or(&self.conversation_split),
        ));
        let text = |msgs: &[Message], role: &str| {
            msgs.iter()
                .filter(|m| m.role.as_str() == role)
                .map(Message::text)
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string()
        };
        let mut d = json!({
            "query": text(&query, Role::USER),
            "response": text(&response, Role::ASSISTANT),
            "query_messages": convert_messages(&query),
            "response_messages": convert_messages(&response),
        });
        if let Some(tools) = item.tools.as_ref().filter(|t| !t.is_empty()) {
            d["tool_definitions"] = Value::Array(
                tools
                    .iter()
                    .map(|t| json!({"name": t.name, "description": t.description, "parameters": t.parameters}))
                    .collect(),
            );
        }
        if let Some(context) = &item.context {
            d["context"] = json!(context);
        }
        if let Some(expected) = &item.expected_output {
            d["ground_truth"] = json!(expected);
        }
        d
    }

    async fn run_and_poll(
        &self,
        eval: Value,
        data_source: Value,
        eval_name: &str,
    ) -> Result<EvalResults> {
        let created = self
            .call(reqwest::Method::POST, "/evals", Some(eval))
            .await?;
        let eval_id = id_of(&created)?;
        let run = self
            .call(
                reqwest::Method::POST,
                &format!("/evals/{eval_id}/runs"),
                Some(json!({"name": format!("{eval_name} Run"), "data_source": data_source})),
            )
            .await?;
        let run_id = id_of(&run)?;
        poll_run(self, &eval_id, &run_id, "Microsoft Foundry").await
    }

    /// Evaluate stored Responses API responses by id, with the built-in
    /// `evaluators` (upstream's `evaluate_traces(response_ids=...)`).
    pub async fn evaluate_responses(
        &self,
        response_ids: &[String],
        evaluators: &[EvaluatorSpec],
        eval_name: &str,
    ) -> Result<EvalResults> {
        if response_ids.is_empty() {
            return Err(Error::Configuration(
                "Provide at least one response id.".into(),
            ));
        }
        let eval = json!({
            "name": eval_name,
            "data_source_config": {"type": "azure_ai_source", "scenario": "responses"},
            "testing_criteria": build_testing_criteria(evaluators, &self.model, false, false)?,
        });
        let data_source = json!({
            "type": "azure_ai_responses",
            "item_generation_params": {
                "type": "response_retrieval",
                "data_mapping": {"response_id": "{{item.resp_id}}"},
                "source": {
                    "type": "file_content",
                    "content": response_ids.iter().map(|id| json!({"item": {"resp_id": id}})).collect::<Vec<_>>(),
                },
            },
        });
        self.run_and_poll(eval, data_source, eval_name).await
    }
}

fn id_of(value: &Value) -> Result<String> {
    value
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Error::Service(format!("Foundry evals response has no id: {value}")))
}

fn counts(run: &Value) -> Option<BTreeMap<String, i64>> {
    let c = run.get("result_counts").filter(|c| c.is_object())?;
    Some(
        ["errored", "failed", "passed", "total"]
            .iter()
            .map(|k| {
                (
                    k.to_string(),
                    c.get(*k).and_then(Value::as_i64).unwrap_or(0),
                )
            })
            .collect(),
    )
}

fn per_evaluator(run: &Value) -> BTreeMap<String, BTreeMap<String, i64>> {
    let mut out = BTreeMap::new();
    for item in run
        .get("per_testing_criteria_results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(name) = item.get("testing_criteria").and_then(Value::as_str) {
            out.insert(
                name.to_string(),
                [("passed", "passed"), ("failed", "failed")]
                    .iter()
                    .map(|(k, f)| {
                        (
                            k.to_string(),
                            item.get(*f).and_then(Value::as_i64).unwrap_or(0),
                        )
                    })
                    .collect(),
            );
        }
    }
    out
}

async fn poll_run(
    evals: &FoundryEvals,
    eval_id: &str,
    run_id: &str,
    provider: &str,
) -> Result<EvalResults> {
    let deadline = tokio::time::Instant::now() + evals.timeout;
    loop {
        let run = evals
            .call(
                reqwest::Method::GET,
                &format!("/evals/{eval_id}/runs/{run_id}"),
                None,
            )
            .await?;
        let status = run
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if matches!(status.as_str(), "completed" | "failed" | "canceled") {
            let error = (status == "failed").then(|| match run.get("error") {
                Some(Value::String(s)) => s.clone(),
                Some(e) => e
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| e.to_string()),
                None => "unknown error".into(),
            });
            let items = if status == "completed" {
                fetch_output_items(evals, eval_id, run_id).await
            } else {
                Vec::new()
            };
            return Ok(EvalResults {
                provider: provider.into(),
                eval_id: eval_id.into(),
                run_id: run_id.into(),
                status,
                result_counts: counts(&run),
                report_url: run
                    .get("report_url")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                error,
                per_evaluator: per_evaluator(&run),
                items,
                ..Default::default()
            });
        }
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Ok(EvalResults {
                provider: provider.into(),
                eval_id: eval_id.into(),
                run_id: run_id.into(),
                status: "timeout".into(),
                ..Default::default()
            });
        }
        tokio::time::sleep(evals.poll_interval.min(deadline - now)).await;
    }
}

fn joined(parts: Option<&Vec<Value>>, role: &str) -> Option<String> {
    let texts: Vec<String> = parts?
        .iter()
        .filter(|p| p.get("role").and_then(Value::as_str) == Some(role))
        .filter_map(|p| p.get("content").and_then(Value::as_str).map(str::to_string))
        .collect();
    (!texts.is_empty()).then(|| texts.join(" "))
}

fn parse_output_item(oi: &Value) -> EvalItemResult {
    let scores = oi
        .get("results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|r| {
            let sample = r.get("sample").cloned().filter(|s| !s.is_null());
            EvalScoreResult {
                name: r
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                score: r.get("score").and_then(Value::as_f64).unwrap_or(0.0),
                passed: r.get("passed").and_then(Value::as_bool),
                dimensions: sample.as_ref().and_then(extract_rubric_scores),
                sample,
            }
        })
        .collect();
    let sample = oi.get("sample").filter(|s| s.is_object());
    let error = sample
        .and_then(|s| s.get("error"))
        .filter(|e| e.is_object());
    let non_empty = |v: Option<&Value>| {
        v.and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let usage = sample
        .and_then(|s| s.get("usage"))
        .filter(|u| u.get("total_tokens").and_then(Value::as_i64).unwrap_or(0) > 0);
    let ds = oi.get("datasource_item");
    EvalItemResult {
        item_id: oi
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        status: oi
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        scores,
        error_code: non_empty(error.and_then(|e| e.get("code"))),
        error_message: non_empty(error.and_then(|e| e.get("message"))),
        response_id: ds
            .and_then(|d| d.get("resp_id").or_else(|| d.get("response_id")))
            .and_then(|v| match v {
                Value::String(s) if !s.is_empty() => Some(s.clone()),
                Value::Null | Value::String(_) => None,
                other => Some(other.to_string()),
            }),
        input_text: joined(
            sample
                .and_then(|s| s.get("input"))
                .and_then(Value::as_array),
            "user",
        ),
        output_text: joined(
            sample
                .and_then(|s| s.get("output"))
                .and_then(Value::as_array),
            "assistant",
        ),
        token_usage: usage.map(|u| {
            [
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cached_tokens",
            ]
            .iter()
            .map(|k| {
                (
                    k.to_string(),
                    u.get(*k).and_then(Value::as_i64).unwrap_or(0),
                )
            })
            .collect()
        }),
        metadata: None,
    }
}

async fn fetch_output_items(
    evals: &FoundryEvals,
    eval_id: &str,
    run_id: &str,
) -> Vec<EvalItemResult> {
    let mut items = Vec::new();
    let mut after: Option<String> = None;
    loop {
        let path = match &after {
            Some(a) => format!("/evals/{eval_id}/runs/{run_id}/output_items?after={a}"),
            None => format!("/evals/{eval_id}/runs/{run_id}/output_items"),
        };
        let page = match evals.call(reqwest::Method::GET, &path, None).await {
            Ok(page) => page,
            Err(e) => {
                tracing::warn!(run = %run_id, error = %e, "could not fetch output_items");
                return items;
            }
        };
        let data = page
            .get("data")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        items.extend(data.iter().map(parse_output_item));
        let more = page
            .get("has_more")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let last = page
            .get("last_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                data.last()
                    .and_then(|d| d.get("id"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            });
        match (more, last) {
            (true, Some(l)) if after.as_deref() != Some(&l) => after = Some(l),
            _ => return items,
        }
    }
}

#[async_trait]
impl Evaluator for FoundryEvals {
    fn name(&self) -> &str {
        "Microsoft Foundry"
    }

    async fn evaluate(&self, items: &[EvalItem], eval_name: &str) -> Result<EvalResults> {
        let specs = resolve_for_items(self.evaluators.as_deref(), items)?;
        let dicts: Vec<Value> = items.iter().map(|i| self.item_payload(i)).collect();
        let has = |k: &str| dicts.iter().any(|d| d.get(k).is_some());
        let (has_context, has_truth, has_tools) =
            (has("context"), has("ground_truth"), has("tool_definitions"));
        let eval = json!({
            "name": eval_name,
            "data_source_config": {
                "type": "custom",
                "item_schema": build_item_schema(has_context, has_tools, has_truth),
                "include_sample_schema": true,
            },
            "testing_criteria": build_testing_criteria(&specs, &self.model, true, has_tools)?,
        });
        let data_source = json!({
            "type": "jsonl",
            "source": {
                "type": "file_content",
                "content": dicts.into_iter().map(|d| json!({"item": d})).collect::<Vec<_>>(),
            },
        });
        self.run_and_poll(eval, data_source, eval_name).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::FunctionCallContent;

    #[test]
    fn evaluator_names_resolve() {
        assert_eq!(resolve_evaluator("relevance").unwrap(), "builtin.relevance");
        assert_eq!(
            resolve_evaluator("builtin.new_thing").unwrap(),
            "builtin.new_thing"
        );
        assert!(resolve_evaluator("relevanse").is_err());
    }

    #[test]
    fn criteria_map_agent_and_quality_evaluators_differently() {
        let specs: Vec<EvaluatorSpec> = vec![
            "relevance".into(),
            "task_adherence".into(),
            "similarity".into(),
            GeneratedEvaluatorRef::new("policy", "2").into(),
        ];
        let c = build_testing_criteria(&specs, "gpt-4o", true, true).unwrap();
        assert_eq!(c[0]["data_mapping"]["query"], "{{item.query}}");
        assert_eq!(c[1]["data_mapping"]["query"], "{{item.query_messages}}");
        assert_eq!(
            c[1]["data_mapping"]["tool_definitions"],
            "{{item.tool_definitions}}"
        );
        assert_eq!(
            c[2]["data_mapping"]["ground_truth"],
            "{{item.ground_truth}}"
        );
        assert_eq!(c[3]["evaluator_version"], "2");
        assert_eq!(c[3]["evaluator_name"], "policy");
    }

    #[test]
    fn tool_evaluators_need_tools() {
        let none = resolve_for_items(None, &[EvalItem::new(vec![])]).unwrap();
        assert_eq!(none.len(), 3);
        let only_tools: Vec<EvaluatorSpec> = vec!["tool_call_accuracy".into()];
        assert!(resolve_for_items(Some(&only_tools), &[EvalItem::new(vec![])]).is_err());
    }

    #[test]
    fn messages_convert_to_the_wire_format() {
        let call = Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "c1",
                "get",
                Some(FunctionArguments::Raw("not json".into())),
            ))],
        );
        let wire = convert_message(&call);
        assert_eq!(
            wire[0]["content"][0]["arguments"],
            json!({"_raw_arguments": "[unparseable]"})
        );
        assert_eq!(
            convert_message(&Message::user(""))[0]["content"][0]["text"],
            ""
        );
    }

    #[test]
    fn rubric_scores_parse_from_either_key() {
        let sample = json!({"properties": {"dimension_scores": [
            {"id": "tone", "score": 4, "applicable": true, "weight": 2, "reason": "ok"},
            {"id": "bad"}
        ]}});
        let scores = extract_rubric_scores(&sample).unwrap();
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].score, Some(4));
        assert!(extract_rubric_scores(&json!({"rubric_scores": []})).is_none());
    }
}
