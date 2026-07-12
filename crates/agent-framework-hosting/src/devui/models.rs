//! Serde models for the DevUI-style API.
//!
//! Field names mirror the Python DevUI server's models
//! (`agent_framework_devui/models`) so responses are interchangeable with the
//! reference implementation. Divergences are documented on the relevant fields.
//!
//! The OpenAI-Responses wire types (`ResponsesRequest`, `ResponseObject`,
//! `Usage`, …) and the `openai_error` helper used to live here; they are now
//! the reusable [`crate::responses`] module (UPSTREAM_DRIFT.md §14) and are
//! re-exported below so existing `devui::models::…` paths keep resolving.
//! Only the DevUI-specific entity/discovery/health types remain defined here.

use serde::Serialize;
use serde_json::{Map, Value};

pub use crate::responses::{
    openai_error, InputTokensDetails, OutputMessage, OutputText, OutputTokensDetails,
    ResponseObject, ResponsesRequest, Usage,
};

/// `GET /health` payload — mirrors DevUI's `health_check`.
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub entities_count: usize,
    pub framework: &'static str,
}

/// Entity descriptor — mirrors DevUI's `EntityInfo`.
///
/// Divergences from DevUI: `null`-valued optional fields are omitted rather than
/// serialized as `null`; `tools`/`model` are populated only when cheaply
/// available from the concrete agent type (the core `SupportsAgentRun` trait exposes
/// neither, so both are usually absent — see crate docs).
#[derive(Debug, Clone, Serialize)]
pub struct EntityInfo {
    pub id: String,
    #[serde(rename = "type")]
    pub entity_type: &'static str,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub framework: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    pub metadata: Map<String, Value>,
    pub source: &'static str,
    // SupportsAgentRun-specific.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    // Workflow-specific.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub executors: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_executor_id: Option<String>,
}

/// `GET /v1/entities` payload — mirrors DevUI's `DiscoveryResponse`.
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryResponse {
    pub entities: Vec<EntityInfo>,
}
