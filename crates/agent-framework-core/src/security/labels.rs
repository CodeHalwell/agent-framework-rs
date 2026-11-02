//! Security labels, their lattice, and the labeled value types.

use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::types::{Message, Role};

/// The label-metadata key holding a `USER_IDENTITY` label's principal set:
/// a non-empty list of `{"tenant_id", "user_id"}` objects.
pub const PRINCIPAL_METADATA_KEY: &str = "agent_framework.security.principals";

/// The integrity of content: where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntegrityLabel {
    /// From a trusted source (the user, the system, a trusted tool).
    #[default]
    Trusted,
    /// From an untrusted source (external data, a model influenced by it).
    Untrusted,
}

impl IntegrityLabel {
    /// The wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Trusted => "trusted",
            Self::Untrusted => "untrusted",
        }
    }

    /// Parse a wire value.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "trusted" => Some(Self::Trusted),
            "untrusted" => Some(Self::Untrusted),
            _ => None,
        }
    }
}

impl std::fmt::Display for IntegrityLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The confidentiality of content: who may see it. Ordered
/// `Public < Private < UserIdentity`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ConfidentialityLabel {
    /// May be shared publicly.
    #[default]
    Public,
    /// Private; must not reach public destinations.
    Private,
    /// Restricted to specific principals (see [`PRINCIPAL_METADATA_KEY`]).
    UserIdentity,
}

impl ConfidentialityLabel {
    /// The wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Private => "private",
            Self::UserIdentity => "user_identity",
        }
    }

    /// Parse a wire value.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "public" => Some(Self::Public),
            "private" => Some(Self::Private),
            "user_identity" => Some(Self::UserIdentity),
            _ => None,
        }
    }
}

impl std::fmt::Display for ConfidentialityLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A tenant/user identity a `USER_IDENTITY` label is restricted to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Principal {
    /// The tenant.
    pub tenant_id: String,
    /// The user.
    pub user_id: String,
}

impl Principal {
    /// A principal.
    pub fn new(tenant_id: impl Into<String>, user_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            user_id: user_id.into(),
        }
    }
}

/// Validate and canonicalize (dedupe, sort) a principal-set declaration.
pub(crate) fn canonical_principals(
    value: Option<&Value>,
    source: &str,
) -> Result<BTreeSet<Principal>, String> {
    let items = value
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| format!("{source} principals must be a non-empty sequence"))?;
    let mut out = BTreeSet::new();
    for item in items {
        let map = item
            .as_object()
            .ok_or_else(|| format!("{source} principals must contain mappings"))?;
        if map.len() != 2 || !map.contains_key("tenant_id") || !map.contains_key("user_id") {
            return Err(format!(
                "{source} principal fields must be tenant_id and user_id"
            ));
        }
        let field = |k: &str| {
            map.get(k)
                .and_then(Value::as_str)
                .filter(|s| !s.trim().is_empty())
                .map(str::to_string)
        };
        match (field("tenant_id"), field("user_id")) {
            (Some(tenant_id), Some(user_id)) => {
                out.insert(Principal { tenant_id, user_id });
            }
            _ => {
                return Err(format!(
                    "{source} principal identifiers must be non-empty strings"
                ))
            }
        }
    }
    Ok(out)
}

/// The serialized canonical form of a principal set.
pub(crate) fn principal_list<'a>(principals: impl IntoIterator<Item = &'a Principal>) -> Value {
    Value::Array(
        principals
            .into_iter()
            .map(|p| json!({"tenant_id": p.tenant_id, "user_id": p.user_id}))
            .collect(),
    )
}

/// The security label of a piece of content.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ContentLabel {
    /// Integrity.
    pub integrity: IntegrityLabel,
    /// Confidentiality.
    pub confidentiality: ConfidentialityLabel,
    /// Extra metadata (source information, a `USER_IDENTITY` principal set).
    pub metadata: Map<String, Value>,
}

impl ContentLabel {
    /// A label.
    pub fn new(integrity: IntegrityLabel, confidentiality: ConfidentialityLabel) -> Self {
        Self {
            integrity,
            confidentiality,
            metadata: Map::new(),
        }
    }

    /// `TRUSTED` + `PUBLIC` (the default).
    pub fn trusted() -> Self {
        Self::default()
    }

    /// `UNTRUSTED` + `PUBLIC`.
    pub fn untrusted() -> Self {
        Self::new(IntegrityLabel::Untrusted, ConfidentialityLabel::Public)
    }

    /// Builder: the confidentiality.
    pub fn with_confidentiality(mut self, confidentiality: ConfidentialityLabel) -> Self {
        self.confidentiality = confidentiality;
        self
    }

    /// Builder: one metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// A `USER_IDENTITY` label restricted to `principals`.
    pub fn user_identity(
        integrity: IntegrityLabel,
        principals: impl IntoIterator<Item = Principal>,
    ) -> Self {
        let set: BTreeSet<Principal> = principals.into_iter().collect();
        Self::new(integrity, ConfidentialityLabel::UserIdentity)
            .with_metadata(PRINCIPAL_METADATA_KEY, principal_list(&set))
    }

    /// Whether the content is trusted.
    pub fn is_trusted(&self) -> bool {
        self.integrity == IntegrityLabel::Trusted
    }

    /// Whether the content is public.
    pub fn is_public(&self) -> bool {
        self.confidentiality == ConfidentialityLabel::Public
    }

    /// The label's principal set, when it carries a valid one.
    pub fn principals(&self) -> Option<BTreeSet<Principal>> {
        canonical_principals(self.metadata.get(PRINCIPAL_METADATA_KEY), "label").ok()
    }

    /// The wire form: `{"integrity", "confidentiality", "metadata"?}`.
    pub fn to_value(&self) -> Value {
        let mut out = Map::new();
        out.insert("integrity".into(), json!(self.integrity.as_str()));
        out.insert(
            "confidentiality".into(),
            json!(self.confidentiality.as_str()),
        );
        if !self.metadata.is_empty() {
            out.insert("metadata".into(), Value::Object(self.metadata.clone()));
        }
        Value::Object(out)
    }

    /// Parse the wire form leniently, as upstream's `from_dict`: missing
    /// fields default to `TRUSTED` / `PUBLIC`; unknown values are an error.
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let map = value
            .as_object()
            .ok_or_else(|| "a security label must be an object".to_string())?;
        let integrity = match map.get("integrity") {
            None | Some(Value::Null) => IntegrityLabel::Trusted,
            Some(v) => v
                .as_str()
                .and_then(IntegrityLabel::parse)
                .ok_or_else(|| "invalid integrity".to_string())?,
        };
        let confidentiality = match map.get("confidentiality") {
            None | Some(Value::Null) => ConfidentialityLabel::Public,
            Some(v) => v
                .as_str()
                .and_then(ConfidentialityLabel::parse)
                .ok_or_else(|| "invalid confidentiality".to_string())?,
        };
        let metadata = map
            .get("metadata")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        Ok(Self {
            integrity,
            confidentiality,
            metadata,
        })
    }
}

impl std::fmt::Display for ContentLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.integrity, self.confidentiality)
    }
}

/// Parse explicit label fields strictly (both `integrity` and
/// `confidentiality` required). Principal metadata is kept only when
/// `allow_principals` (a framework-produced label) and the label is
/// `USER_IDENTITY`, where it must then be valid.
pub(crate) fn parse_content_label(
    value: &Value,
    source: &str,
    allow_principals: bool,
) -> Result<ContentLabel, String> {
    let map = value
        .as_object()
        .ok_or_else(|| format!("{source} security label must be an object"))?;
    let integrity = map
        .get("integrity")
        .and_then(Value::as_str)
        .and_then(IntegrityLabel::parse);
    let confidentiality = map
        .get("confidentiality")
        .and_then(Value::as_str)
        .and_then(ConfidentialityLabel::parse);
    let (Some(integrity), Some(confidentiality)) = (integrity, confidentiality) else {
        return Err(format!(
            "{source} security label requires integrity and confidentiality"
        ));
    };
    let mut metadata = map
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if allow_principals && confidentiality == ConfidentialityLabel::UserIdentity {
        let principals = canonical_principals(metadata.get(PRINCIPAL_METADATA_KEY), source)?;
        metadata.insert(PRINCIPAL_METADATA_KEY.into(), principal_list(&principals));
    } else {
        metadata.remove(PRINCIPAL_METADATA_KEY);
    }
    Ok(ContentLabel {
        integrity,
        confidentiality,
        metadata,
    })
}

/// Combine labels under the most restrictive policy: `UNTRUSTED` if any is,
/// the highest confidentiality, the union of `USER_IDENTITY` principal sets
/// (dropped when any is invalid), and merged other metadata. No labels give
/// the default `TRUSTED` + `PUBLIC`.
pub fn combine_labels<'a>(labels: impl IntoIterator<Item = &'a ContentLabel>) -> ContentLabel {
    let labels: Vec<&ContentLabel> = labels.into_iter().collect();
    if labels.is_empty() {
        return ContentLabel::default();
    }
    let integrity = if labels
        .iter()
        .any(|l| l.integrity == IntegrityLabel::Untrusted)
    {
        IntegrityLabel::Untrusted
    } else {
        IntegrityLabel::Trusted
    };
    let confidentiality = labels
        .iter()
        .map(|l| l.confidentiality)
        .max()
        .unwrap_or_default();
    let mut metadata = Map::new();
    for label in &labels {
        for (k, v) in &label.metadata {
            if k != PRINCIPAL_METADATA_KEY {
                metadata.insert(k.clone(), v.clone());
            }
        }
    }
    if confidentiality == ConfidentialityLabel::UserIdentity {
        let mut union = BTreeSet::new();
        let mut valid = true;
        for label in labels
            .iter()
            .filter(|l| l.confidentiality == ConfidentialityLabel::UserIdentity)
        {
            match canonical_principals(
                label.metadata.get(PRINCIPAL_METADATA_KEY),
                "USER_IDENTITY label",
            ) {
                Ok(set) => union.extend(set),
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        }
        if valid && !union.is_empty() {
            metadata.insert(PRINCIPAL_METADATA_KEY.into(), principal_list(&union));
        }
    }
    ContentLabel {
        integrity,
        confidentiality,
        metadata,
    }
}

/// Whether data labeled `label` may be written to a destination accepting
/// at most `max_allowed`. `USER_IDENTITY` data additionally needs a valid
/// principal set that is a subset of `authorized_principals`.
pub fn check_confidentiality_allowed(
    label: &ContentLabel,
    max_allowed: ConfidentialityLabel,
    authorized_principals: Option<&[Principal]>,
) -> bool {
    if label.confidentiality > max_allowed {
        return false;
    }
    if label.confidentiality != ConfidentialityLabel::UserIdentity {
        return true;
    }
    let Ok(source) = canonical_principals(label.metadata.get(PRINCIPAL_METADATA_KEY), "source")
    else {
        return false;
    };
    let Some(destination) = authorized_principals.filter(|p| !p.is_empty()) else {
        return false;
    };
    let destination: BTreeSet<&Principal> = destination.iter().collect();
    source.iter().all(|p| destination.contains(p))
}

/// A standalone store of hidden content behind variable ids (`var_<hex>`).
///
/// The label-tracking middleware keeps its own per-session store (see
/// [`LabelTrackingFunctionMiddleware::variable_store`](super::LabelTrackingFunctionMiddleware::variable_store));
/// this one backs [`store_untrusted_content`](super::store_untrusted_content).
#[derive(Debug, Default)]
pub struct ContentVariableStore {
    storage: Mutex<Vec<(String, Value, ContentLabel)>>,
}

impl ContentVariableStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `content` and return its new variable id.
    pub fn store(&self, content: Value, label: ContentLabel) -> String {
        let id = new_variable_id();
        self.storage
            .lock()
            .unwrap()
            .push((id.clone(), content, label));
        id
    }

    /// The content and label behind `id`, if present.
    pub fn retrieve(&self, id: &str) -> Option<(Value, ContentLabel)> {
        self.storage
            .lock()
            .unwrap()
            .iter()
            .find(|(k, _, _)| k == id)
            .map(|(_, v, l)| (v.clone(), l.clone()))
    }

    /// Whether `id` exists.
    pub fn exists(&self, id: &str) -> bool {
        self.storage.lock().unwrap().iter().any(|(k, _, _)| k == id)
    }

    /// Remove everything.
    pub fn clear(&self) {
        self.storage.lock().unwrap().clear();
    }

    /// The stored ids, in insertion order.
    pub fn list_variables(&self) -> Vec<String> {
        self.storage
            .lock()
            .unwrap()
            .iter()
            .map(|(k, _, _)| k.clone())
            .collect()
    }
}

/// A fresh `var_<16 hex>` id.
pub(crate) fn new_variable_id() -> String {
    format!("var_{}", &uuid::Uuid::new_v4().simple().to_string()[..16])
}

/// A reference to hidden content, placed in the model context instead of
/// the content itself.
#[derive(Debug, Clone, PartialEq)]
pub struct VariableReferenceContent {
    /// The variable id.
    pub variable_id: String,
    /// The hidden content's label.
    pub label: ContentLabel,
    /// What the content is.
    pub description: Option<String>,
}

impl VariableReferenceContent {
    /// A reference.
    pub fn new(
        variable_id: impl Into<String>,
        label: ContentLabel,
        description: Option<String>,
    ) -> Self {
        Self {
            variable_id: variable_id.into(),
            label,
            description,
        }
    }

    /// The wire form: `{"type": "variable_reference", "variable_id",
    /// "security_label", "description"?}`.
    pub fn to_value(&self) -> Value {
        let mut out = Map::new();
        out.insert("type".into(), json!("variable_reference"));
        out.insert("variable_id".into(), json!(self.variable_id));
        out.insert("security_label".into(), self.label.to_value());
        if let Some(d) = &self.description {
            out.insert("description".into(), json!(d));
        }
        Value::Object(out)
    }

    /// Parse the wire form (`label` is accepted as a legacy key for
    /// `security_label`).
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let variable_id = value
            .get("variable_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "variable_id is required".to_string())?;
        let label = match value.get("security_label").or_else(|| value.get("label")) {
            Some(l @ Value::Object(_)) => ContentLabel::from_value(l)?,
            _ => ContentLabel::default(),
        };
        Ok(Self {
            variable_id: variable_id.to_string(),
            label,
            description: value
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string),
        })
    }
}

/// A message with its security label and provenance.
///
/// Wraps a [`Message`] (it dereferences to it). Without an explicit label
/// one is inferred from the role: `user` and `system` are trusted, `tool`
/// untrusted, `assistant` the combination of its source labels (trusted
/// with none), and any other role untrusted.
#[derive(Debug, Clone, PartialEq)]
pub struct LabeledMessage {
    /// The message.
    pub message: Message,
    /// Its label.
    pub security_label: ContentLabel,
    /// Its index in the conversation.
    pub message_index: Option<usize>,
    /// Labels of the content that contributed to it.
    pub source_labels: Vec<ContentLabel>,
    /// Extra metadata.
    pub metadata: HashMap<String, Value>,
}

impl LabeledMessage {
    /// Wrap `message`, inferring its label from the role and
    /// `source_labels` when `security_label` is `None`.
    pub fn new(
        message: Message,
        security_label: Option<ContentLabel>,
        source_labels: Vec<ContentLabel>,
    ) -> Self {
        let security_label =
            security_label.unwrap_or_else(|| infer_label(&message.role, &source_labels));
        Self {
            message,
            security_label,
            message_index: None,
            source_labels,
            metadata: HashMap::new(),
        }
    }

    /// Builder: the conversation index.
    pub fn with_index(mut self, index: usize) -> Self {
        self.message_index = Some(index);
        self
    }

    /// Whether the message is trusted.
    pub fn is_trusted(&self) -> bool {
        self.security_label.is_trusted()
    }

    /// The wire form: `{"role", "content", "security_label",
    /// "message_index"?, "source_labels"?, "metadata"?}`.
    pub fn to_value(&self) -> Value {
        let mut out = Map::new();
        out.insert("role".into(), json!(self.message.role.to_string()));
        out.insert("content".into(), json!(self.message.text()));
        out.insert("security_label".into(), self.security_label.to_value());
        if let Some(i) = self.message_index {
            out.insert("message_index".into(), json!(i));
        }
        if !self.source_labels.is_empty() {
            out.insert(
                "source_labels".into(),
                Value::Array(
                    self.source_labels
                        .iter()
                        .map(ContentLabel::to_value)
                        .collect(),
                ),
            );
        }
        if !self.metadata.is_empty() {
            out.insert("metadata".into(), json!(self.metadata));
        }
        Value::Object(out)
    }

    /// Parse the wire form.
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let role = value
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| "role is required".to_string())?;
        let content = match value.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) | None => String::new(),
            Some(other) => other.to_string(),
        };
        let source_labels = value
            .get("source_labels")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(ContentLabel::from_value).collect())
            .transpose()?
            .unwrap_or_default();
        let label = value
            .get("security_label")
            .map(ContentLabel::from_value)
            .transpose()?;
        let mut out = Self::new(Message::new(Role::new(role), content), label, source_labels);
        out.message_index = value
            .get("message_index")
            .and_then(Value::as_u64)
            .map(|i| i as usize);
        if let Some(m) = value.get("metadata").and_then(Value::as_object) {
            out.metadata = m.clone().into_iter().collect();
        }
        Ok(out)
    }
}

impl std::ops::Deref for LabeledMessage {
    type Target = Message;
    fn deref(&self) -> &Message {
        &self.message
    }
}

fn auto(reason: &str, integrity: IntegrityLabel) -> ContentLabel {
    ContentLabel::new(integrity, ConfidentialityLabel::Public)
        .with_metadata("auto_labeled", json!(true))
        .with_metadata("reason", json!(reason))
}

fn infer_label(role: &Role, source_labels: &[ContentLabel]) -> ContentLabel {
    match role.to_string().as_str() {
        r @ ("user" | "system") => auto(&format!("{r}_message"), IntegrityLabel::Trusted),
        "assistant" if !source_labels.is_empty() => combine_labels(source_labels),
        "assistant" => auto("assistant_no_sources", IntegrityLabel::Trusted),
        "tool" => auto("tool_result", IntegrityLabel::Untrusted),
        other => auto(&format!("unknown_role_{other}"), IntegrityLabel::Untrusted),
    }
}
