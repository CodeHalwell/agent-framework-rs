//! A client for the Foundry storage API's state stores (upstream agentserver
//! `FoundryStateStore`): named stores of JSON items with ETags.
//!
//! Wire shape: `{storage base}/state_stores[/<name>[/items[/<key>]]]?api-version=v1`,
//! where names and keys are unpadded URL-safe base64. Writes are
//! `POST …/items` (create, fails on a duplicate key) and `PUT …/items/<key>`
//! (create or replace, with an optional `If-Match`).

use std::sync::Arc;

use agent_framework_azure::TokenCredential;
use agent_framework_core::error::Error;
use base64::Engine;
use reqwest::{Method, StatusCode};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::scope::headers;

/// The OAuth scope for the Foundry storage API.
pub const FOUNDRY_TOKEN_SCOPE: &str = "https://ai.azure.com/.default";
/// The default storage API version.
pub const DEFAULT_API_VERSION: &str = "v1";
/// The default item time-to-live: 30 days.
pub const DEFAULT_ITEM_TTL_SECONDS: u64 = 30 * 24 * 60 * 60;
/// The header carrying the delegated user id on item operations.
pub const DELEGATED_USER_ID_HEADER: &str = "x-ms-user-id";

/// A storage failure, classified as upstream classifies it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    /// 404.
    NotFound(String),
    /// 409: a duplicate key or store.
    Conflict(String),
    /// 412: an `If-Match` that did not match.
    PreconditionFailed(String),
    /// Any other failure.
    Other(String),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(m) => write!(f, "Foundry storage: not found: {m}"),
            Self::Conflict(m) => write!(f, "Foundry storage: conflict: {m}"),
            Self::PreconditionFailed(m) => write!(f, "Foundry storage: precondition failed: {m}"),
            Self::Other(m) => write!(f, "Foundry storage: {m}"),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<StorageError> for Error {
    fn from(e: StorageError) -> Self {
        Error::Service(e.to_string())
    }
}

/// The storage endpoint (upstream `FoundryStorageEndpoint`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundryStorageEndpoint {
    /// The `…/storage/` base URL.
    pub storage_base_url: String,
    /// The API version appended to every request.
    pub api_version: String,
}

impl FoundryStorageEndpoint {
    /// From a Foundry project endpoint: `<endpoint>/storage/` (an endpoint
    /// already ending in `/storage` is kept).
    pub fn from_endpoint(endpoint: &str) -> Result<Self, StorageError> {
        if !(endpoint.starts_with("http://") || endpoint.starts_with("https://")) {
            return Err(StorageError::Other(format!(
                "endpoint must be a valid absolute URL, got: {endpoint:?}"
            )));
        }
        let normalized = endpoint.trim_end_matches('/');
        let storage_base_url = if normalized.ends_with("/storage") {
            format!("{normalized}/")
        } else {
            format!("{normalized}/storage/")
        };
        Ok(Self {
            storage_base_url,
            api_version: DEFAULT_API_VERSION.into(),
        })
    }

    /// The URL for `path` with `api-version` and `extra` query parameters.
    pub fn build_url(&self, path: &str, extra: &[(&str, &str)]) -> String {
        let mut url = format!(
            "{}{path}?api-version={}",
            self.storage_base_url,
            encode_query(&self.api_version)
        );
        for (k, v) in extra {
            url.push_str(&format!("&{}={}", encode_query(k), encode_query(v)));
        }
        url
    }
}

fn encode_query(value: &str) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn encode_segment(value: &str) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(value.as_bytes())
}

/// One stored item.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct StateStoreItem {
    /// The key.
    pub key: String,
    /// The JSON object value.
    #[serde(default)]
    pub value: Map<String, Value>,
    /// The optimistic-concurrency token.
    #[serde(default)]
    pub etag: String,
    /// Tags.
    #[serde(default)]
    pub tags: Option<Map<String, Value>>,
}

/// A write's acknowledgement.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct StateStoreItemRef {
    /// The key.
    pub key: String,
    /// The new ETag.
    #[serde(default)]
    pub etag: String,
}

/// One page of keys.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KeyPage {
    /// The keys.
    pub keys: Vec<String>,
    /// Whether more pages follow.
    pub has_more: bool,
    /// The last key's cursor, for `after`.
    pub last_id: Option<String>,
}

/// Settings shared by every store a server opens: the endpoint, the
/// credential, and the HTTP client.
#[derive(Clone)]
pub struct FoundryStorage {
    endpoint: FoundryStorageEndpoint,
    credential: Option<Arc<dyn TokenCredential>>,
    http: reqwest::Client,
}

impl std::fmt::Debug for FoundryStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryStorage")
            .field("endpoint", &self.endpoint)
            .field("credential", &self.credential.is_some())
            .finish()
    }
}

impl FoundryStorage {
    /// Storage at `endpoint`, authenticating with `credential` (scope
    /// [`FOUNDRY_TOKEN_SCOPE`]). Without a credential no `Authorization`
    /// header is sent (for a local emulator or a test double).
    pub fn new(
        endpoint: FoundryStorageEndpoint,
        credential: Option<Arc<dyn TokenCredential>>,
    ) -> Self {
        Self {
            endpoint,
            credential,
            http: reqwest::Client::new(),
        }
    }

    /// The endpoint.
    pub fn endpoint(&self) -> &FoundryStorageEndpoint {
        &self.endpoint
    }

    /// A handle on the store `name`, creating it if needed (upstream
    /// `FoundryStateStore.get_or_create`): fetch, else create, and on a
    /// creation race fetch again.
    pub async fn get_or_create(
        &self,
        name: &str,
        options: StoreOptions,
    ) -> Result<FoundryStateStore, StorageError> {
        let store = FoundryStateStore {
            storage: self.clone(),
            name: name.to_string(),
            user_id: options.user_id.clone(),
            call_id: options.call_id.clone(),
        };
        match store.fetch().await {
            Ok(()) => return Ok(store),
            Err(StorageError::NotFound(_)) => {}
            Err(e) => return Err(e),
        }
        let body = json!({
            "name": name,
            "user_isolation": options.user_isolation,
            "item_ttl_seconds": options.item_ttl_seconds,
        });
        match store
            .send(Method::POST, "state_stores", Some(body), false, None, &[])
            .await
        {
            Ok(_) => Ok(store),
            Err(StorageError::Conflict(_)) => store.fetch().await.map(|()| store),
            Err(e) => Err(e),
        }
    }
}

/// How a store is opened.
#[derive(Debug, Clone)]
pub struct StoreOptions {
    /// Whether items are isolated per delegated user.
    pub user_isolation: bool,
    /// Item time-to-live, in seconds.
    pub item_ttl_seconds: u64,
    /// The delegated user id sent on item operations.
    pub user_id: Option<String>,
    /// The platform call id forwarded on every request.
    pub call_id: Option<String>,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            user_isolation: true,
            item_ttl_seconds: DEFAULT_ITEM_TTL_SECONDS,
            user_id: None,
            call_id: None,
        }
    }
}

/// A client bound to one named state store.
#[derive(Debug, Clone)]
pub struct FoundryStateStore {
    storage: FoundryStorage,
    name: String,
    user_id: Option<String>,
    call_id: Option<String>,
}

impl FoundryStateStore {
    /// The store's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    fn store_path(&self) -> String {
        format!("state_stores/{}", encode_segment(&self.name))
    }

    fn item_path(&self, key: &str) -> Result<String, StorageError> {
        if key.is_empty() {
            return Err(StorageError::Other("key must be a non-empty string".into()));
        }
        Ok(format!(
            "{}/items/{}",
            self.store_path(),
            encode_segment(key)
        ))
    }

    async fn fetch(&self) -> Result<(), StorageError> {
        self.send(Method::GET, &self.store_path(), None, false, None, &[])
            .await
            .map(|_| ())
    }

    async fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
        include_user_id: bool,
        if_match: Option<&str>,
        query: &[(&str, &str)],
    ) -> Result<Value, StorageError> {
        let url = self.storage.endpoint.build_url(path, query);
        let mut request = self.storage.http.request(method, &url);
        if let Some(credential) = &self.storage.credential {
            let token = credential
                .get_token_for_scope(FOUNDRY_TOKEN_SCOPE)
                .await
                .map_err(|e| StorageError::Other(format!("credential: {e}")))?;
            request = request.bearer_auth(token);
        }
        if let Some(call_id) = &self.call_id {
            request = request.header(headers::FOUNDRY_CALL_ID, call_id);
        }
        if include_user_id {
            if let Some(user) = &self.user_id {
                request = request.header(DELEGATED_USER_ID_HEADER, user);
            }
        }
        if let Some(tag) = if_match {
            request = request.header("If-Match", tag);
        }
        if let Some(body) = body {
            request = request
                .header("Content-Type", "application/json; charset=utf-8")
                .body(body.to_string());
        }
        let response = request
            .send()
            .await
            .map_err(|e| StorageError::Other(format!("request failed: {e}")))?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            // The detail names the status only; bodies may echo stored data.
            let detail = format!("HTTP {}", status.as_u16());
            return Err(match status {
                StatusCode::NOT_FOUND => StorageError::NotFound(detail),
                StatusCode::CONFLICT => StorageError::Conflict(detail),
                StatusCode::PRECONDITION_FAILED => StorageError::PreconditionFailed(detail),
                _ => StorageError::Other(detail),
            });
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text)
            .map_err(|e| StorageError::Other(format!("invalid response body: {e}")))
    }

    /// Fetch one item; `None` when absent.
    pub async fn get_item(&self, key: &str) -> Result<Option<StateStoreItem>, StorageError> {
        match self
            .send(Method::GET, &self.item_path(key)?, None, true, None, &[])
            .await
        {
            Ok(body) => serde_json::from_value(body)
                .map(Some)
                .map_err(|e| StorageError::Other(format!("invalid item: {e}"))),
            Err(StorageError::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Create an item; [`StorageError::Conflict`] when the key exists.
    pub async fn create_item(
        &self,
        key: &str,
        value: Map<String, Value>,
    ) -> Result<StateStoreItemRef, StorageError> {
        let path = format!("{}/items", self.store_path());
        let body = self
            .send(
                Method::POST,
                &path,
                Some(json!({"key": key, "value": value})),
                true,
                None,
                &[],
            )
            .await?;
        parse_ref(body)
    }

    /// Create or replace an item. With `if_match` the write succeeds only
    /// when the current ETag matches (`"*"`: only when it exists).
    pub async fn set_item(
        &self,
        key: &str,
        value: Map<String, Value>,
        if_match: Option<&str>,
    ) -> Result<StateStoreItemRef, StorageError> {
        let body = self
            .send(
                Method::PUT,
                &self.item_path(key)?,
                Some(json!({"value": value})),
                true,
                if_match,
                &[],
            )
            .await?;
        parse_ref(body)
    }

    /// Delete an item.
    pub async fn delete_item(&self, key: &str, if_match: Option<&str>) -> Result<(), StorageError> {
        self.send(
            Method::DELETE,
            &self.item_path(key)?,
            None,
            true,
            if_match,
            &[],
        )
        .await
        .map(|_| ())
    }

    /// List keys, newest first, following every page.
    pub async fn list_keys(&self) -> Result<Vec<String>, StorageError> {
        let path = format!("{}/items:keys", self.store_path());
        let mut keys = Vec::new();
        let mut after: Option<String> = None;
        loop {
            let mut query = vec![("order", "desc")];
            if let Some(cursor) = &after {
                query.push(("after", cursor.as_str()));
            }
            let body = self
                .send(Method::GET, &path, None, true, None, &query)
                .await?;
            let page = parse_keys(&body);
            keys.extend(page.keys);
            match (page.has_more, page.last_id) {
                (true, Some(last)) => after = Some(last),
                _ => return Ok(keys),
            }
        }
    }
}

fn parse_ref(body: Value) -> Result<StateStoreItemRef, StorageError> {
    serde_json::from_value(body).map_err(|e| StorageError::Other(format!("invalid item ref: {e}")))
}

fn parse_keys(body: &Value) -> KeyPage {
    let keys = body
        .get("data")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|i| i.get("key").and_then(Value::as_str).map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    KeyPage {
        keys,
        has_more: body
            .get("has_more")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        last_id: body
            .get("last_id")
            .and_then(Value::as_str)
            .map(str::to_string),
    }
}
