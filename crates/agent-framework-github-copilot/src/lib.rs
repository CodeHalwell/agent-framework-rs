//! # agent-framework-github-copilot
//!
//! A [GitHub Copilot](https://github.com/features/copilot) chat client for
//! `agent-framework-rs`.
//!
//! GitHub Copilot's chat endpoint is OpenAI Chat-Completions-compatible
//! (`POST {base_url}/chat/completions`, default base URL
//! `https://api.githubcopilot.com`), so request/response conversion is reused
//! from [`agent_framework_openai::convert`] rather than duplicated — the same
//! streaming approach used by `agent-framework-ollama` and
//! `agent-framework-foundry-local` (see [`convert`] for the small,
//! Copilot-specific streaming-delta parser that *is* implemented locally).
//!
//! Unlike those two clients, Copilot's endpoint is not directly reachable
//! with a plain API key: callers supply a long-lived GitHub OAuth token or
//! personal access token (the `github_token`), and before every request this
//! client transparently exchanges it for a short-lived Copilot API bearer
//! token via `GET https://api.github.com/copilot_internal/v2/token`
//! (`Authorization: token <github_token>`). The exchanged token is cached in
//! the client and only re-fetched once it is missing or within about 60
//! seconds of its `expires_at`. Every `/chat/completions` request also
//! carries two headers the Copilot API requires beyond the OpenAI-compatible
//! shape: `Editor-Version` and `Copilot-Integration-Id`; requests missing
//! them are rejected by the real service.
//!
//! ```no_run
//! use agent_framework_github_copilot::GitHubCopilotChatClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = GitHubCopilotChatClient::new("gho_examplegithubtoken", "gpt-4o");
//! let agent = Agent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```
//!
//! Pointing at a non-default (e.g. enterprise or proxied) endpoint:
//!
//! ```no_run
//! use agent_framework_github_copilot::GitHubCopilotChatClient;
//!
//! let client = GitHubCopilotChatClient::new("gho_examplegithubtoken", "gpt-4o")
//!     .with_base_url("https://copilot-proxy.example.internal");
//! # let _ = client;
//! ```

pub mod convert;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::streaming::Utf8StreamDecoder;
use agent_framework_core::types::{ChatOptions, ChatResponse, ChatResponseUpdate, Message};
use futures::StreamExt;
use serde_json::Value;
use tokio::sync::Mutex;

/// The default GitHub Copilot chat-completions base URL.
const DEFAULT_BASE_URL: &str = "https://api.githubcopilot.com";

/// The GitHub token-exchange endpoint used to trade a long-lived GitHub OAuth
/// token / PAT for a short-lived Copilot API bearer token.
const TOKEN_EXCHANGE_URL: &str = "https://api.github.com/copilot_internal/v2/token";

/// The `Editor-Version` header value sent on every `/chat/completions`
/// request. The Copilot API requires this header to be present; its exact
/// value is not validated beyond being well-formed, but it identifies this
/// crate as the calling "editor".
const EDITOR_VERSION: &str = "agent-framework-rs/0.1.0";

/// The `Copilot-Integration-Id` header value sent on every
/// `/chat/completions` request. The Copilot API requires this header to be
/// present to authorize the request as coming from a chat surface.
const COPILOT_INTEGRATION_ID: &str = "vscode-chat";

/// A cached Copilot API token is refreshed once fewer than this many seconds
/// remain before its `expires_at`, to avoid a token expiring mid-flight.
const TOKEN_REFRESH_MARGIN_SECS: i64 = 60;

/// The environment variable [`GitHubCopilotChatClient::from_env`] reads for
/// the long-lived GitHub OAuth token / PAT.
const GITHUB_TOKEN_ENV: &str = "GITHUB_COPILOT_TOKEN";

/// An alternate environment variable [`GitHubCopilotChatClient::from_env`]
/// falls back to when [`GITHUB_TOKEN_ENV`] is unset.
const GITHUB_TOKEN_ENV_ALT: &str = "GH_COPILOT_TOKEN";

/// The environment variable [`GitHubCopilotChatClient::from_env`] reads for a
/// non-default base URL (e.g. an enterprise deployment or a proxy in front of
/// the Copilot API).
const BASE_URL_ENV: &str = "GITHUB_COPILOT_BASE_URL";

/// Parse a `Retry-After` header into a delay in seconds. Mirrors the
/// OpenAI/Anthropic/Azure/Ollama/Foundry Local clients (Copilot's
/// OpenAI-compatibility layer doesn't generally emit this, but is tolerated
/// if a proxy in front of it does).
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<f64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
}

/// The current time as Unix seconds, matching the `expires_at` field returned
/// by the Copilot token-exchange endpoint.
fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Whether a cached Copilot token needs to be (re-)exchanged: `true` if there
/// is no cached token, or if fewer than [`TOKEN_REFRESH_MARGIN_SECS`] seconds
/// remain before it expires. Factored out as a small pure function so the
/// refresh policy is unit-testable without any network access.
fn token_needs_refresh(expires_at: Option<i64>, now: i64) -> bool {
    match expires_at {
        None => true,
        Some(expires_at) => expires_at - now < TOKEN_REFRESH_MARGIN_SECS,
    }
}

/// A cached, exchanged Copilot API token and its expiry (Unix seconds, as
/// returned by the token-exchange endpoint's `expires_at` field).
#[derive(Clone, Debug, PartialEq)]
struct CachedToken {
    token: String,
    expires_at: i64,
}

/// A GitHub Copilot chat client (`POST {base_url}/chat/completions`).
#[derive(Clone)]
pub struct GitHubCopilotChatClient {
    inner: Arc<Inner>,
}

struct Inner {
    http: reqwest::Client,
    base_url: String,
    model: String,
    /// The long-lived GitHub OAuth token / PAT supplied by the caller,
    /// exchanged for short-lived Copilot API tokens by
    /// [`GitHubCopilotChatClient::ensure_copilot_token`].
    github_token: String,
    /// The most recently exchanged Copilot API token, if any, refreshed on
    /// demand by [`GitHubCopilotChatClient::ensure_copilot_token`]. Guarded
    /// by an async mutex since the exchange itself is an async HTTP call.
    copilot_token: Mutex<Option<CachedToken>>,
}

impl Clone for Inner {
    fn clone(&self) -> Self {
        // `Arc::make_mut` (used by the builder setters below) requires
        // `Inner: Clone`, but `tokio::sync::Mutex` intentionally has no
        // `Clone` impl of its own (cloning while a lock might be held is
        // exactly what it prevents). The setters only ever run before the
        // client has been shared (i.e. while the `Arc` refcount is 1), so
        // `Arc::make_mut` never actually needs to materialize a real clone —
        // but the bound must still typecheck, hence this manual impl.
        // `try_lock` is fine here: at refcount 1 nothing else can hold the
        // lock, and if it were somehow held we'd rather start unauthenticated
        // (forcing a fresh exchange) than block synchronously inside `Clone`.
        let copilot_token = self
            .copilot_token
            .try_lock()
            .ok()
            .and_then(|guard| guard.clone());
        Self {
            http: self.http.clone(),
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            github_token: self.github_token.clone(),
            copilot_token: Mutex::new(copilot_token),
        }
    }
}

impl std::fmt::Debug for GitHubCopilotChatClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GitHubCopilotChatClient")
            .field("base_url", &self.inner.base_url)
            .field("model", &self.inner.model)
            .finish_non_exhaustive()
    }
}

impl GitHubCopilotChatClient {
    /// Create a client for the given GitHub OAuth token / PAT and default
    /// model, targeting the default Copilot API base URL.
    ///
    /// `github_token` is *not* sent directly to the chat endpoint — it is
    /// exchanged for a short-lived Copilot API token on first use (see the
    /// crate-level docs).
    pub fn new(github_token: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                base_url: DEFAULT_BASE_URL.to_string(),
                model: model.into(),
                github_token: github_token.into(),
                copilot_token: Mutex::new(None),
            }),
        }
    }

    /// Build a client from the environment: the GitHub OAuth token / PAT is
    /// read from [`GITHUB_TOKEN_ENV`] (`GITHUB_COPILOT_TOKEN`), falling back
    /// to [`GITHUB_TOKEN_ENV_ALT`] (`GH_COPILOT_TOKEN`) if unset; an
    /// [`Error::Configuration`] is returned if neither is set. The optional
    /// [`BASE_URL_ENV`] (`GITHUB_COPILOT_BASE_URL`) overrides the default
    /// base URL when present.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let token = std::env::var(GITHUB_TOKEN_ENV)
            .or_else(|_| std::env::var(GITHUB_TOKEN_ENV_ALT))
            .map_err(|_| {
                Error::Configuration(format!(
                    "{GITHUB_TOKEN_ENV} (or {GITHUB_TOKEN_ENV_ALT}) is not set"
                ))
            })?;
        let mut client = Self::new(token, model);
        if let Ok(base_url) = std::env::var(BASE_URL_ENV) {
            if !base_url.trim().is_empty() {
                client = client.with_base_url(base_url);
            }
        }
        Ok(client)
    }

    /// Override the base URL (e.g. for an enterprise deployment or a proxy in
    /// front of the Copilot API).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// The default model id.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// The configured base URL.
    pub fn base_url(&self) -> &str {
        &self.inner.base_url
    }

    fn build_body(&self, messages: &[Message], options: &ChatOptions, stream: bool) -> Value {
        let model = options
            .model
            .clone()
            .unwrap_or_else(|| self.inner.model.clone());
        convert::build_request(messages, options, &model, stream)
    }

    /// Return a valid Copilot API bearer token, exchanging the configured
    /// GitHub token for a fresh one if none is cached or the cached one is
    /// missing/within [`TOKEN_REFRESH_MARGIN_SECS`] seconds of expiring.
    ///
    /// This performs a live `GET` against [`TOKEN_EXCHANGE_URL`] only when a
    /// refresh is actually needed; the refresh *decision* itself is the pure,
    /// unit-tested [`token_needs_refresh`].
    async fn ensure_copilot_token(&self) -> Result<String> {
        let now = unix_now();
        {
            let guard = self.inner.copilot_token.lock().await;
            if let Some(cached) = guard.as_ref() {
                if !token_needs_refresh(Some(cached.expires_at), now) {
                    return Ok(cached.token.clone());
                }
            }
        }

        let resp = self
            .inner
            .http
            .get(TOKEN_EXCHANGE_URL)
            .header(
                "Authorization",
                format!("token {}", self.inner.github_token),
            )
            .send()
            .await
            .map_err(|e| Error::service(format!("Copilot token exchange failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::service_invalid_auth(format!(
                "Copilot token exchange error {status}: {text}"
            )));
        }

        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid Copilot token exchange response: {e}")))?;
        let token = value
            .get("token")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::service("Copilot token exchange response missing `token` field"))?
            .to_string();
        let expires_at = value
            .get("expires_at")
            .and_then(Value::as_i64)
            .unwrap_or(now + TOKEN_REFRESH_MARGIN_SECS);

        let mut guard = self.inner.copilot_token.lock().await;
        *guard = Some(CachedToken {
            token: token.clone(),
            expires_at,
        });
        Ok(token)
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        let copilot_token = self.ensure_copilot_token().await?;
        let url = format!(
            "{}/chat/completions",
            self.inner.base_url.trim_end_matches('/')
        );
        let resp = self
            .inner
            .http
            .post(&url)
            .bearer_auth(&copilot_token)
            .header("Editor-Version", EDITOR_VERSION)
            .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
            .json(body)
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            // The Copilot chat endpoint is OpenAI-wire-compatible for errors
            // too, so status/body classification is shared verbatim with
            // `agent-framework-openai` rather than duplicated.
            return Err(agent_framework_openai::classify_service_error(
                status.as_u16(),
                &text,
                format!("GitHub Copilot API error {status}: {text}"),
                retry_after,
            ));
        }
        Ok(resp)
    }
}

#[async_trait::async_trait]
impl ChatClient for GitHubCopilotChatClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let body = self.build_body(&messages, &options, false);
        let resp = self.post(&body).await?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        Ok(agent_framework_openai::convert::parse_response(&value))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let body = self.build_body(&messages, &options, true);
        let resp = self.post(&body).await?;
        Ok(parse_sse_stream(resp).boxed())
    }

    fn model(&self) -> Option<&str> {
        Some(&self.inner.model)
    }
}

type ByteStream =
    std::pin::Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>;

/// Turn an SSE HTTP response into a stream of [`ChatResponseUpdate`]s.
fn parse_sse_stream(
    resp: reqwest::Response,
) -> impl futures::Stream<Item = Result<ChatResponseUpdate>> + Send {
    let byte_stream: ByteStream = Box::pin(resp.bytes_stream());
    futures::stream::unfold(
        SseState {
            byte_stream,
            buffer: String::new(),
            utf8: Utf8StreamDecoder::new(),
            queued: VecDeque::new(),
            tool_ids: HashMap::new(),
            done: false,
        },
        |mut state| async move {
            loop {
                if let Some(update) = state.queued.pop_front() {
                    return Some((Ok(update), state));
                }
                if state.done {
                    return None;
                }
                match state.byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        let decoded = state.utf8.push(&bytes);
                        state.buffer.push_str(&decoded);
                        while let Some(pos) = state.buffer.find('\n') {
                            let line = state.buffer[..pos].trim().to_string();
                            state.buffer.drain(..=pos);
                            if let Some(data) = line.strip_prefix("data:") {
                                let data = data.trim();
                                if data == "[DONE]" {
                                    return drain_or_end(state);
                                }
                                if data.is_empty() {
                                    continue;
                                }
                                if let Ok(value) = serde_json::from_str::<Value>(data) {
                                    if let Some(err) = value.get("error") {
                                        let msg = err
                                            .get("message")
                                            .and_then(Value::as_str)
                                            .unwrap_or("unknown stream error")
                                            .to_string();
                                        state.done = true;
                                        return Some((Err(Error::service(msg)), state));
                                    }
                                    if let Some(update) =
                                        convert::parse_delta(&value, &mut state.tool_ids)
                                    {
                                        state.queued.push_back(update);
                                    }
                                }
                            }
                        }
                    }
                    Some(Err(e)) => {
                        state.done = true;
                        return Some((Err(Error::service(format!("stream error: {e}"))), state));
                    }
                    None => return drain_or_end(state),
                }
            }
        },
    )
}

/// State carried across `unfold` iterations while parsing the SSE stream.
struct SseState {
    byte_stream: ByteStream,
    buffer: String,
    utf8: Utf8StreamDecoder,
    queued: VecDeque<ChatResponseUpdate>,
    tool_ids: HashMap<i64, String>,
    done: bool,
}

fn drain_or_end(mut state: SseState) -> Option<(Result<ChatResponseUpdate>, SseState)> {
    match state.queued.pop_front() {
        Some(update) => {
            state.done = true;
            Some((Ok(update), state))
        }
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // region: base URL and constructor defaults

    #[test]
    fn new_sets_default_base_url_and_model() {
        let client = GitHubCopilotChatClient::new("gho_token", "gpt-4o");
        assert_eq!(client.base_url(), DEFAULT_BASE_URL);
        assert_eq!(client.model(), "gpt-4o");
    }

    #[test]
    fn with_base_url_overrides_default() {
        let client = GitHubCopilotChatClient::new("gho_token", "gpt-4o")
            .with_base_url("https://copilot-proxy.example.internal");
        assert_eq!(client.base_url(), "https://copilot-proxy.example.internal");
    }

    #[test]
    fn debug_impl_redacts_tokens() {
        let client = GitHubCopilotChatClient::new("gho_super_secret_token", "gpt-4o");
        let debug = format!("{client:?}");
        assert!(!debug.contains("gho_super_secret_token"));
        assert!(debug.contains("gpt-4o"));
        assert!(debug.contains(DEFAULT_BASE_URL));
    }

    // endregion

    // region: from_env

    /// Guards env mutation: tests within a crate run on multiple threads, and
    /// env vars are process-global.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_errors_when_token_missing() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX against the other env-var tests in
        // this module; no other test in this crate touches these variables.
        unsafe {
            std::env::remove_var(GITHUB_TOKEN_ENV);
            std::env::remove_var(GITHUB_TOKEN_ENV_ALT);
            std::env::remove_var(BASE_URL_ENV);
        }
        let result = GitHubCopilotChatClient::from_env("gpt-4o");
        assert!(matches!(result, Err(Error::Configuration(_))));
    }

    #[test]
    fn from_env_reads_primary_token_var() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::set_var(GITHUB_TOKEN_ENV, "gho_from_env_token");
            std::env::remove_var(GITHUB_TOKEN_ENV_ALT);
            std::env::remove_var(BASE_URL_ENV);
        }
        let client = GitHubCopilotChatClient::from_env("gpt-4o").unwrap();
        assert_eq!(client.model(), "gpt-4o");
        assert_eq!(client.base_url(), DEFAULT_BASE_URL);
        unsafe {
            std::env::remove_var(GITHUB_TOKEN_ENV);
        }
    }

    #[test]
    fn from_env_falls_back_to_alt_token_var() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::remove_var(GITHUB_TOKEN_ENV);
            std::env::set_var(GITHUB_TOKEN_ENV_ALT, "gho_alt_token");
        }
        let client = GitHubCopilotChatClient::from_env("gpt-4o").unwrap();
        assert_eq!(client.model(), "gpt-4o");
        unsafe {
            std::env::remove_var(GITHUB_TOKEN_ENV_ALT);
        }
    }

    #[test]
    fn from_env_reads_base_url_override() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::set_var(GITHUB_TOKEN_ENV, "gho_from_env_token");
            std::env::set_var(BASE_URL_ENV, "https://copilot-proxy.example.internal");
        }
        let client = GitHubCopilotChatClient::from_env("gpt-4o").unwrap();
        assert_eq!(client.base_url(), "https://copilot-proxy.example.internal");
        unsafe {
            std::env::remove_var(GITHUB_TOKEN_ENV);
            std::env::remove_var(BASE_URL_ENV);
        }
    }

    // endregion

    // region: request building

    #[test]
    fn build_body_uses_client_default_model_when_options_model_unset() {
        let client = GitHubCopilotChatClient::new("gho_token", "gpt-4o");
        let body = client.build_body(&[Message::user("hi")], &ChatOptions::new(), false);
        assert_eq!(body["model"], serde_json::json!("gpt-4o"));
        assert_eq!(
            body["messages"],
            serde_json::json!([{ "role": "user", "content": "hi" }])
        );
    }

    #[test]
    fn build_body_prefers_per_request_model() {
        let client = GitHubCopilotChatClient::new("gho_token", "gpt-4o");
        let options = ChatOptions {
            model: Some("claude-3.5-sonnet".to_string()),
            ..ChatOptions::new()
        };
        let body = client.build_body(&[Message::user("hi")], &options, false);
        assert_eq!(body["model"], serde_json::json!("claude-3.5-sonnet"));
    }

    #[test]
    fn build_body_sets_stream_flag() {
        let client = GitHubCopilotChatClient::new("gho_token", "gpt-4o");
        let body = client.build_body(&[Message::user("hi")], &ChatOptions::new(), true);
        assert_eq!(body["stream"], serde_json::json!(true));
    }

    // endregion

    // region: Copilot token refresh predicate (pure, no network)

    #[test]
    fn token_needs_refresh_when_none_cached() {
        assert!(token_needs_refresh(None, 1_000));
    }

    #[test]
    fn token_needs_refresh_when_within_margin_of_expiry() {
        // Expires in 30s, margin is 60s: needs refresh.
        assert!(token_needs_refresh(Some(1_030), 1_000));
    }

    #[test]
    fn token_needs_refresh_when_already_expired() {
        assert!(token_needs_refresh(Some(900), 1_000));
    }

    #[test]
    fn token_does_not_need_refresh_when_comfortably_valid() {
        // Expires in 300s, well beyond the 60s margin.
        assert!(!token_needs_refresh(Some(1_300), 1_000));
    }

    #[test]
    fn token_needs_refresh_boundary_at_exactly_margin() {
        // Exactly at the margin (60s remaining) is comfortably valid — the
        // predicate is a strict `<`, so 60s remaining itself does not
        // trigger a refresh but one second less does. This pins that
        // boundary.
        assert!(!token_needs_refresh(Some(1_060), 1_000));
        assert!(token_needs_refresh(Some(1_059), 1_000));
    }

    // endregion

    // region: Inner clone (used by `Arc::make_mut` in builder setters)

    #[tokio::test]
    async fn inner_clone_preserves_cached_token() {
        let client = GitHubCopilotChatClient::new("gho_token", "gpt-4o");
        {
            let mut guard = client.inner.copilot_token.lock().await;
            *guard = Some(CachedToken {
                token: "cached-copilot-token".to_string(),
                expires_at: 9_999_999_999,
            });
        }
        // Exercises the manual `Clone for Inner` impl the same way
        // `with_base_url`'s `Arc::make_mut` does internally.
        let cloned_inner = client.inner.as_ref().clone();
        let guard = cloned_inner.copilot_token.lock().await;
        assert_eq!(
            guard.as_ref().map(|c| c.token.as_str()),
            Some("cached-copilot-token")
        );
    }

    // endregion

    // region: SSE stream parsing over a synthetic byte stream

    fn sse_bytes(lines: &[String]) -> bytes::Bytes {
        bytes::Bytes::from(lines.join("\n") + "\n\n")
    }

    async fn collect_via_state(text: bytes::Bytes) -> Vec<ChatResponseUpdate> {
        let stream = futures::stream::once(async move { Ok::<_, reqwest::Error>(text) });
        let byte_stream: ByteStream = Box::pin(stream);
        let mut state = SseState {
            byte_stream,
            buffer: String::new(),
            utf8: Utf8StreamDecoder::new(),
            queued: VecDeque::new(),
            tool_ids: HashMap::new(),
            done: false,
        };
        let mut updates = Vec::new();
        if let Some(Ok(bytes)) = state.byte_stream.next().await {
            let decoded = state.utf8.push(&bytes);
            state.buffer.push_str(&decoded);
            while let Some(pos) = state.buffer.find('\n') {
                let line = state.buffer[..pos].trim().to_string();
                state.buffer.drain(..=pos);
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let value: Value = serde_json::from_str(data).unwrap();
                if let Some(update) = convert::parse_delta(&value, &mut state.tool_ids) {
                    updates.push(update);
                }
            }
        }
        updates
    }

    #[tokio::test]
    async fn streaming_chunk_produces_text_update() {
        let chunk = serde_json::json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o",
            "choices": [{ "delta": { "role": "assistant", "content": "Hello" }, "finish_reason": null }],
        });
        let bytes = sse_bytes(&[format!("data: {chunk}")]);
        let updates = collect_via_state(bytes).await;
        assert_eq!(updates.len(), 1);
        let resp = ChatResponse::from_updates(updates);
        assert_eq!(resp.text(), "Hello");
        assert_eq!(resp.response_id.as_deref(), Some("chatcmpl-1"));
    }

    #[tokio::test]
    async fn streaming_tool_call_and_finish_reason_accumulate() {
        let call_chunk = serde_json::json!({
            "id": "chatcmpl-2",
            "choices": [{
                "delta": { "tool_calls": [{ "index": 0, "id": "call_1", "function": { "name": "get_weather", "arguments": "{}" } }] },
                "finish_reason": null,
            }],
        });
        let finish_chunk = serde_json::json!({
            "id": "chatcmpl-2",
            "choices": [{ "delta": {}, "finish_reason": "tool_calls" }],
        });
        let bytes = sse_bytes(&[
            format!("data: {call_chunk}"),
            format!("data: {finish_chunk}"),
        ]);
        let updates = collect_via_state(bytes).await;
        assert_eq!(updates.len(), 2);
        let resp = ChatResponse::from_updates(updates);
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            resp.finish_reason,
            Some(agent_framework_core::types::FinishReason::tool_calls())
        );
    }

    #[tokio::test]
    async fn streaming_done_sentinel_ends_stream_without_extra_update() {
        let chunk = serde_json::json!({
            "id": "chatcmpl-3",
            "choices": [{ "delta": { "content": "hi" }, "finish_reason": null }],
        });
        let bytes = sse_bytes(&[format!("data: {chunk}"), "data: [DONE]".to_string()]);
        let updates = collect_via_state(bytes).await;
        // `collect_via_state` mirrors the raw line loop (no special-casing of
        // [DONE] beyond `continue`), so this asserts the sentinel line never
        // parses into a spurious update, matching the real stream (which
        // instead terminates on it).
        assert_eq!(updates.len(), 1);
    }

    // endregion
}
