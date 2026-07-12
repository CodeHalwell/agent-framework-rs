//! # agent-framework-purview
//!
//! [Microsoft Purview](https://learn.microsoft.com/purview/) compliance
//! middleware for `agent-framework-rs`: evaluate an agent's/chat client's
//! outgoing prompt and incoming response against Microsoft Graph
//! `dataSecurityAndGovernance` data-loss-prevention policy, blocking either
//! direction when policy says to.
//!
//! This is the Rust equivalent of `agent_framework_purview`
//! (`PurviewPolicyMiddleware` / `PurviewChatPolicyMiddleware`) in the Python
//! reference implementation, hand-rolled against `reqwest` rather than
//! wrapping `httpx` + `azure-identity` (which have no Rust equivalents in
//! this workspace).
//!
//! ## Scope: what this port narrows down to, and why
//!
//! The Python package's `ScopedContentProcessor` orchestrates **three**
//! Graph endpoints per evaluated message: it first computes applicable
//! *protection scopes* (`protectionScopes/compute`, cached with ETag-based
//! invalidation and a 4-hour default TTL), only calls `processContent`
//! inline when a scope actually applies and demands inline evaluation
//! (queuing it in the background otherwise), and — when no scope applies at
//! all — fires a background `activities/contentActivities` audit log entry
//! instead. This work package's brief scopes this crate down to the single
//! call that actually produces a block/allow verdict:
//!
//! > "PurviewClient: POST the processContent route Python uses ... parse
//! > verdict (policy actions / restrict/block per Python)."
//!
//! So **this crate calls `processContent` directly**, on every evaluated
//! message, and parses *its* `policyActions` for a block verdict. It does
//! **not** call `protectionScopes/compute` or `activities/contentActivities`,
//! does not cache anything, and has no background-task queuing. A
//! consequence worth calling out: since there's no scopes precheck to
//! determine "does any policy even apply here", this port evaluates inline
//! (and synchronously) on every call, which is both simpler and more
//! conservative (never silently skips evaluation the way Python's
//! "no applicable scope → log-only, don't block" fallback can) but forgoes
//! the offline/background execution mode and the performance benefit of
//! Python's caching layer. [`PurviewSettings::cache_ttl_seconds`] /
//! [`PurviewSettings::max_cache_size_bytes`] are still present on the
//! settings struct (parity with Python's configuration surface) but are
//! currently unused by this port.
//!
//! Two further, independent scope cuts, both driven by this crate's
//! self-contained [`TokenProvider`] (see the `auth` module docs — there's no
//! `azure-identity`-equivalent credential to introspect a JWT with here):
//!
//! - **No token-derived `tenant_id`/`user_id`/app-location fallback.**
//!   Python's `PurviewClient.get_user_info_from_token` decodes the bearer
//!   token's JWT payload (`tid`/`oid`/`appid` claims, unverified — no
//!   signature check either side) to fill in `tenant_id` when
//!   `PurviewSettings.tenant_id` is unset, and an application-location
//!   fallback when `PurviewSettings.purview_app_location` is unset. This
//!   port requires both to be set explicitly on [`PurviewSettings`] —
//!   attempting to evaluate without them is a [`agent_framework_core::error::Error::Configuration`]
//!   error (itself subject to [`PurviewSettings::ignore_exceptions`], same
//!   as any other evaluation failure).
//! - **User id resolution from messages is still ported.** The *other* half
//!   of Python's user-id resolution — scanning `Message::additional_properties["user_id"]`,
//!   falling back to a GUID-shaped `author_name` — has nothing to do with
//!   the bearer token and is faithfully ported; see
//!   [`processor::resolve_user_id`].
//!
//! ## A curious fidelity note: both directions check `UploadText`
//!
//! Python's `Activity` enum has both `UPLOAD_TEXT` and `DOWNLOAD_TEXT`
//! variants, and it would be reasonable to expect the response-direction
//! (egress) check to use `DOWNLOAD_TEXT`. It does not: both
//! `PurviewPolicyMiddleware.process`'s pre- *and* post-check call
//! `self._processor.process_messages(messages, Activity.UPLOAD_TEXT, ...)`
//! — the exact same activity constant, confirmed against the Python
//! package's own test suite
//! (`tests/test_middleware.py::test_middleware_processor_receives_correct_activity`
//! asserts `Activity.UPLOAD_TEXT` for *both* calls). This port mirrors that
//! exactly rather than "fixing" it — see
//! [`processor::ContentProcessor::evaluate`]'s caller in [`middleware`].
//!
//! ## Layout
//!
//! - [`settings`] — [`PurviewSettings`], [`PurviewAppLocation`],
//!   [`PurviewLocationType`].
//! - [`auth`] — [`TokenProvider`]: bring-your-own bearer token (see "Auth
//!   burden" below).
//! - [`models`] — the `processContent` request/response wire shapes.
//! - [`client`] — [`PurviewClient`]: the single `processContent` HTTP call.
//! - [`processor`] — [`processor::ContentProcessor`]: message → request
//!   mapping, user-id resolution, and the per-message evaluate-until-blocked
//!   loop.
//! - [`middleware`] — [`PurviewAgentMiddleware`] / [`PurviewChatMiddleware`]:
//!   the two middleware hook points.
//!
//! ## Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_core::prelude::*;
//! use agent_framework_purview::{
//!     PurviewAgentMiddleware, PurviewAppLocation, PurviewLocationType, PurviewSettings,
//!     StaticTokenProvider,
//! };
//!
//! # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
//! let settings = PurviewSettings::new("My App")
//!     .with_tenant_id("00000000-0000-0000-0000-000000000000")
//!     .with_purview_app_location(PurviewAppLocation::new(
//!         PurviewLocationType::Application,
//!         "00000000-0000-0000-0000-000000000001", // the app registration's client id
//!     ));
//! // Bring your own Microsoft Graph bearer token (see the `auth` module docs).
//! let token_provider = StaticTokenProvider::new("<graph-bearer-token>");
//! let middleware = PurviewAgentMiddleware::new(token_provider, settings);
//!
//! let agent = Agent::builder(client)
//!     .instructions("You are a helpful assistant.")
//!     .middleware(Arc::new(middleware))
//!     .build();
//!
//! // A `Message` needs a GUID-shaped `additional_properties["user_id"]`
//! // (or `author_name`) for policy evaluation to run at all -- see the
//! // crate docs' "Scope" section.
//! let mut message = Message::user("Summarize this quarter's roadmap.");
//! message
//!     .additional_properties
//!     .insert("user_id".into(), serde_json::json!("00000000-0000-0000-0000-0000000000aa"));
//!
//! let response = agent.run(vec![message], None).await?;
//! println!("{}", response.text());
//! # Ok(())
//! # }
//! ```
//!
//! ## Auth burden
//!
//! The Python reference accepts an `azure-identity` `TokenCredential` /
//! `AsyncTokenCredential` directly, inheriting whatever credential chain the
//! caller already has configured. This crate has no such dependency and is
//! deliberately self-contained per this work package's brief: implement
//! [`TokenProvider`] to bring a Microsoft Graph bearer token (scope
//! `https://graph.microsoft.com/.default`, or the equivalent for a custom
//! [`PurviewSettings::graph_base_uri`] — see [`PurviewSettings::get_scopes`]),
//! carrying the `dataSecurityAndGovernance` Graph permission. See the `auth`
//! module docs for the full rationale; [`StaticTokenProvider`] is provided
//! for a fixed/pre-fetched token (tests, short scripts, externally-managed
//! refresh).
//!
//! ## Divergences from the Python reference (summary)
//!
//! - Calls only `processContent` — no `protectionScopes/compute` precheck,
//!   no caching, no background `contentActivities` logging (see "Scope"
//!   above).
//! - No JWT-derived `tenant_id` / app-location fallback — both must be set
//!   explicitly on [`PurviewSettings`] (see "Scope" above).
//! - No distinct exception *type* hierarchy
//!   (`PurviewAuthenticationError`/`PurviewRateLimitError`/...): every HTTP
//!   failure becomes [`agent_framework_core::error::Error::ServiceStatus`],
//!   carrying the real status code; callers (namely this crate's own
//!   middleware) branch on [`agent_framework_core::error::Error::status`]
//!   the same way Python's `except PurviewPaymentRequiredError` branches on
//!   exception type — status 402 is still special-cased identically (see
//!   [`client::PurviewClient::process_content`] and
//!   [`middleware::PurviewAgentMiddleware`]).

pub mod auth;
pub mod client;
pub mod middleware;
pub mod models;
pub mod processor;
pub mod settings;

pub use auth::{StaticTokenProvider, TokenProvider};
pub use client::PurviewClient;
pub use middleware::{PurviewAgentMiddleware, PurviewChatMiddleware};
pub use models::{
    Activity, DlpAction, DlpActionInfo, ProcessContentRequest, ProcessContentResponse,
    ProtectionScopeState, RestrictionAction,
};
pub use processor::ContentProcessor;
pub use settings::{PurviewAppLocation, PurviewLocationType, PurviewSettings};
