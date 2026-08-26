//! Proves the Azure SDK's HTTP client can actually complete a TLS handshake.
//!
//! `#[ignore]`d because it needs outbound network to
//! `login.microsoftonline.com`; run it with
//! `cargo test -p agent-framework-azure --features entra-sdk -- --ignored --nocapture`.
//!
//! It exists because this failure is invisible to every other check. `azure_core`
//! uses **reqwest 0.13** while the rest of the workspace uses 0.12, and cargo
//! does not unify features across semver-incompatible versions — so the
//! workspace's `rustls-tls` does not apply to the SDK's client. With
//! `default-features = false` and no TLS feature of its own, reqwest 0.13
//! resolves with no TLS backend at all: everything still compiles, every unit
//! test still passes, and every HTTPS request fails at run time. The
//! `reqwest_rustls` feature on `azure_core` is what prevents that.
//!
//! The two outcomes are easy to tell apart, and were confirmed both ways:
//!
//! * **TLS working** — Entra answers, so the error is a service error quoting
//!   an `AADSTS…` code and a Trace ID, after a real round trip.
//! * **TLS missing** — no response body ever arrives, so the error is a JSON
//!   parse failure (`expected value at line 1 column 1`) returned instantly.
#![cfg(feature = "entra-sdk")]

use std::sync::Arc;

use agent_framework_azure::{SdkTokenCredential, TokenCredential};

#[tokio::test]
#[ignore = "requires outbound network to login.microsoftonline.com"]
async fn an_https_handshake_to_entra_completes() {
    // A syntactically valid but non-existent tenant/client, so Entra rejects
    // the request on its merits rather than us needing a real secret.
    let cred = azure_identity::ClientSecretCredential::new(
        "cc7d0b33-84c6-4bd1-bbfc-1b5b1cd8ca3a",
        "00000000-0000-0000-0000-000000000000".into(),
        "not-a-real-secret".into(),
        None,
    )
    .expect("credential constructs");

    let adapter: Arc<dyn TokenCredential> = Arc::new(SdkTokenCredential::azure_openai(cred));
    let err = adapter
        .get_token()
        .await
        .expect_err("bogus credentials cannot yield a token")
        .to_string();

    assert!(
        err.contains("AADSTS"),
        "expected a real Entra rejection, which proves the handshake completed; \
         got {err:?} — a JSON parse error here means reqwest 0.13 has no TLS backend"
    );
}
