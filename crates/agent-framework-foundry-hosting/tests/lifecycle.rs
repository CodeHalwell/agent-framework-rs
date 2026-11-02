//! The after-restore lifecycle route (its own test binary: it sets process
//! environment variables).

mod common;

use std::sync::{Arc, Mutex};

use agent_framework_foundry_hosting::{AgentSessionContext, HostOptions, ResponsesHostServer};
use axum::http::StatusCode;
use common::*;
use serde_json::json;

/// Each hook call's context and the overridden variable it saw.
type Seen = Arc<Mutex<Vec<(AgentSessionContext, Option<String>)>>>;

#[tokio::test]
async fn after_restore_binds_the_session_and_applies_its_environment() {
    let seen: Seen = Arc::default();
    let witness = seen.clone();
    let fail_next = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let fail = fail_next.clone();
    let options = HostOptions::default().after_restore_with_context(Arc::new(move |ctx| {
        let witness = witness.clone();
        let fail = fail.clone();
        Box::pin(async move {
            let env = std::env::var("AF_TEST_RESTORE_VALUE").ok();
            witness.lock().unwrap().push((ctx, env));
            if fail.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(agent_framework_core::error::Error::Other("boom".into()));
            }
            Ok(())
        })
    }));
    let router = ResponsesHostServer::agent(agent(Echo::default()))
        .with_host_options(options)
        .into_router();
    let restore = |session: &str, restore: &str, value: &str| {
        post(
            "/_agent/after-restore",
            json!({"session_context": {
                "session_id": session,
                "restore_id": restore,
                "session_env_overrides": {"AF_TEST_RESTORE_VALUE": value},
            }}),
        )
    };

    let (status, body) = json(&router, restore("s1", "r1", "one")).await;
    assert_eq!((status, body), (StatusCode::OK, json!({"status": "ok"})));
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].0.session_id, "s1");
        assert_eq!(seen[0].1.as_deref(), Some("one"), "applied before the hook");
    }
    assert_eq!(
        std::env::var("FOUNDRY_AGENT_SESSION_ID").as_deref(),
        Ok("s1")
    );

    // A repeated restore id is a no-op; another session is refused.
    let (status, _) = json(&router, restore("s1", "r1", "ignored")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(seen.lock().unwrap().len(), 1);
    let (status, body) = json(&router, restore("s2", "r2", "x")).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"]["code"], "session_mismatch");

    // A failing hook restores the previous environment.
    fail_next.store(true, std::sync::atomic::Ordering::SeqCst);
    let (status, _) = json(&router, restore("s1", "r3", "three")).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(std::env::var("AF_TEST_RESTORE_VALUE").as_deref(), Ok("one"));

    for bad in [
        json!({}),
        json!({"session_context": {"session_id": "s1"}}),
        json!({"session_context": {"session_id": "s1", "restore_id": "r", "session_env_overrides": {"A=B": "x"}}}),
        json!({"session_context": {"session_id": "s1", "restore_id": "r", "session_env_overrides": {"FOUNDRY_AGENT_SESSION_ID": "other"}}}),
    ] {
        let (status, body) = json(&router, post("/_after_restore", bad)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
}
