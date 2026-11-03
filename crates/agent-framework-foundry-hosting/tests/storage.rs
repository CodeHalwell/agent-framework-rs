//! Foundry state-store clients and stores against an in-process double of
//! the storage API (ETags, `If-Match`, create conflicts, key listing).

mod common;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use agent_framework_azure::StaticTokenCredential;
use agent_framework_core::types::Content;
use agent_framework_core::workflow::{CheckpointStorage, WorkflowCheckpoint};
use agent_framework_foundry_hosting::stores::FunctionApprovalStore;
use agent_framework_foundry_hosting::{
    FoundryCheckpointStore, FoundryFunctionApprovalStore, FoundryRequestContext,
    FoundryRequestScope, FoundryStorage, FoundryStorageEndpoint,
};
#[cfg(feature = "session-store")]
use agent_framework_foundry_hosting::{
    FoundryHostConfig, FoundrySessionStoreProvider, InvocationsHostServer,
};
use axum::extract::{Request, State};
use axum::http::{Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use base64::Engine;
#[cfg(feature = "session-store")]
use common::*;
use http_body_util::BodyExt;
use serde_json::{json, Value};

#[derive(Default)]
struct Double {
    /// store → key → (value, etag)
    stores: HashMap<String, HashMap<String, (Value, u64)>>,
    log: Vec<(String, String, Option<String>, Option<String>)>,
    next_etag: u64,
}

type Shared = Arc<Mutex<Double>>;

impl Double {
    fn bump(&mut self) -> u64 {
        self.next_etag += 1;
        self.next_etag
    }
}

fn decode(segment: &str) -> String {
    String::from_utf8(
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(segment)
            .unwrap(),
    )
    .unwrap()
}

async fn handle(State(state): State<Shared>, request: Request) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    assert!(request
        .uri()
        .query()
        .unwrap_or_default()
        .contains("api-version=v1"));
    let (auth, user, if_match) = {
        let header = |n: &str| {
            request
                .headers()
                .get(n)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        };
        (
            header("authorization"),
            header("x-ms-user-id"),
            header("if-match"),
        )
    };
    let body: Value =
        serde_json::from_slice(&request.into_body().collect().await.unwrap().to_bytes())
            .unwrap_or(Value::Null);
    apply(&state, method, path, auth, user, if_match, body)
}

fn apply(
    state: &Shared,
    method: Method,
    path: String,
    auth: Option<String>,
    user: Option<String>,
    if_match: Option<String>,
    body: Value,
) -> Response {
    let mut db = state.lock().unwrap();
    db.log.push((method.to_string(), path.clone(), auth, user));
    let parts: Vec<&str> = path.trim_start_matches("/storage/").split('/').collect();
    match (method, parts.as_slice()) {
        (Method::POST, ["state_stores"]) => {
            let name = body["name"].as_str().unwrap().to_string();
            if db.stores.contains_key(&name) {
                return StatusCode::CONFLICT.into_response();
            }
            db.stores.insert(name.clone(), HashMap::new());
            Json(json!({"name": name})).into_response()
        }
        (Method::GET, ["state_stores", store]) => match db.stores.contains_key(&decode(store)) {
            true => Json(json!({"name": decode(store)})).into_response(),
            false => StatusCode::NOT_FOUND.into_response(),
        },
        (Method::POST, ["state_stores", store, "items"]) => {
            let tag = db.bump();
            let items = db.stores.get_mut(&decode(store)).unwrap();
            let key = body["key"].as_str().unwrap().to_string();
            if items.contains_key(&key) {
                return StatusCode::CONFLICT.into_response();
            }
            items.insert(key.clone(), (body["value"].clone(), tag));
            Json(json!({"key": key, "etag": tag.to_string()})).into_response()
        }
        (Method::GET, ["state_stores", store, "items:keys"]) => {
            let keys: Vec<Value> = db.stores[&decode(store)]
                .keys()
                .map(|k| json!({"key": k}))
                .collect();
            Json(json!({"data": keys, "has_more": false})).into_response()
        }
        (method, ["state_stores", store, "items", key]) => {
            let tag = db.bump();
            let items = db.stores.get_mut(&decode(store)).unwrap();
            let key = decode(key);
            let current = items.get(&key).cloned();
            if let Some(expected) = &if_match {
                let ok = match (&current, expected.as_str()) {
                    (Some(_), "*") => true,
                    (Some((_, t)), e) => t.to_string() == e,
                    (None, _) => false,
                };
                if !ok {
                    return StatusCode::PRECONDITION_FAILED.into_response();
                }
            }
            match method {
                Method::GET => match current {
                    Some((value, t)) => {
                        Json(json!({"key": key, "value": value, "etag": t.to_string()}))
                            .into_response()
                    }
                    None => StatusCode::NOT_FOUND.into_response(),
                },
                Method::PUT => {
                    items.insert(key.clone(), (body["value"].clone(), tag));
                    Json(json!({"key": key, "etag": tag.to_string()})).into_response()
                }
                Method::DELETE => match items.remove(&key) {
                    Some(_) => Json(json!({"key": key, "deleted": true})).into_response(),
                    None => StatusCode::NOT_FOUND.into_response(),
                },
                _ => StatusCode::METHOD_NOT_ALLOWED.into_response(),
            }
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn double() -> (FoundryStorage, Shared) {
    let state: Shared = Arc::default();
    let app = Router::new().fallback(handle).with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let endpoint = FoundryStorageEndpoint::from_endpoint(&format!("http://{addr}/")).unwrap();
    let storage = FoundryStorage::new(endpoint, Some(Arc::new(StaticTokenCredential::new("tok"))));
    (storage, state)
}

#[test]
fn endpoints_and_store_names() {
    let e = FoundryStorageEndpoint::from_endpoint("https://x.services.ai.azure.com/api/projects/p")
        .unwrap();
    assert_eq!(
        e.storage_base_url,
        "https://x.services.ai.azure.com/api/projects/p/storage/"
    );
    let e = FoundryStorageEndpoint::from_endpoint("https://x/storage/").unwrap();
    assert_eq!(e.storage_base_url, "https://x/storage/");
    assert!(FoundryStorageEndpoint::from_endpoint("x").is_err());
    assert_eq!(
        e.build_url("state_stores", &[("after", "a b")]),
        "https://x/storage/state_stores?api-version=v1&after=a%20b"
    );
    let scope = FoundryRequestScope {
        session_id: "s".into(),
        user_id: Some("u".into()),
        call_id: None,
        is_hosted: true,
    };
    let name = agent_framework_foundry_hosting::scope::store_name(
        "checkpoints",
        Some(&scope),
        Some("ctx"),
    );
    assert!(name.starts_with(&format!("checkpoints/v2/{}/", scope.storage_key())));
    assert!(!name.contains("ctx"));
    assert_eq!(scope.storage_key().len(), 64);
}

#[cfg(feature = "session-store")]
#[tokio::test]
async fn invocation_sessions_round_trip_through_foundry_storage() {
    let (storage, state) = double().await;
    let router = InvocationsHostServer::agent(agent(Echo::default()))
        .with_session_store_provider(Arc::new(FoundrySessionStoreProvider::new(
            storage,
            Some("invocation_sessions".into()),
        )))
        .into_router();
    for (message, expected) in [
        ("one", "1 messages, last: one"),
        ("two", "3 messages, last: two"),
    ] {
        let mut request = post(
            "/invocations?agent_session_id=s1",
            json!({"message": message}),
        );
        request
            .headers_mut()
            .insert("x-agent-user-id", "user-7".parse().unwrap());
        let (status, reply) = json(&router, request).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(reply["response"], expected);
    }
    let db = state.lock().unwrap();
    // Keyed by the host's agent identity, the session id and its partitions.
    assert!(
        db.stores["invocation_sessions"].contains_key(r#"["foundry-invocations/echo","s1",{}]"#),
        "{:?}",
        db.stores["invocation_sessions"].keys().collect::<Vec<_>>()
    );
    assert!(db
        .log
        .iter()
        .all(|(_, _, auth, _)| auth.as_deref() == Some("Bearer tok")));
    assert!(db
        .log
        .iter()
        .filter(|(_, p, _, _)| p.contains("/items/"))
        .all(|(_, _, _, user)| user.as_deref() == Some("user-7")));
    // The first write creates (so a racing creator wins); the second is a
    // conditional replace of the loaded item.
    let count = |method: &str, suffix: &str| {
        db.log
            .iter()
            .filter(|(m, p, _, _)| m == method && p.contains(suffix))
            .count()
    };
    assert_eq!(count("POST", "/items"), 1);
    assert_eq!(count("PUT", "/items/"), 1);
}

#[tokio::test]
async fn checkpoints_and_approvals() {
    let (storage, state) = double().await;
    let context = FoundryRequestContext::default();
    let store =
        FoundryCheckpointStore::new(storage.clone(), context.clone(), "conv-1", None).unwrap();
    assert_eq!(store.store_name(), "checkpoints/conv-1");
    let checkpoint = |id: &str, workflow: &str, ts: u64| {
        serde_json::from_value::<WorkflowCheckpoint>(json!({
            "checkpoint_id": id,
            "workflow_id": workflow,
            "timestamp_millis": ts,
            "iteration_count": 0,
            "messages": [],
            "shared_state": {},
            "executor_states": {},
            "pending_requests": [],
        }))
    };
    let Ok(first) = checkpoint("c1", "w", 2) else {
        // The checkpoint wire shape is owned by core; build one through its
        // storage instead when the literal above drifts.
        panic!("checkpoint literal no longer matches WorkflowCheckpoint");
    };
    store.save(first).await.unwrap();
    store.save(checkpoint("c0", "w", 1).unwrap()).await.unwrap();
    store
        .save(checkpoint("x", "other", 3).unwrap())
        .await
        .unwrap();
    assert_eq!(store.load("c1").await.unwrap().unwrap().workflow_id, "w");
    assert!(store.load("nope").await.unwrap().is_none());
    let listed: Vec<String> = store
        .list(Some("w"))
        .await
        .unwrap()
        .into_iter()
        .map(|c| c.checkpoint_id)
        .collect();
    assert_eq!(listed, ["c0", "c1"], "oldest first");
    assert!(store.delete("c0").await.unwrap());
    assert!(!store.delete("c0").await.unwrap());

    let approvals = FoundryFunctionApprovalStore::new(storage, context, None);
    let request = Content::text("approve me");
    approvals
        .save_approval_request("a1", &request)
        .await
        .unwrap();
    assert!(
        approvals
            .save_approval_request("a1", &request)
            .await
            .is_err(),
        "ids are unique"
    );
    let loaded = approvals.load_approval_request("a1").await.unwrap();
    assert_eq!(
        serde_json::to_value(&loaded).unwrap(),
        serde_json::to_value(&request).unwrap()
    );
    assert!(approvals.load_approval_request("missing").await.is_err());
    assert!(state
        .lock()
        .unwrap()
        .stores
        .contains_key("function_approvals"));
}

#[cfg(feature = "session-store")]
#[tokio::test]
async fn hosted_stores_hide_identities_in_their_names() {
    let (storage, state) = double().await;
    let config = FoundryHostConfig {
        is_hosted: true,
        session_id: "sandbox".into(),
        ..FoundryHostConfig::local()
    };
    let router = InvocationsHostServer::agent(agent(Echo::default()))
        .with_config(config)
        .with_session_store_provider(Arc::new(FoundrySessionStoreProvider::new(storage, None)))
        .into_router();
    let mut request = post("/invocations", json!({"message": "hi"}));
    request
        .headers_mut()
        .insert("x-agent-user-id", "alice".parse().unwrap());
    request
        .headers_mut()
        .insert("x-agent-foundry-call-id", "call-1".parse().unwrap());
    let (status, reply) = json(&router, request).await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    let db = state.lock().unwrap();
    let names: Vec<&String> = db.stores.keys().collect();
    assert_eq!(names.len(), 1);
    assert!(names[0].starts_with("agent_sessions/v2/"));
    assert!(!names[0].contains("alice") && !names[0].contains("sandbox"));
}

#[cfg(feature = "session-store")]
#[tokio::test]
async fn a_stale_session_write_is_refused() {
    use agent_framework_core::session_store::{AgentSessionStore, AgentSessionStoreKey};
    use agent_framework_foundry_hosting::FoundryAgentSessionStore;
    let (storage, _) = double().await;
    let agent = agent(Echo::default());
    let key = AgentSessionStoreKey::new("s").unwrap();
    let ctx = FoundryRequestContext::default();
    let seed = FoundryAgentSessionStore::new(storage.clone(), ctx.clone(), None, None);
    seed.save_session(agent.as_ref(), &key, &agent.create_session())
        .await
        .unwrap();
    let a = FoundryAgentSessionStore::new(storage.clone(), ctx.clone(), None, None);
    let b = FoundryAgentSessionStore::new(storage, ctx, None, None);
    let sa = a.get_session(agent.as_ref(), &key).await.unwrap().unwrap();
    let sb = b.get_session(agent.as_ref(), &key).await.unwrap().unwrap();
    a.save_session(agent.as_ref(), &key, &sa).await.unwrap();
    let err = b.save_session(agent.as_ref(), &key, &sb).await.unwrap_err();
    assert!(
        err.to_string().contains("Another request advanced"),
        "{err}"
    );
    b.delete_session(agent.as_ref(), &key).await.unwrap();
    assert!(a.get_session(agent.as_ref(), &key).await.unwrap().is_none());
}

#[cfg(feature = "session-store")]
#[tokio::test]
async fn agent_history_sessions_survive_a_restart() {
    use agent_framework_foundry_hosting::{
        HistorySource, InMemoryResponseStore, ResponsesHostServer,
    };
    let (storage, _) = double().await;
    let responses: Arc<InMemoryResponseStore> = Arc::default();
    let echo = common::Echo::default();
    let build = || {
        ResponsesHostServer::agent(common::agent(echo.clone()))
            .with_history_source(HistorySource::Agent)
            .with_response_store(responses.clone())
            .with_session_store_provider(Arc::new(
                agent_framework_foundry_hosting::sessions::FoundrySessionStoreProvider::new(
                    storage.clone(),
                    None,
                ),
            ))
            .into_router()
    };
    let host = build();
    let (status, first) =
        common::json(&host, common::post("/responses", json!({"input": "one"}))).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    // A fresh host (a restart) continues from the stored agent session.
    drop(host);
    let restarted = build();
    let (status, second) = common::json(
        &restarted,
        common::post(
            "/responses",
            json!({"input": "two", "previous_response_id": first["id"]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    let seen = echo.seen.lock().unwrap();
    assert_eq!(
        seen.last().unwrap().len(),
        3,
        "prior user + assistant + new input"
    );
}

#[cfg(feature = "session-store")]
#[tokio::test]
async fn hosted_conversation_turns_save_their_sessions() {
    use agent_framework_foundry_hosting::{HistorySource, ResponsesHostServer};
    for history in [HistorySource::Agent, HistorySource::Service] {
        let (storage, _) = double().await;
        let echo = Echo::default();
        let router = ResponsesHostServer::agent(agent(echo.clone()))
            .with_history_source(history)
            .with_config(FoundryHostConfig {
                is_hosted: true,
                session_id: "sandbox".into(),
                ..FoundryHostConfig::local()
            })
            .with_session_store_provider(Arc::new(FoundrySessionStoreProvider::new(storage, None)))
            .into_router();
        let mut ids = Vec::new();
        for input in ["one", "two"] {
            let mut request = post(
                "/responses",
                json!({"input": input, "conversation": "conv_h"}),
            );
            request
                .headers_mut()
                .insert("x-agent-user-id", "alice".parse().unwrap());
            request
                .headers_mut()
                .insert("x-agent-foundry-call-id", "call-1".parse().unwrap());
            request
                .headers_mut()
                .insert("x-agent-foundry-call-id", "call-1".parse().unwrap());
            let (status, reply) = json(&router, request).await;
            assert_eq!(status, StatusCode::OK, "{history:?}: {reply}");
            ids.push(reply["id"].clone());
        }
        // The first turn's own session record continues it.
        let mut request = post(
            "/responses",
            json!({"input": "three", "previous_response_id": ids[0]}),
        );
        request
            .headers_mut()
            .insert("x-agent-user-id", "alice".parse().unwrap());
        request
            .headers_mut()
            .insert("x-agent-foundry-call-id", "call-1".parse().unwrap());
        let (status, reply) = json(&router, request).await;
        assert_eq!(status, StatusCode::OK, "{history:?}: {reply}");
    }
}

/// An agent of its own (a distinct id) for the session-store tests.
#[cfg(feature = "session-store")]
fn named_agent(name: &str) -> Arc<dyn agent_framework_core::agent::SupportsAgentRun> {
    use agent_framework_core::prelude::Agent;
    Arc::new(Agent::builder(Echo::default()).name(name).build())
}

#[cfg(feature = "session-store")]
#[tokio::test]
async fn partitioned_and_look_alike_session_keys_stay_apart() {
    use agent_framework_core::session_store::{AgentSessionStore, AgentSessionStoreKey};
    use agent_framework_foundry_hosting::FoundryAgentSessionStore;
    let (storage, _) = double().await;
    let agent = named_agent("one");
    let store =
        FoundryAgentSessionStore::new(storage, FoundryRequestContext::default(), None, None);
    let partitioned = AgentSessionStoreKey::new("s")
        .unwrap()
        .with_partition("tenant", "a")
        .unwrap();
    // An unpartitioned id spelled like the partitioned key's old framing.
    let look_alike = AgentSessionStoreKey::new(r#"s#{"tenant":"a"}"#).unwrap();
    let session = agent.create_session();
    session.state.insert("owner", json!("partitioned"));
    store
        .save_session(agent.as_ref(), &partitioned, &session)
        .await
        .unwrap();
    assert!(
        store
            .get_session(agent.as_ref(), &look_alike)
            .await
            .unwrap()
            .is_none(),
        "the look-alike key does not read the partitioned session"
    );
    store
        .delete_session(agent.as_ref(), &look_alike)
        .await
        .unwrap();
    let kept = store
        .get_session(agent.as_ref(), &partitioned)
        .await
        .unwrap()
        .expect("deleting the look-alike key keeps the partitioned session");
    assert_eq!(kept.state.get("owner"), Some(json!("partitioned")));
}

#[cfg(feature = "session-store")]
#[tokio::test]
async fn sessions_are_scoped_to_their_agent() {
    use agent_framework_core::session_store::{AgentSessionStore, AgentSessionStoreKey};
    use agent_framework_foundry_hosting::FoundryAgentSessionStore;
    let (storage, _) = double().await;
    let (one, two) = (named_agent("one"), named_agent("two"));
    assert_ne!(one.id(), two.id());
    let store =
        FoundryAgentSessionStore::new(storage, FoundryRequestContext::default(), None, None);
    let key = AgentSessionStoreKey::new("shared").unwrap();
    let session = one.create_session();
    session.state.insert("owner", json!("one"));
    store
        .save_session(one.as_ref(), &key, &session)
        .await
        .unwrap();
    assert!(
        store
            .get_session(two.as_ref(), &key)
            .await
            .unwrap()
            .is_none(),
        "another agent does not read it"
    );
    // Another agent's write and delete under the same key leave it alone.
    store
        .save_session(two.as_ref(), &key, &two.create_session())
        .await
        .unwrap();
    store.delete_session(two.as_ref(), &key).await.unwrap();
    let kept = store
        .get_session(one.as_ref(), &key)
        .await
        .unwrap()
        .expect("the owner's session survives");
    assert_eq!(kept.state.get("owner"), Some(json!("one")));
}
