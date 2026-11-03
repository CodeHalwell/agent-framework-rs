//! `FoundryToolbox` against an in-process MCP server double.

use std::sync::{Arc, Mutex};

use agent_framework_azure::StaticTokenCredential;
use agent_framework_foundry_hosting::{FoundryRequestContext, FoundryToolbox};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

type Seen = Arc<Mutex<Vec<HeaderMap>>>;

async fn mcp(State(seen): State<Seen>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    seen.lock().unwrap().push(headers);
    let Some(id) = body.get("id").cloned() else {
        return StatusCode::ACCEPTED.into_response();
    };
    let reply = match body["method"].as_str() {
        Some("initialize") => json!({"jsonrpc": "2.0", "id": id, "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "toolbox", "version": "1"},
        }}),
        Some("resources/read") => {
            let text = match body["params"]["uri"].as_str() {
                Some("skill://index.json") => json!({"skills": [
                    {"name": "unit-converter", "type": "skill-md", "description": "Convert units.",
                     "url": "skill://unit-converter/SKILL.md"},
                    {"name": "packed", "type": "archive", "url": "skill://packed.zip"},
                ]})
                .to_string(),
                Some("skill://unit-converter/SKILL.md") => {
                    "---\nname: unit-converter\ndescription: Converts units.\n---\nMultiply by 2.54.".into()
                }
                _ => return Json(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32002, "message": "not found"}})).into_response(),
            };
            json!({"jsonrpc": "2.0", "id": id, "result": {"contents": [{"uri": body["params"]["uri"], "text": text}]}})
        }
        Some("tools/list") => json!({"jsonrpc": "2.0", "id": id, "result": {"tools": [
            {"name": "search", "description": "Search.", "inputSchema": {"type": "object"}},
        ]}}),
        _ => {
            json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "Method not found"}})
        }
    };
    Json(reply).into_response()
}

#[tokio::test]
async fn toolbox_requests_carry_auth_features_and_call_id() {
    let seen: Seen = Arc::default();
    let app = Router::new()
        .route(
            "/toolboxes/tb/mcp",
            post(mcp).delete(|| async { StatusCode::OK }),
        )
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Without a URL, TOOLBOX_ENDPOINT, or project endpoint + TOOLBOX_NAME
    // there is nothing to connect to.
    let missing = FoundryToolbox::new(Arc::new(StaticTokenCredential::new("tok")))
        .build_with(|name| (name == "FOUNDRY_PROJECT_ENDPOINT").then(|| format!("http://{addr}")))
        .await;
    assert!(missing.is_err());

    let tool = FoundryToolbox::new(Arc::new(StaticTokenCredential::new("tok")))
        .url(format!("http://{addr}/toolboxes/tb/mcp"))
        .request_context(FoundryRequestContext {
            call_id: Some("call-9".into()),
            ..Default::default()
        })
        .build_with(|name| (name == "FOUNDRY_AGENT_TOOLSET_FEATURES").then(|| "Extra=1".into()))
        .await
        .unwrap();
    let definitions = tool.tool_definitions().await.unwrap();
    assert_eq!(definitions.len(), 1);
    assert!(definitions[0].name.contains("search"));
    let seen = seen.lock().unwrap();
    assert!(!seen.is_empty());
    for headers in seen.iter() {
        assert_eq!(headers["authorization"], "Bearer tok");
        assert_eq!(headers["foundry-features"], "Toolboxes=V1Preview,Extra=1");
        assert_eq!(headers["x-agent-foundry-call-id"], "call-9");
    }
}

/// Returns `tok-1`, `tok-2`, ... : a credential whose token rotates.
struct Rotating(std::sync::atomic::AtomicUsize);

#[async_trait::async_trait]
impl agent_framework_azure::TokenCredential for Rotating {
    async fn get_token(&self) -> agent_framework_core::error::Result<String> {
        let n = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        Ok(format!("tok-{n}"))
    }
}

#[tokio::test]
async fn the_token_is_fetched_for_every_request() {
    let seen: Seen = Arc::default();
    let app = Router::new()
        .route(
            "/toolboxes/tb/mcp",
            post(mcp).delete(|| async { StatusCode::OK }),
        )
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let tool = FoundryToolbox::new(Arc::new(Rotating(Default::default())))
        .url(format!("http://{addr}/toolboxes/tb/mcp"))
        .build_with(|_| None)
        .await
        .unwrap();
    tool.tool_definitions().await.unwrap();
    let tokens: Vec<String> = seen
        .lock()
        .unwrap()
        .iter()
        .map(|h| h["authorization"].to_str().unwrap().to_string())
        .collect();
    assert!(tokens.len() >= 2, "{tokens:?}");
    let mut distinct = tokens.clone();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        tokens.len(),
        "a fresh token per request: {tokens:?}"
    );
}

#[cfg(feature = "skills")]
#[tokio::test]
async fn the_toolbox_serves_its_skills() {
    use agent_framework_core::memory::ContextProvider;
    let seen: Seen = Arc::default();
    let app = Router::new()
        .route(
            "/toolboxes/tb/mcp",
            post(mcp).delete(|| async { StatusCode::OK }),
        )
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let tool = FoundryToolbox::new(Arc::new(StaticTokenCredential::new("tok")))
        .url(format!("http://{addr}/toolboxes/tb/mcp"))
        .build_with(|_| None)
        .await
        .unwrap();
    let skills = tool.skills().await.unwrap();
    assert_eq!(skills.len(), 1, "the archive entry is skipped");
    assert_eq!(skills[0].name, "unit-converter");
    assert_eq!(skills[0].description, "Convert units.");
    assert_eq!(skills[0].instructions.trim(), "Multiply by 2.54.");
    let provider = FoundryToolbox::new(Arc::new(StaticTokenCredential::new("tok")))
        .url(format!("http://{addr}/toolboxes/tb/mcp"))
        .skills_provider_with(|_| None)
        .await
        .unwrap();
    let _: &dyn ContextProvider = &provider;
}

/// A toolbox serving an `archive` skill, whose index can grow, whose archive
/// reads can fail, and which counts index reads.
#[cfg(feature = "skills")]
#[derive(Default)]
struct SkillServer {
    index_reads: std::sync::atomic::AtomicUsize,
    grown: std::sync::atomic::AtomicBool,
    failing: std::sync::atomic::AtomicBool,
}

#[cfg(feature = "skills")]
fn packed_zip() -> Vec<u8> {
    use std::io::Write;
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (name, content) in [
        (
            "packed/SKILL.md",
            "---\nname: packed\ndescription: A packed skill.\n---\nRead the table.",
        ),
        ("packed/table.csv", "inch,cm\n1,2.54"),
    ] {
        writer.start_file(name, options).unwrap();
        writer.write_all(content.as_bytes()).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

#[cfg(feature = "skills")]
async fn skill_mcp(State(server): State<Arc<SkillServer>>, Json(body): Json<Value>) -> Response {
    use base64::Engine as _;
    use std::sync::atomic::Ordering::SeqCst;
    let Some(id) = body.get("id").cloned() else {
        return StatusCode::ACCEPTED.into_response();
    };
    let uri = body["params"]["uri"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let content = match (body["method"].as_str(), uri.as_str()) {
        (Some("initialize"), _) => {
            return Json(json!({"jsonrpc": "2.0", "id": id, "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"resources": {}},
                "serverInfo": {"name": "toolbox", "version": "1"},
            }}))
            .into_response()
        }
        (Some("resources/read"), "skill://index.json") => {
            server.index_reads.fetch_add(1, SeqCst);
            let mut skills = vec![json!({"name": "packed", "type": "archive", "url": "skill://packed.zip"})];
            if server.grown.load(SeqCst) {
                skills.push(json!({"name": "late", "type": "skill-md", "url": "skill://late/SKILL.md"}));
            }
            json!({"uri": uri, "text": json!({"skills": skills}).to_string()})
        }
        (Some("resources/read"), "skill://packed.zip") if server.failing.load(SeqCst) => {
            return Json(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32603, "message": "down"}}))
                .into_response()
        }
        (Some("resources/read"), "skill://packed.zip") => json!({
            "uri": uri,
            "mimeType": "application/zip",
            "blob": base64::engine::general_purpose::STANDARD.encode(packed_zip()),
        }),
        (Some("resources/read"), "skill://late/SKILL.md") => json!({
            "uri": uri,
            "text": "---\nname: late\ndescription: A late skill.\n---\nArrived later.",
        }),
        _ => {
            return Json(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "Method not found"}}))
                .into_response()
        }
    };
    Json(json!({"jsonrpc": "2.0", "id": id, "result": {"contents": [content]}})).into_response()
}

#[cfg(feature = "skills")]
async fn skill_toolbox() -> (Arc<SkillServer>, FoundryToolbox) {
    let server = Arc::new(SkillServer::default());
    let app = Router::new()
        .route(
            "/toolboxes/tb/mcp",
            post(skill_mcp).delete(|| async { StatusCode::OK }),
        )
        .with_state(server.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let toolbox = FoundryToolbox::new(Arc::new(StaticTokenCredential::new("tok")))
        .url(format!("http://{addr}/toolboxes/tb/mcp"));
    (server, toolbox)
}

#[cfg(feature = "skills")]
async fn catalog(provider: &dyn agent_framework_core::memory::ContextProvider) -> String {
    let mut ctx = agent_framework_core::memory::SessionContext::new(Vec::new());
    provider.before_run(&mut ctx).await.unwrap();
    ctx.instructions.unwrap_or_default()
}

#[cfg(feature = "skills")]
#[tokio::test]
async fn archive_skills_load_with_the_toolboxs_archive_options() {
    use agent_framework_foundry_hosting::toolbox::ToolboxSkillsOptions;
    use agent_framework_mcp::skills::ArchiveOptions;
    let (_, toolbox) = skill_toolbox().await;
    let tool = toolbox.build_with(|_| None).await.unwrap();
    let skills = tool.skills().await.unwrap();
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].name, "packed");
    assert_eq!(skills[0].instructions.trim(), "Read the table.");
    assert_eq!(skills[0].resources["table.csv"], "inch,cm\n1,2.54");
    let provider = toolbox.skills_provider_with(|_| None).await.unwrap();
    assert!(catalog(&provider)
        .await
        .contains("- packed: A packed skill."));
    // The toolbox's archive options apply: one file is too many.
    let strict = toolbox.clone().skills_options(ToolboxSkillsOptions {
        archive: ArchiveOptions {
            max_file_count: 1,
            ..Default::default()
        },
        ..Default::default()
    });
    let provider = strict.skills_provider_with(|_| None).await.unwrap();
    assert!(!catalog(&provider).await.contains("packed"));
}

#[cfg(feature = "skills")]
#[tokio::test]
async fn the_refreshing_skills_provider_caches_and_refreshes() {
    use agent_framework_foundry_hosting::toolbox::ToolboxSkillsOptions;
    use std::sync::atomic::Ordering::SeqCst;
    let (server, toolbox) = skill_toolbox().await;
    let provider = toolbox
        .skills_options(ToolboxSkillsOptions {
            refresh_interval: Some(std::time::Duration::from_millis(100)),
            ..Default::default()
        })
        .refreshing_skills_provider_with(|_| None)
        .await
        .unwrap();
    // Discovery is lazy, then cached.
    assert_eq!(server.index_reads.load(SeqCst), 0);
    assert!(catalog(&provider).await.contains("packed"));
    server.grown.store(true, SeqCst);
    assert!(!catalog(&provider).await.contains("late"), "still cached");
    assert_eq!(server.index_reads.load(SeqCst), 1);
    // Stale after the interval: re-read.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    let text = catalog(&provider).await;
    assert!(
        text.contains("- late: A late skill.") && text.contains("packed"),
        "{text}"
    );
    // A failed refresh keeps the previous list.
    server.failing.store(true, SeqCst);
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    let text = catalog(&provider).await;
    assert!(text.contains("late") && text.contains("packed"), "{text}");
    assert_eq!(server.index_reads.load(SeqCst), 3);

    // Without caching every run re-reads; an initial failure fails the run.
    let (server, toolbox) = skill_toolbox().await;
    let provider = toolbox
        .skills_options(ToolboxSkillsOptions {
            disable_caching: true,
            ..Default::default()
        })
        .refreshing_skills_provider_with(|_| None)
        .await
        .unwrap();
    catalog(&provider).await;
    catalog(&provider).await;
    assert_eq!(server.index_reads.load(SeqCst), 2);
    server.failing.store(true, SeqCst);
    let mut ctx = agent_framework_core::memory::SessionContext::new(Vec::new());
    assert!(
        agent_framework_core::memory::ContextProvider::before_run(&provider, &mut ctx)
            .await
            .is_err()
    );
}
