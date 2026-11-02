//! FIDES for MCP tools: annotation-derived declarations and server
//! `_meta.ifc` result labels, against a hermetic python3 stdio server.

#![cfg(feature = "fides")]

use std::io::Write as _;
use std::sync::Arc;

use agent_framework_core::middleware::{FunctionInvocationContext, MiddlewarePipeline};
use agent_framework_core::security::{
    ConfidentialityLabel, ContentLabel, IntegrityLabel, LabelTrackingFunctionMiddleware,
    ToolSecurity, ToolSecurityRegistry, RESULT_LABEL_KEY,
};
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::{ToolDefinition, ToolSource};
use agent_framework_mcp::security::{
    label_from_mcp_meta, secure_tool_definitions, stamp_mcp_result, McpSecurityPolicy,
    SecureMcpToolProxy,
};
use agent_framework_mcp::McpStdioTool;
use serde_json::{json, Value};

/// Tools: `read_page` (openWorldHint, plain text, no label), `read_db` (an
/// object labeled trusted/private), `labeled_text` (text labeled
/// untrusted/public), `bad_meta` (a partial label), `local_only`
/// (openWorldHint false).
const SERVER_PY: &str = r#"
import sys, json

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

TOOLS = [
    {"name": "read_page", "inputSchema": {"type": "object"}, "annotations": {"openWorldHint": True}},
    {"name": "read_db", "inputSchema": {"type": "object"}},
    {"name": "labeled_text", "inputSchema": {"type": "object"}},
    {"name": "bad_meta", "inputSchema": {"type": "object"}},
    {"name": "local_only", "inputSchema": {"type": "object"}, "annotations": {"openWorldHint": False, "readOnlyHint": True}},
    {"name": "private_error", "inputSchema": {"type": "object"}},
    {"name": "plain_error", "inputSchema": {"type": "object"}},
]

RESULTS = {
    "read_page": {"content": [{"type": "text", "text": "page text"}]},
    "read_db": {"content": [{"type": "text", "text": "{\"rows\": 1}"}],
                "_meta": {"ifc": {"integrity": "trusted", "confidentiality": "private"}}},
    "labeled_text": {"content": [{"type": "text", "text": "hello"}],
                     "_meta": {"ifc": {"integrity": "untrusted", "confidentiality": "public"}}},
    "bad_meta": {"content": [{"type": "text", "text": "x"}], "_meta": {"ifc": {"integrity": "trusted"}}},
    "local_only": {"content": [{"type": "text", "text": "local"}]},
    "private_error": {"isError": True, "content": [{"type": "text", "text": "secret row 42 not found"}],
                      "_meta": {"ifc": {"integrity": "trusted", "confidentiality": "private"}}},
    "plain_error": {"isError": True, "content": [{"type": "text", "text": "bad input"}]},
}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method, msg_id = msg.get("method"), msg.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
            "serverInfo": {"name": "ifc", "version": "1"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        name = msg["params"]["name"]
        send({"jsonrpc": "2.0", "id": msg_id, "result": RESULTS[name]})
    elif msg_id is not None:
        send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "Method not found"}})
"#;

fn server() -> McpStdioTool {
    let path = std::env::temp_dir().join(format!(
        "af_mcp_ifc_server_{}_{}.py",
        std::process::id(),
        uuid_like()
    ));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(SERVER_PY.as_bytes()).unwrap();
    McpStdioTool::new("ifc", "python3").args([path.to_string_lossy().to_string()])
}

fn uuid_like() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

fn definition<'a>(defs: &'a [ToolDefinition], name: &str) -> &'a ToolDefinition {
    defs.iter().find(|d| d.name == name).unwrap()
}

/// Invoke `name` under the label-tracking middleware and return the
/// invocation (its result and `RESULT_LABEL_KEY`).
async fn tracked_call(
    registry: &ToolSecurityRegistry,
    defs: &[ToolDefinition],
    name: &str,
    auto_hide: bool,
) -> FunctionInvocationContext {
    let tracker = Arc::new(
        LabelTrackingFunctionMiddleware::new()
            .tool_registry(registry.clone())
            .auto_hide_untrusted(auto_hide),
    );
    let pipeline: MiddlewarePipeline<FunctionInvocationContext> =
        MiddlewarePipeline::new(vec![tracker]);
    let tool = definition(defs, name).executor.clone().unwrap();
    let session = AgentSession::new();
    pipeline
        .execute(
            FunctionInvocationContext::new(name, json!({})).with_session(Some(session)),
            Box::new(move |mut ctx: FunctionInvocationContext| {
                Box::pin(async move {
                    ctx.result = Some(tool.invoke(ctx.arguments.clone()).await?);
                    Ok(ctx)
                })
            }),
        )
        .await
        .unwrap()
}

fn result_label(ctx: &FunctionInvocationContext) -> ContentLabel {
    ContentLabel::from_value(&ctx.metadata[RESULT_LABEL_KEY]).unwrap()
}

#[tokio::test]
async fn annotations_declare_restriction_only_labels() {
    let registry = ToolSecurityRegistry::new();
    registry.declare(
        "read_db",
        ToolSecurity::new().confidentiality(ConfidentialityLabel::Private),
    );
    let defs = secure_tool_definitions(
        server(),
        &registry,
        McpSecurityPolicy::new()
            .default_integrity(IntegrityLabel::Trusted)
            .annotation_override("local_only", IntegrityLabel::Trusted, None),
    )
    .await
    .unwrap();
    assert_eq!(defs.len(), 7);

    // openWorldHint makes results untrusted even with a trusted default.
    let page = registry.get("read_page").unwrap();
    assert_eq!(page.source_integrity, Some(IntegrityLabel::Untrusted));
    assert_eq!(
        page.max_allowed_confidentiality,
        Some(ConfidentialityLabel::Public)
    );
    assert!(!page.accepts_untrusted, "annotations never grant trust");

    // No hint: the local default; an existing local declaration is kept.
    let db = registry.get("read_db").unwrap();
    assert_eq!(db.source_integrity, Some(IntegrityLabel::Trusted));
    assert_eq!(db.confidentiality, Some(ConfidentialityLabel::Private));

    // A local override (by remote name) replaces the derived labels.
    let local = registry.get("local_only").unwrap();
    assert_eq!(local.source_integrity, Some(IntegrityLabel::Trusted));
    assert_eq!(local.max_allowed_confidentiality, None);

    // Without sinks marked, no confidentiality cap.
    let unmarked = ToolSecurityRegistry::new();
    secure_tool_definitions(
        server(),
        &unmarked,
        McpSecurityPolicy::new().mark_write_tools_as_sinks(false),
    )
    .await
    .unwrap();
    assert_eq!(
        unmarked
            .get("read_page")
            .unwrap()
            .max_allowed_confidentiality,
        None
    );
}

#[tokio::test]
async fn server_labels_only_restrict_by_default() {
    let registry = ToolSecurityRegistry::new();
    let proxy = SecureMcpToolProxy::new(server())
        .registry(registry.clone())
        .policy(McpSecurityPolicy::new().default_integrity(IntegrityLabel::Trusted));
    let defs = proxy.resolve_tools().await.unwrap();

    // A trusted/private server label on a trusted local tool: private.
    let ctx = tracked_call(&registry, &defs, "read_db", true).await;
    let label = result_label(&ctx);
    assert_eq!(label.integrity, IntegrityLabel::Trusted);
    assert_eq!(label.confidentiality, ConfidentialityLabel::Private);
    assert_eq!(ctx.result.as_ref().unwrap()["rows"], 1);

    // An untrusted server label taints a trusted tool; its text result is
    // wrapped to carry the label, then hidden by the tracker.
    let ctx = tracked_call(&registry, &defs, "labeled_text", false).await;
    assert_eq!(result_label(&ctx).integrity, IntegrityLabel::Untrusted);
    assert_eq!(ctx.result.as_ref().unwrap()["result"], "hello");
    let ctx = tracked_call(&registry, &defs, "labeled_text", true).await;
    assert_eq!(ctx.result.as_ref().unwrap()["type"], "variable_reference");

    // Partial metadata and no metadata: the local declaration.
    let ctx = tracked_call(&registry, &defs, "bad_meta", true).await;
    assert_eq!(result_label(&ctx).integrity, IntegrityLabel::Trusted);
    assert_eq!(ctx.result, Some(json!("x")));
    let ctx = tracked_call(&registry, &defs, "read_page", false).await;
    assert_eq!(result_label(&ctx).integrity, IntegrityLabel::Untrusted);
    assert_eq!(ctx.result, Some(json!("page text")));
    proxy.close().await.unwrap();
}

#[tokio::test]
async fn a_server_label_can_only_relax_when_trusted_locally() {
    // The local default is untrusted; the server says read_db is trusted.
    let restrictive = ToolSecurityRegistry::new();
    let defs = secure_tool_definitions(server(), &restrictive, McpSecurityPolicy::new())
        .await
        .unwrap();
    let ctx = tracked_call(&restrictive, &defs, "read_db", false).await;
    assert_eq!(result_label(&ctx).integrity, IntegrityLabel::Untrusted);

    let trusting = ToolSecurityRegistry::new();
    let defs = secure_tool_definitions(
        server(),
        &trusting,
        McpSecurityPolicy::new().trust_server_ifc(true),
    )
    .await
    .unwrap();
    let ctx = tracked_call(&trusting, &defs, "read_db", true).await;
    let label = result_label(&ctx);
    assert_eq!(label.integrity, IntegrityLabel::Trusted, "authoritative");
    assert_eq!(label.confidentiality, ConfidentialityLabel::Private);
    assert_eq!(
        ctx.result.as_ref().unwrap(),
        &json!({"rows": 1}),
        "not wrapped"
    );
}

#[test]
fn meta_parsing_and_stamping() {
    let label = |i: &str, c: &str| json!({"ifc": {"integrity": i, "confidentiality": c}});
    assert!(label_from_mcp_meta(Some(&label("trusted", "private"))).is_some());
    for bad in [
        json!({"ifc": {"integrity": "trusted"}}),
        label("trusted", "secret"),
        label("trusted", "user_identity"),
        json!({"ifc": "trusted"}),
        json!({}),
    ] {
        assert!(label_from_mcp_meta(Some(&bad)).is_none(), "{bad}");
    }
    assert!(label_from_mcp_meta(None).is_none());

    let local = ContentLabel::trusted();
    let server = ContentLabel::untrusted();
    assert_eq!(
        stamp_mcp_result(json!("x"), &local, None, false),
        json!("x")
    );
    let wrapped = stamp_mcp_result(json!([1]), &local, Some(server.clone()), false);
    assert_eq!(wrapped["result"], json!([1]));
    assert_eq!(wrapped["security_label"]["integrity"], "untrusted");

    // A label already embedded by the server is kept in the combination.
    let embedded =
        json!({"a": 1, "security_label": {"integrity": "trusted", "confidentiality": "private"}});
    let stamped = stamp_mcp_result(embedded, &local, Some(server), false);
    let combined: Value = stamped["security_label"].clone();
    assert_eq!(combined["integrity"], "untrusted");
    assert_eq!(combined["confidentiality"], "private");

    // Trusted without a tracking middleware running: left as is.
    assert_eq!(
        stamp_mcp_result(json!("x"), &local, Some(ContentLabel::untrusted()), true),
        json!("x")
    );
}

#[tokio::test]
async fn server_labeled_error_details_are_withheld() {
    // Locally trusted/public, so core would pass an error's text through;
    // the server's private label on the error must still withhold it.
    for trust_server_ifc in [false, true] {
        let registry = ToolSecurityRegistry::new();
        let defs = secure_tool_definitions(
            server(),
            &registry,
            McpSecurityPolicy::new()
                .default_integrity(IntegrityLabel::Trusted)
                .trust_server_ifc(trust_server_ifc),
        )
        .await
        .unwrap();
        let call = |name: &str| definition(&defs, name).executor.clone().unwrap();
        let err = call("private_error")
            .invoke(json!({}))
            .await
            .unwrap_err()
            .to_string();
        assert!(!err.contains("secret"), "{err}");
        assert!(err.contains("details withheld"), "{err}");

        // An unlabeled error keeps its details.
        let err = call("plain_error")
            .invoke(json!({}))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("bad input"), "{err}");
    }
}
