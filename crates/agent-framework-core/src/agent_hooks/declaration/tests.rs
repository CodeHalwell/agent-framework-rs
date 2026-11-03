//! Declaration loader tests, ported from the agent-hooks SDK's
//! `declaration` unit tests and `tests/declaration*.rs`, adapted to this
//! host's surface.

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use super::*;
use crate::agent::SupportsAgentRun;
use crate::agent_hooks::protocol::{
    host_error, interceptor_fn, InterceptionContextBuilder, InterceptionRecord, Verdict,
};

fn allow() -> Arc<dyn Interceptor> {
    Arc::new(interceptor_fn(|_| async { Ok(Verdict::allow()) }))
}

/// The smallest document this host accepts.
fn minimal() -> Value {
    json!({
        "declaration": DECLARATION_VERSION,
        "configuration": {"posture": {"tool_seam_host_error": "terminate"}},
        "bindings": [{"id": "guard", "kind": INTERCEPTOR_KIND}]
    })
}

fn registry() -> HostRegistry {
    HostRegistry::new()
        .interceptors([("guard", allow())])
        .unwrap()
}

fn load(v: Value) -> std::result::Result<ResolvedDeclaration, DeclarationError> {
    registry().resolve(&HostDeclaration::from_value(v)?)
}

fn class_of(v: Value) -> (DeclarationErrorClass, Vec<String>) {
    let e = load(v).expect_err("refused");
    (e.class, e.findings.into_iter().map(|f| f.pointer).collect())
}

fn with(mut v: Value, pointer: &str, value: Value) -> Value {
    *v.pointer_mut(pointer).expect("pointer exists") = value;
    v
}

#[test]
fn a_minimal_document_resolves_to_the_spec_defaults() {
    let r = load(minimal()).unwrap();
    let c = &r.configuration;
    assert_eq!(c.mode, EnforcementMode::Enforce);
    assert_eq!(c.composition, Composition::default());
    assert_eq!(c.identity_provider.as_deref(), Some(JCS_SHA256));
    assert_eq!(
        (c.interceptor_ms, c.approval_resolver_ms),
        (Some(5000), Some(5000))
    );
    assert_eq!(c.tool_seam_host_error, ToolSeamPosture::Terminate);
    assert_eq!(r.spec, SPEC_VERSION);
    assert_eq!(r.surface, {
        let s = HostSurface::agent_framework();
        ResolvedSurface {
            interception_points: s.interception_points,
            capabilities: s.capabilities,
            profiles: s.profiles,
            buffered_output: true,
            exposure_bound: None,
            declaration_versions: s.declaration_versions,
        }
    });
    assert_eq!(
        r.bindings[0].at.len(),
        8,
        "at defaults to the whole surface"
    );
    assert_eq!(r.bindings[0].timeout_ms, Some(5000));
    assert_eq!(r.bindings[0].config, json!({}));
}

#[test]
fn version_and_spec_checks_run_first() {
    let mut v = minimal();
    v.as_object_mut().unwrap().remove("declaration");
    v["unknown"] = json!(1);
    let e = load(v).unwrap_err();
    assert_eq!(e.class, DeclarationErrorClass::VersionUnsupported);
    assert_eq!(e.accepted, [DECLARATION_VERSION]);
    assert_eq!(
        class_of(with(
            minimal(),
            "/declaration",
            json!("agent-hooks-declaration/2.0")
        ))
        .0,
        DeclarationErrorClass::VersionUnsupported
    );
    for spec in [
        "agent-hooks/1.0",
        "agent-hooks/0.9",
        "agent-hooks/x",
        "nope",
    ] {
        let mut v = minimal();
        v["spec"] = json!(spec);
        assert_eq!(
            class_of(v).0,
            DeclarationErrorClass::SpecUnsupported,
            "{spec}"
        );
    }
    let mut v = minimal();
    v["spec"] = json!("agent-hooks/0.0");
    assert!(load(v).is_ok(), "an older minor is accepted");
}

#[test]
fn unknown_members_are_refused_at_every_closed_level_before_types() {
    let mut v = minimal();
    v["extra"] = json!(1);
    v["configuration"]["timeouts"] = json!({"interceptor_ms": "slow", "other": 1});
    v["bindings"][0]["when"] = json!("x");
    v["surface"] = json!({"profiles": {"parallel/whatever": {}}});
    let (class, pointers) = class_of(v);
    assert_eq!(
        class,
        DeclarationErrorClass::UnknownField,
        "unknown wins over invalid"
    );
    for p in [
        "/extra",
        "/configuration/timeouts/other",
        "/bindings/0/when",
        "/surface/profiles/parallel~1whatever",
    ] {
        assert!(pointers.iter().any(|x| x == p), "{p} in {pointers:?}");
    }
}

#[test]
fn invalid_fields_name_their_pointers() {
    let mut v = minimal();
    v["id"] = json!("Not Valid");
    v["configuration"]["mode"] = json!("audit");
    v["configuration"]["timeouts"] = json!({"interceptor_ms": 0});
    v["configuration"]["identity_provider"] = json!("jcs-custom");
    v["bindings"][0]["kind"] = json!("nodot");
    v["bindings"][0]["at"] = json!(["input", "input"]);
    let (class, pointers) = class_of(v);
    assert_eq!(class, DeclarationErrorClass::InvalidField);
    for p in [
        "/id",
        "/configuration/mode",
        "/configuration/timeouts/interceptor_ms",
        "/configuration/identity_provider",
        "/bindings/0/kind",
        "/bindings/0/at/1",
    ] {
        assert!(pointers.iter().any(|x| x == p), "{p} in {pointers:?}");
    }
    let mut v = minimal();
    v.as_object_mut().unwrap().remove("bindings");
    assert_eq!(class_of(v).1, ["/bindings"]);
}

#[test]
fn integers_beyond_2_53_are_refused_anywhere_even_in_config() {
    let text = r#"{"declaration":"agent-hooks-declaration/1.0",
        "configuration":{"posture":{"tool_seam_host_error":"terminate"}},
        "bindings":[{"id":"guard","kind":"agent_framework.interceptor",
                     "config":{"n":18446744073709551616}}]}"#;
    let e = HostDeclaration::from_json(text).unwrap_err();
    assert_eq!(e.class, DeclarationErrorClass::InvalidField);
    assert_eq!(e.findings[0].pointer, "/bindings/0/config/n");
    let at_bound = with(
        minimal(),
        "/configuration",
        json!({"posture": {"tool_seam_host_error": "terminate"}, "records": {"max_buffered": 9_007_199_254_740_991_u64}}),
    );
    assert!(load(at_bound).is_ok());
}

#[test]
fn parse_bounds_duplicates_and_depth() {
    let dup = r#"{"declaration":"agent-hooks-declaration/1.0","declaration":"x","bindings":[]}"#;
    assert_eq!(
        HostDeclaration::from_json(dup).unwrap_err().class,
        DeclarationErrorClass::Malformed
    );
    let deep = format!(
        r#"{{"declaration":"agent-hooks-declaration/1.0","extensions":{{"x":{}1{}}},"bindings":[]}}"#,
        "[".repeat(40),
        "]".repeat(40)
    );
    assert_eq!(
        HostDeclaration::from_json(&deep).unwrap_err().class,
        DeclarationErrorClass::Malformed
    );
    for bad in ["[]", "{} {}", "not json"] {
        assert_eq!(
            HostDeclaration::from_json(bad).unwrap_err().class,
            DeclarationErrorClass::Malformed,
            "{bad}"
        );
    }
}

#[test]
fn knobs_are_consulted_only_by_their_profile() {
    let doc = |composition: Value| {
        with(
            minimal(),
            "/configuration",
            json!({"posture": {"tool_seam_host_error": "terminate"}, "composition": composition}),
        )
    };
    assert_eq!(
        class_of(doc(
            json!({"profile": "sequential/run_all", "on_approval": "stop"})
        ))
        .0,
        DeclarationErrorClass::Inconsistent
    );
    let r = load(doc(
        json!({"profile": "parallel/unanimous", "on_disagreement": "approval"}),
    ))
    .unwrap();
    assert_eq!(
        r.configuration.composition,
        Composition::unanimous(SynthesisPolicy::Approval)
    );
    let r = load(doc(json!({"profile": "parallel/strictest"}))).unwrap();
    assert_eq!(
        r.configuration.composition,
        Composition::strictest(SynthesisPolicy::Deny),
        "knob defaults filled"
    );
    // A stated surface.profiles bounds the composition.
    let mut v = doc(json!({"profile": "sequential/first_deny", "on_approval": "resume"}));
    v["surface"] = json!({"profiles": {"sequential/first_deny": {}}});
    assert_eq!(
        class_of(v).1,
        ["/configuration/composition/on_approval"],
        "defaults_only support refuses resume"
    );
}

#[test]
fn the_floor_and_the_point_pairs_are_enforced() {
    let mut v = minimal();
    v["surface"] = json!({"interception_points": ["agent_startup", "input", "output"]});
    let (class, _) = class_of(v);
    assert_eq!(
        class,
        DeclarationErrorClass::InvalidField,
        "fewer than four is invalid"
    );
    let mut v = minimal();
    v["surface"] = json!({
        "interception_points": ["agent_startup", "input", "output", "pre_tool_call", "post_tool_call"],
        "capabilities": ["tool_calls", "host_declaration"]
    });
    assert_eq!(
        class_of(v).0,
        DeclarationErrorClass::Inconsistent,
        "agent_shutdown is in the floor"
    );
    let mut v = minimal();
    v["surface"] = json!({
        "interception_points": ["agent_startup", "input", "output", "agent_shutdown", "pre_model_call"],
        "capabilities": ["host_declaration"]
    });
    assert_eq!(class_of(v).0, DeclarationErrorClass::Inconsistent);
    // A narrower, consistent surface loads; at defaults to it.
    let mut v = minimal();
    v["surface"] = json!({
        "interception_points": ["agent_startup", "input", "output", "agent_shutdown"],
        "capabilities": ["host_declaration"]
    });
    let r = load(v).unwrap();
    assert_eq!(r.bindings[0].at.len(), 4);
}

#[test]
fn bindings_need_unique_ids_and_at_within_the_surface() {
    let mut v = minimal();
    v["bindings"] = json!([
        {"id": "guard", "kind": INTERCEPTOR_KIND},
        {"id": "guard", "kind": INTERCEPTOR_KIND}
    ]);
    assert_eq!(class_of(v).1, ["/bindings/1/id"]);
    let mut v = minimal();
    v["surface"] = json!({
        "interception_points": ["agent_startup", "input", "output", "agent_shutdown"],
        "capabilities": ["host_declaration"]
    });
    v["bindings"][0]["at"] = json!(["pre_tool_call"]);
    assert_eq!(
        class_of(v),
        (
            DeclarationErrorClass::Inconsistent,
            vec!["/bindings/0/at".into()]
        )
    );
}

#[test]
fn the_surface_is_checked_against_this_host() {
    // The spec default posture is continue; this host terminates.
    let mut v = minimal();
    v["configuration"] = json!({});
    assert_eq!(
        class_of(v),
        (
            DeclarationErrorClass::SurfaceUnsupported,
            vec!["/configuration/posture/tool_seam_host_error".into()]
        )
    );
    let mut v = minimal();
    v["surface"] = json!({"buffered_output": false, "exposure_bound": "one chunk"});
    assert_eq!(class_of(v).0, DeclarationErrorClass::SurfaceUnsupported);
    let mut v = minimal();
    v["surface"] =
        json!({"capabilities": ["model_calls", "tool_calls", "streaming", "host_declaration"]});
    assert_eq!(
        class_of(v),
        (
            DeclarationErrorClass::SurfaceUnsupported,
            vec!["/surface/capabilities".into()]
        )
    );
}

#[test]
fn references_then_kinds_must_be_registered() {
    let mut v = minimal();
    v["configuration"]["approval"] = json!({"resolver": "human", "redactor": "strip"});
    v["configuration"]["identity_provider"] = json!("mine");
    v["bindings"][0]["kind"] = json!("acme.guard");
    let (class, pointers) = class_of(v.clone());
    assert_eq!(class, DeclarationErrorClass::ReferenceUnresolved);
    assert_eq!(pointers.len(), 3);

    let resolver: Arc<dyn ApprovalResolver> = Arc::new(crate::agent_hooks::approval_resolver_fn(
        |r: crate::agent_hooks::ApprovalRequest| async move {
            Ok(crate::agent_hooks::ApprovalResolution::approve(
                &r,
                Verdict::allow(),
            ))
        },
    ));
    let registry = registry()
        .approval_resolver("human", resolver)
        .unwrap()
        .approval_redactor("strip", Arc::new(|c: &InterceptionContext| c.clone()))
        .unwrap()
        .identity_provider("mine", |c| c.sequence().to_string())
        .unwrap();
    let e = registry
        .resolve(&HostDeclaration::from_value(v.clone()).unwrap())
        .unwrap_err();
    assert_eq!(
        (e.class, e.findings[0].pointer.as_str()),
        (DeclarationErrorClass::KindUnknown, "/bindings/0/kind")
    );
    let registry = registry
        .kind(
            "acme.guard",
            Arc::new(|_config: &Value, _b: &BindingContext<'_>| Ok(allow())),
        )
        .unwrap();
    let options =
        AgentHooksOptions::from_declaration(&HostDeclaration::from_value(v).unwrap(), &registry);
    assert!(options.is_ok());
}

#[test]
fn kind_resolvers_can_refuse_their_binding_without_echoing_config() {
    let mut v = minimal();
    v["bindings"][0]["config"] = json!({"secret": "s3cr3t"});
    let e =
        AgentHooksOptions::from_declaration(&HostDeclaration::from_value(v).unwrap(), &registry())
            .unwrap_err();
    assert_eq!(e.class, DeclarationErrorClass::BindingRejected);
    assert!(!e.to_string().contains("s3cr3t"), "{e}");

    // config.name picks a differently named interceptor.
    let mut v = minimal();
    v["bindings"][0] =
        json!({"id": "other", "kind": INTERCEPTOR_KIND, "config": {"name": "guard"}});
    assert!(AgentHooksOptions::from_declaration(
        &HostDeclaration::from_value(v).unwrap(),
        &registry()
    )
    .is_ok());

    let panicking: KindResolver = Arc::new(|_, _| panic!("boom"));
    let registry = HostRegistry::new().kind("acme.boom", panicking).unwrap();
    let v = with(minimal(), "/bindings/0/kind", json!("acme.boom"));
    let e =
        AgentHooksOptions::from_declaration(&HostDeclaration::from_value(v).unwrap(), &registry)
            .unwrap_err();
    assert_eq!(e.class, DeclarationErrorClass::BindingRejected);
}

#[test]
fn registry_names_are_checked() {
    let kind = || -> KindResolver { Arc::new(|_, _| Ok(allow())) };
    assert!(HostRegistry::new().kind("single", kind()).is_err());
    assert!(HostRegistry::new().kind("agent_hooks.x", kind()).is_err());
    assert!(HostRegistry::new().kind("ctk.x", kind()).is_err());
    assert!(HostRegistry::new()
        .kind("acme.x", kind())
        .unwrap()
        .kind("acme.x", kind())
        .is_err());
    assert!(HostRegistry::new()
        .identity_provider("jcs-mine", |_| String::new())
        .is_err());
    assert!(HostRegistry::new()
        .approval_redactor("Bad", Arc::new(|c: &InterceptionContext| c.clone()))
        .is_err());
}

#[test]
fn a_missing_file_is_unreadable() {
    let e = HostDeclaration::from_path("/definitely/not/here.json").unwrap_err();
    assert_eq!(e.class, DeclarationErrorClass::Unreadable);
    let dir = std::env::temp_dir().join(format!("ah-decl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("d.json");
    std::fs::write(&path, minimal().to_string()).unwrap();
    assert!(HostDeclaration::from_path(&path).is_ok());
    let mut bom = vec![0xEF, 0xBB, 0xBF];
    bom.extend(minimal().to_string().into_bytes());
    std::fs::write(&path, bom).unwrap();
    assert_eq!(
        HostDeclaration::from_path(&path).unwrap_err().class,
        DeclarationErrorClass::Unreadable
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn bindings_run_only_at_their_points_and_records_carry_the_declaration() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let witness = seen.clone();
    let output_only: Arc<dyn Interceptor> = Arc::new(interceptor_fn(move |ctx| {
        witness.lock().unwrap().push(ctx.point());
        async { Ok(Verdict::deny("no_output")) }
    }));
    let registry = HostRegistry::new()
        .interceptors([("guard", allow()), ("egress", output_only)])
        .unwrap();
    let mut v = minimal();
    v["bindings"] = json!([
        {"id": "guard", "kind": INTERCEPTOR_KIND},
        {"id": "egress", "kind": INTERCEPTOR_KIND, "at": ["output"], "timeout_ms": null}
    ]);
    let records = Arc::new(Mutex::new(Vec::<InterceptionRecord>::new()));
    let sink = records.clone();
    let options =
        AgentHooksOptions::from_declaration(&HostDeclaration::from_value(v).unwrap(), &registry)
            .unwrap()
            .record_sink(Arc::new(move |r: &InterceptionRecord| {
                sink.lock().unwrap().push(r.clone())
            }));
    // Build the emitter the bundle would, through the bundle's own agent.
    let hooks = AgentHooks::new(options).unwrap();
    let agent = hooks.build_agent(crate::agent::Agent::builder(Echo));
    let err = agent
        .run(vec![crate::types::Message::user("hi")], None)
        .await
        .unwrap_err();
    assert_eq!(
        err.interception_blocked().and_then(|b| b.reason()),
        Some("no_output")
    );
    assert_eq!(*seen.lock().unwrap(), [InterceptionPoint::Output]);
    let records = records.lock().unwrap();
    assert!(records
        .iter()
        .all(|r| r.declaration.as_deref() == Some(DECLARATION_VERSION)));
    let output = records
        .iter()
        .find(|r| r.interception_point == InterceptionPoint::Output)
        .unwrap();
    assert_eq!(output.verdicts.len(), 2);
    assert_eq!(output.verdicts[1].name.as_deref(), Some("egress"));
    let input = records
        .iter()
        .find(|r| r.interception_point == InterceptionPoint::Input)
        .unwrap();
    assert_eq!(input.verdicts.len(), 1, "egress does not run at input");
}

#[tokio::test]
async fn a_point_with_no_binding_fails_closed() {
    let mut v = minimal();
    v["bindings"][0]["at"] = json!(["output"]);
    let options =
        AgentHooksOptions::from_declaration(&HostDeclaration::from_value(v).unwrap(), &registry())
            .unwrap();
    let hooks = AgentHooks::new(options).unwrap();
    let agent = hooks.build_agent(crate::agent::Agent::builder(Echo));
    let err = agent
        .run(vec![crate::types::Message::user("hi")], None)
        .await
        .unwrap_err();
    assert_eq!(
        err.interception_blocked().and_then(|b| b.reason()),
        Some(host_error::NO_INTERCEPTOR)
    );
}

#[test]
fn a_declaration_without_bindings_is_refused_by_the_bundle() {
    let v = with(minimal(), "/bindings", json!([]));
    let err = AgentHooks::from_declaration(&HostDeclaration::from_value(v).unwrap(), &registry())
        .unwrap_err();
    assert!(
        err.to_string().contains("at least one interceptor"),
        "{err}"
    );
    let _ = InterceptionContextBuilder::new("a", "agent-framework", "s");
}

/// Replies "ok".
struct Echo;

#[async_trait::async_trait]
impl crate::client::ChatClient for Echo {
    async fn get_response(
        &self,
        _messages: Vec<crate::types::Message>,
        _options: crate::types::ChatOptions,
    ) -> crate::Result<crate::types::ChatResponse> {
        Ok(crate::types::ChatResponse::from_text("ok"))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<crate::types::Message>,
        options: crate::types::ChatOptions,
    ) -> crate::Result<crate::client::ChatStream> {
        use futures::StreamExt;
        let response = self.get_response(messages, options).await?;
        let update = crate::types::ChatResponseUpdate {
            contents: vec![crate::types::Content::text(response.text())],
            role: Some(crate::types::Role::assistant()),
            ..Default::default()
        };
        Ok(futures::stream::iter(vec![Ok(update)]).boxed())
    }
}
