//! Composition profiles (§7) and the approval seam (§9).
//!
//! Composition is host configuration, never verdict content: the host picks
//! a profile before dispatch and every record carries it. The profile set is
//! closed:
//!
//! | Profile | Dispatch | Combination |
//! | --- | --- | --- |
//! | `sequential/first_deny` (default) | in order; each sees its predecessors' transforms | the first deny wins; with a resolver, a liftable deny is consulted and `on_approval` decides whether the fold stops or resumes |
//! | `sequential/run_all` | in order, every interceptor runs; transforms fold through | the most severe verdict wins (`deny` > liftable deny > `transform` > `allow`) |
//! | `parallel/strictest` | concurrently, on isolated snapshots | the most severe verdict wins; two transforms conflict (`on_transform_conflict`) |
//! | `parallel/unanimous` | concurrently, on isolated snapshots | anything but unanimous `allow` is a disagreement (`on_disagreement`) |
//!
//! A conflict or disagreement synthesizes a deny; with the `approval`
//! policy it is a liftable one, consulted through the resolver.
//!
//! The approval seam: a liftable deny ([`Verdict::escalate`]) in `enforce`
//! mode, at any point but `agent_shutdown`, is handed to the registered
//! [`ApprovalResolver`] (typically a human). Its resolution must echo the
//! request's `context_identity` byte for byte, and its verdict must agree
//! with its outcome (`approve` carries a permit, `reject` a deny);
//! anything else fails closed. Without a resolver the deny stands.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::protocol::{Decision, InterceptionContext, InterceptionPoint, Verdict};

/// The closed profile set (§7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum CompositionProfile {
    /// Fold in order; the first deny short-circuits.
    #[default]
    #[serde(rename = "sequential/first_deny")]
    SequentialFirstDeny,
    /// Fold in order; everything runs; the most severe verdict wins.
    #[serde(rename = "sequential/run_all")]
    SequentialRunAll,
    /// Isolated snapshots; the most severe verdict wins.
    #[serde(rename = "parallel/strictest")]
    ParallelStrictest,
    /// Isolated snapshots; anything but unanimous allow is a disagreement.
    #[serde(rename = "parallel/unanimous")]
    ParallelUnanimous,
}

impl CompositionProfile {
    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SequentialFirstDeny => "sequential/first_deny",
            Self::SequentialRunAll => "sequential/run_all",
            Self::ParallelStrictest => "parallel/strictest",
            Self::ParallelUnanimous => "parallel/unanimous",
        }
    }

    /// Whether interceptors see their predecessors' transforms.
    pub fn is_sequential(self) -> bool {
        matches!(self, Self::SequentialFirstDeny | Self::SequentialRunAll)
    }
}

/// `sequential/first_deny`: what an approved liftable deny does to the rest
/// of the fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnApproval {
    /// The resolution becomes the combined verdict and the emission ends.
    Stop,
    /// The resolution takes the denying interceptor's place and the fold
    /// continues.
    Resume,
}

/// What a parallel conflict or disagreement synthesizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynthesisPolicy {
    /// A plain deny.
    Deny,
    /// A liftable deny, consulted through the approval seam.
    Approval,
}

/// The composition in effect for an emitter; serialized verbatim as each
/// record's `composition` block (§10.3), carrying exactly the knobs the
/// profile consults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Composition {
    /// The profile.
    pub profile: CompositionProfile,
    /// `sequential/first_deny` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_approval: Option<OnApproval>,
    /// `parallel/unanimous` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_disagreement: Option<SynthesisPolicy>,
    /// `parallel/strictest` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_transform_conflict: Option<SynthesisPolicy>,
}

impl Default for Composition {
    /// `sequential/first_deny` with `on_approval: stop`.
    fn default() -> Self {
        Self::first_deny(OnApproval::Stop)
    }
}

impl Composition {
    fn bare(profile: CompositionProfile) -> Self {
        Self {
            profile,
            on_approval: None,
            on_disagreement: None,
            on_transform_conflict: None,
        }
    }

    /// `sequential/first_deny`.
    pub fn first_deny(on_approval: OnApproval) -> Self {
        Self {
            on_approval: Some(on_approval),
            ..Self::bare(CompositionProfile::SequentialFirstDeny)
        }
    }

    /// `sequential/run_all`.
    pub fn run_all() -> Self {
        Self::bare(CompositionProfile::SequentialRunAll)
    }

    /// `parallel/strictest`.
    pub fn strictest(on_transform_conflict: SynthesisPolicy) -> Self {
        Self {
            on_transform_conflict: Some(on_transform_conflict),
            ..Self::bare(CompositionProfile::ParallelStrictest)
        }
    }

    /// `parallel/unanimous`.
    pub fn unanimous(on_disagreement: SynthesisPolicy) -> Self {
        Self {
            on_disagreement: Some(on_disagreement),
            ..Self::bare(CompositionProfile::ParallelUnanimous)
        }
    }

    /// Fill the §7.2 knob defaults the profile consults and clear the
    /// others, so every record carries the resolved configuration.
    pub(crate) fn normalized(self) -> Self {
        use CompositionProfile as P;
        match self.profile {
            P::SequentialFirstDeny => {
                Self::first_deny(self.on_approval.unwrap_or(OnApproval::Stop))
            }
            P::SequentialRunAll => Self::run_all(),
            P::ParallelStrictest => {
                Self::strictest(self.on_transform_conflict.unwrap_or(SynthesisPolicy::Deny))
            }
            P::ParallelUnanimous => {
                Self::unanimous(self.on_disagreement.unwrap_or(SynthesisPolicy::Deny))
            }
        }
    }
}

/// The §5.1 severity order: `deny` > liftable deny > `transform` > `allow`.
pub(crate) fn severity(v: &Verdict) -> u8 {
    match (v.decision, v.is_liftable()) {
        (Decision::Deny, false) => 3,
        (Decision::Deny, true) => 2,
        (Decision::Transform, _) => 1,
        (Decision::Allow, _) => 0,
    }
}

/// Severity-max aggregation (§7.3).
pub(crate) enum Aggregate {
    /// The winner's index.
    Winner(usize),
    /// Two or more transforms tied at the top on isolated snapshots.
    TransformConflict(Vec<usize>),
}

/// The most severe verdict. Ties between blocks go to the lowest index;
/// sequential transforms compose, so the last one carries the cumulative
/// state; parallel transforms against one snapshot do not compose.
pub(crate) fn aggregate_strictest(verdicts: &[Verdict], sequential: bool) -> Aggregate {
    let max = verdicts.iter().map(severity).max().unwrap_or(0);
    let top: Vec<usize> = verdicts
        .iter()
        .enumerate()
        .filter(|(_, v)| severity(v) == max)
        .map(|(i, _)| i)
        .collect();
    if max == 1 && top.len() >= 2 {
        if sequential {
            Aggregate::Winner(top[top.len() - 1])
        } else {
            Aggregate::TransformConflict(top)
        }
    } else {
        Aggregate::Winner(top.first().copied().unwrap_or(0))
    }
}

/// An approval resolver's outcome (§9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalOutcome {
    /// Lift the deny; the resolution's verdict must be a permit.
    Approve,
    /// Keep it; the resolution's verdict must be a deny.
    Reject,
    /// No decision; the deny stands (`host_error:approval_unresolved`).
    Unresolved,
}

/// What the host hands the resolver for a liftable deny (§9).
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    /// The identity of [`Self::context`] as presented (after any
    /// redaction); `None` when the identity provider is unbound. The
    /// resolution must echo it.
    pub context_identity: Option<String>,
    /// The point being approved.
    pub interception_point: InterceptionPoint,
    /// The liftable deny.
    pub verdict: Verdict,
    /// The context, after any transforms folded so far and the host's
    /// approval redactor.
    pub context: InterceptionContext,
}

/// A resolver's answer (§9).
#[derive(Debug, Clone)]
pub struct ApprovalResolution {
    /// The outcome.
    pub outcome: ApprovalOutcome,
    /// Must equal the request's `context_identity`.
    pub context_identity: Option<String>,
    /// The substituted verdict: a permit for `approve`, a deny for
    /// `reject`. A transform folds like an interceptor's.
    pub verdict: Option<Verdict>,
}

impl ApprovalResolution {
    /// Approve `request`, substituting `verdict` (an allow or transform).
    pub fn approve(request: &ApprovalRequest, verdict: Verdict) -> Self {
        Self {
            outcome: ApprovalOutcome::Approve,
            context_identity: request.context_identity.clone(),
            verdict: Some(verdict),
        }
    }

    /// Reject `request`, keeping a deny with `reason`.
    pub fn reject(request: &ApprovalRequest, reason: impl Into<String>) -> Self {
        Self {
            outcome: ApprovalOutcome::Reject,
            context_identity: request.context_identity.clone(),
            verdict: Some(Verdict::deny(reason)),
        }
    }

    /// Leave `request` unresolved.
    pub fn unresolved(request: &ApprovalRequest) -> Self {
        Self {
            outcome: ApprovalOutcome::Unresolved,
            context_identity: request.context_identity.clone(),
            verdict: None,
        }
    }
}

/// Resolves liftable denies (§9). An error, panic or timeout is a resolver
/// failure (`host_error:approval_resolver_failed`): the deny stands.
#[async_trait]
pub trait ApprovalResolver: Send + Sync {
    /// Resolve one consultation.
    async fn resolve(&self, request: ApprovalRequest) -> crate::Result<ApprovalResolution>;
}

/// Wraps a closure as an [`ApprovalResolver`].
pub struct FnApprovalResolver<F>(F);

/// An [`ApprovalResolver`] from an async closure.
pub fn approval_resolver_fn<F, Fut>(f: F) -> FnApprovalResolver<F>
where
    F: Fn(ApprovalRequest) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = crate::Result<ApprovalResolution>> + Send,
{
    FnApprovalResolver(f)
}

#[async_trait]
impl<F, Fut> ApprovalResolver for FnApprovalResolver<F>
where
    F: Fn(ApprovalRequest) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = crate::Result<ApprovalResolution>> + Send,
{
    async fn resolve(&self, request: ApprovalRequest) -> crate::Result<ApprovalResolution> {
        (self.0)(request).await
    }
}

/// Minimizes the context that leaves through the approval seam (§9, §14).
/// The request's identity is computed over the redacted context, binding
/// the approval to what the approver saw.
pub type ApprovalRedactor = Arc<dyn Fn(&InterceptionContext) -> InterceptionContext + Send + Sync>;

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use super::*;
    use crate::agent_hooks::protocol::{
        host_error, interceptor_fn, InterceptionContextBuilder, InterceptionEmitter, Interceptor,
    };
    use crate::agent_hooks::{context_identity, IdentityProvider};

    fn builder() -> InterceptionContextBuilder {
        InterceptionContextBuilder::new("agent", "agent-framework", "session")
    }

    fn fixed(verdict: Verdict) -> Arc<dyn Interceptor> {
        Arc::new(interceptor_fn(move |_| {
            let verdict = verdict.clone();
            async move { Ok(verdict) }
        }))
    }

    /// Counts its calls and returns `verdict`.
    fn counted(verdict: Verdict, calls: Arc<AtomicUsize>) -> Arc<dyn Interceptor> {
        Arc::new(interceptor_fn(move |_| {
            calls.fetch_add(1, Ordering::SeqCst);
            let verdict = verdict.clone();
            async move { Ok(verdict) }
        }))
    }

    fn emitter(composition: Composition, verdicts: Vec<Verdict>) -> InterceptionEmitter {
        verdicts.into_iter().fold(
            InterceptionEmitter::new().with_composition(composition),
            |e, v| e.register(fixed(v), None),
        )
    }

    #[tokio::test]
    async fn run_all_runs_everything_and_the_strictest_wins() {
        let calls = Arc::new(AtomicUsize::new(0));
        let e = InterceptionEmitter::new()
            .with_composition(Composition::run_all())
            .register(fixed(Verdict::deny("first")), None)
            .register(counted(Verdict::allow(), calls.clone()), None)
            .register(fixed(Verdict::escalate("liftable")), None);
        let blocked = e.emit(builder().output(json!("x"))).await.unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 1, "nothing short-circuits");
        assert_eq!(
            blocked.reason(),
            Some("first"),
            "a plain deny beats a liftable one"
        );
        assert_eq!(blocked.record.decided_by, Some(0));
        assert_eq!(blocked.record.verdicts.len(), 3);
        assert_eq!(blocked.record.fold_truncated, Some(false));
        let wire = serde_json::to_value(&blocked.record).unwrap();
        assert_eq!(
            wire["composition"],
            json!({"profile": "sequential/run_all"})
        );

        // Sequential transforms compose; the last carries the cumulative state.
        let e = emitter(
            Composition::run_all(),
            vec![
                Verdict::transform("$target.content", json!("a")),
                Verdict::transform("$target.content", json!("b")),
            ],
        );
        let out = e.emit(builder().output(json!("x"))).await.unwrap();
        assert_eq!(out.target["content"], "b");
        assert_eq!(out.record.decided_by, Some(1));
    }

    #[tokio::test]
    async fn parallel_strictest_sees_one_snapshot_and_refuses_conflicts() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let rewrite = Arc::new(interceptor_fn(|_| async {
            Ok(Verdict::transform("$target.content", json!("rewritten")))
        }));
        let witness = seen.clone();
        let observe = Arc::new(interceptor_fn(move |ctx| {
            witness.lock().unwrap().push(ctx.target().clone());
            async { Ok(Verdict::allow()) }
        }));
        let e = InterceptionEmitter::new()
            .with_composition(Composition::strictest(SynthesisPolicy::Deny))
            .register(rewrite, None)
            .register(observe, None);
        let out = e.emit(builder().output(json!("x"))).await.unwrap();
        assert_eq!(
            out.target["content"], "rewritten",
            "the single transform applies"
        );
        assert_eq!(seen.lock().unwrap()[0]["content"], "x", "isolated snapshot");
        assert_eq!(out.record.fold_truncated, None);

        let e = emitter(
            Composition::strictest(SynthesisPolicy::Deny),
            vec![
                Verdict::transform("$target.content", json!("a")),
                Verdict::transform("$target.content", json!("b")),
            ],
        );
        let blocked = e.emit(builder().output(json!("x"))).await.unwrap_err();
        assert_eq!(blocked.reason(), Some(host_error::TRANSFORM_CONFLICT));
        assert_eq!(blocked.record.decided_by, None);
    }

    #[tokio::test]
    async fn parallel_unanimous_needs_every_allow() {
        let e = emitter(
            Composition::unanimous(SynthesisPolicy::Deny),
            vec![Verdict::allow(), Verdict::allow()],
        );
        assert!(e.emit(builder().output(json!("x"))).await.is_ok());
        let e = emitter(
            Composition::unanimous(SynthesisPolicy::Deny),
            vec![
                Verdict::allow(),
                Verdict::transform("$target.content", json!("a")),
            ],
        );
        let blocked = e.emit(builder().output(json!("x"))).await.unwrap_err();
        assert_eq!(blocked.reason(), Some(host_error::COMPOSITION_DISAGREEMENT));
    }

    fn approve_all() -> Arc<dyn ApprovalResolver> {
        Arc::new(approval_resolver_fn(
            |request: ApprovalRequest| async move {
                Ok(ApprovalResolution::approve(&request, Verdict::allow()))
            },
        ))
    }

    #[tokio::test]
    async fn an_approved_liftable_deny_stops_or_resumes_the_fold() {
        for (on_approval, later_calls, truncated) in [
            (OnApproval::Stop, 0, Some(true)),
            (OnApproval::Resume, 1, Some(false)),
        ] {
            let calls = Arc::new(AtomicUsize::new(0));
            let e = InterceptionEmitter::new()
                .with_composition(Composition::first_deny(on_approval))
                .with_approval_resolver(approve_all())
                .register(fixed(Verdict::escalate("needs_review")), None)
                .register(counted(Verdict::allow(), calls.clone()), None);
            let out = e.emit(builder().output(json!("x"))).await.unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), later_calls, "{on_approval:?}");
            assert_eq!(out.record.resolved_by, Some("approval"));
            assert_eq!(out.record.fold_truncated, truncated);
        }
    }

    #[tokio::test]
    async fn rejections_failures_and_echo_violations_keep_the_deny() {
        // No resolver: the liftable deny stands, unconsulted.
        let e = emitter(Composition::default(), vec![Verdict::escalate("r")]);
        let blocked = e.emit(builder().output(json!("x"))).await.unwrap_err();
        assert_eq!(blocked.reason(), Some("r"));
        assert_eq!(blocked.record.resolved_by, None);

        let reject: Arc<dyn ApprovalResolver> = Arc::new(approval_resolver_fn(
            |request: ApprovalRequest| async move {
                Ok(ApprovalResolution::reject(&request, "reviewer_said_no"))
            },
        ));
        let wrong_echo: Arc<dyn ApprovalResolver> = Arc::new(approval_resolver_fn(
            |request: ApprovalRequest| async move {
                let mut resolution = ApprovalResolution::approve(&request, Verdict::allow());
                resolution.context_identity = Some("sha256:forged".into());
                Ok(resolution)
            },
        ));
        let failing: Arc<dyn ApprovalResolver> = Arc::new(approval_resolver_fn(
            |_request: ApprovalRequest| async move { Err(crate::Error::Configuration("down".into())) },
        ));
        let disagreeing: Arc<dyn ApprovalResolver> = Arc::new(approval_resolver_fn(
            |request: ApprovalRequest| async move {
                Ok(ApprovalResolution {
                    outcome: ApprovalOutcome::Approve,
                    context_identity: request.context_identity,
                    verdict: Some(Verdict::deny("x")),
                })
            },
        ));
        let unresolved: Arc<dyn ApprovalResolver> = Arc::new(approval_resolver_fn(
            |request: ApprovalRequest| async move { Ok(ApprovalResolution::unresolved(&request)) },
        ));
        for (resolver, reason) in [
            (reject, "reviewer_said_no"),
            (wrong_echo, host_error::APPROVAL_IDENTITY_MISMATCH),
            (failing, host_error::APPROVAL_RESOLVER_FAILED),
            (disagreeing, host_error::VERDICT_INVALID),
            (unresolved, host_error::APPROVAL_UNRESOLVED),
        ] {
            let e = emitter(Composition::default(), vec![Verdict::escalate("r")])
                .with_approval_resolver(resolver);
            let blocked = e.emit(builder().output(json!("x"))).await.unwrap_err();
            assert_eq!(blocked.reason(), Some(reason));
            assert_eq!(blocked.record.resolved_by, Some("rejection"));
        }

        // Never consulted at agent_shutdown.
        let e = emitter(Composition::default(), vec![Verdict::escalate("r")])
            .with_approval_resolver(approve_all());
        let blocked = e
            .emit(builder().agent_shutdown(crate::agent_hooks::ShutdownReason::Completed))
            .await
            .unwrap_err();
        assert_eq!(blocked.record.resolved_by, None);
    }

    #[tokio::test]
    async fn a_disagreement_can_be_put_to_the_resolver() {
        let e = emitter(
            Composition::unanimous(SynthesisPolicy::Approval),
            vec![Verdict::allow(), Verdict::deny("no")],
        )
        .with_approval_resolver(approve_all());
        let out = e.emit(builder().output(json!("x"))).await.unwrap();
        assert_eq!(out.record.resolved_by, Some("approval"));
    }

    #[tokio::test]
    async fn the_resolver_sees_the_redacted_context_and_its_identity() {
        let seen = Arc::new(std::sync::Mutex::new(None));
        let witness = seen.clone();
        let resolver: Arc<dyn ApprovalResolver> =
            Arc::new(approval_resolver_fn(move |request: ApprovalRequest| {
                let witness = witness.clone();
                async move {
                    let expected = context_identity(&request.context).unwrap();
                    assert_eq!(request.context_identity.as_deref(), Some(expected.as_str()));
                    *witness.lock().unwrap() = Some(request.context.target().clone());
                    Ok(ApprovalResolution::approve(&request, Verdict::allow()))
                }
            }));
        let e = emitter(Composition::default(), vec![Verdict::escalate("r")])
            .with_approval_resolver(resolver)
            .with_approval_redactor(Arc::new(|ctx: &InterceptionContext| {
                let mut ctx = ctx.clone();
                ctx.set("target", json!("[redacted]"));
                ctx
            }));
        e.emit(builder().output(json!("secret"))).await.unwrap();
        assert_eq!(seen.lock().unwrap().clone(), Some(json!("[redacted]")));
    }

    #[tokio::test]
    async fn identities_bind_records_to_their_contexts() {
        // Before and after a transform.
        let e = emitter(
            Composition::default(),
            vec![Verdict::transform("$target.content", json!("y"))],
        );
        let out = e.emit(builder().output(json!("x"))).await.unwrap();
        let (input, enforced) = (out.record.input_identity, out.record.enforced_identity);
        assert!(input.is_some() && enforced.is_some());
        assert_ne!(input, enforced);

        // Optional fields do not perturb the identity.
        let plain = builder().output(json!("x"));
        let mut named = plain.clone();
        named.set(
            "agent",
            json!({"id": "agent", "framework": "agent-framework", "name": "n"}),
        );
        assert_eq!(context_identity(&plain), context_identity(&named));

        // Unbound and custom providers.
        let e = emitter(Composition::default(), vec![Verdict::allow()])
            .with_identity_provider(IdentityProvider::Null);
        let record = e.emit(builder().output(json!("x"))).await.unwrap().record;
        assert_eq!(
            (record.input_identity, record.identity_provider),
            (None, None)
        );
        let e = emitter(Composition::default(), vec![Verdict::allow()]).with_identity_provider(
            IdentityProvider::custom("seq", |ctx| ctx.sequence().to_string()).unwrap(),
        );
        let record = e.emit(builder().output(json!("x"))).await.unwrap().record;
        assert_eq!(record.identity_provider.as_deref(), Some("seq"));
        assert_eq!(record.input_identity.as_deref(), Some("0"));

        // Integers beyond 2^53 fail closed before any interceptor runs.
        let calls = Arc::new(AtomicUsize::new(0));
        let e = InterceptionEmitter::new().register(counted(Verdict::allow(), calls.clone()), None);
        let blocked = e
            .emit(builder().output(json!({"id": 9_007_199_254_740_993_u64})))
            .await
            .unwrap_err();
        assert_eq!(blocked.reason(), Some(host_error::CONTEXT_INVALID));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}
