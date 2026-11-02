//! The `User-Agent` every HTTP-backed crate in this workspace sends.
//!
//! A request that sends no `User-Agent` is traffic a provider cannot
//! attribute: Azure reads it for telemetry, throttling diagnostics and
//! supportability, and other providers use it the same way. Upstream
//! identifies itself per package (#9223).
//!
//! Only the *format* lives here. This crate deliberately has no `reqwest`
//! dependency, so each provider crate builds its own client and passes this
//! string to `reqwest::ClientBuilder::user_agent` — which keeps the one
//! format shared without dragging an HTTP stack into the core.

/// The product token every request from this port carries.
pub const USER_AGENT_PRODUCT: &str = "agent-framework-rust";

/// The `User-Agent` value for a request made by `component` at `version`.
///
/// `component` is the crate making the call and `version` its
/// `CARGO_PKG_VERSION`, so a request names both the framework and the
/// connector inside it: `agent-framework-rust/0.14.0 (agent-framework-openai)`.
pub fn user_agent(component: &str, version: &str) -> String {
    format!("{USER_AGENT_PRODUCT}/{version} ({component})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_the_framework_version_and_component() {
        assert_eq!(
            user_agent("agent-framework-openai", "1.2.3"),
            "agent-framework-rust/1.2.3 (agent-framework-openai)"
        );
    }
}
