//! Tests for `${VAR}` / `${VAR:-default}` environment interpolation, both at
//! the string level and through the loader.

use std::collections::HashMap;

use agent_framework_declarative::env::{interpolate, EnvSource};
use agent_framework_declarative::{DeclarativeError, DeclarativeLoader};

fn fixed(pairs: &[(&str, &str)]) -> impl EnvSource + Send + Sync + 'static {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    move |key: &str| map.get(key).cloned()
}

#[test]
fn expands_simple_and_embedded() {
    let env = fixed(&[("FOO", "bar")]);
    assert_eq!(interpolate("${FOO}", &env).unwrap(), "bar");
    assert_eq!(interpolate("a ${FOO} b", &env).unwrap(), "a bar b");
    assert_eq!(
        interpolate("no placeholders", &env).unwrap(),
        "no placeholders"
    );
}

#[test]
fn default_used_when_unset_or_empty() {
    let env = fixed(&[("SET", "value"), ("EMPTY", "")]);
    assert_eq!(
        interpolate("${MISSING:-fallback}", &env).unwrap(),
        "fallback"
    );
    assert_eq!(interpolate("${SET:-fallback}", &env).unwrap(), "value");
    // POSIX `:-` falls back to default on an *empty* value too.
    assert_eq!(interpolate("${EMPTY:-fallback}", &env).unwrap(), "fallback");
}

#[test]
fn nested_default_is_interpolated() {
    let env = fixed(&[("INNER", "xy")]);
    assert_eq!(interpolate("${OUTER:-${INNER}}", &env).unwrap(), "xy");
}

#[test]
fn missing_without_default_errors() {
    let env = fixed(&[]);
    let err = interpolate("${MISSING}", &env).unwrap_err();
    assert!(
        matches!(err, DeclarativeError::MissingEnvVar(ref v) if v == "MISSING"),
        "got {err:?}"
    );
}

#[test]
fn dollar_escape_is_literal() {
    let env = fixed(&[("FOO", "bar")]);
    assert_eq!(interpolate("$${FOO}", &env).unwrap(), "${FOO}");
}

#[test]
fn unterminated_placeholder_errors() {
    let env = fixed(&[]);
    let err = interpolate("prefix ${FOO", &env).unwrap_err();
    assert!(
        matches!(err, DeclarativeError::MalformedPlaceholder { .. }),
        "got {err:?}"
    );
}

#[test]
fn loader_interpolates_across_string_fields() {
    let loader = DeclarativeLoader::new()
        .with_env(fixed(&[("GREETING", "Be terse."), ("MODEL_ID", "gpt-4o")]));

    let yaml = "\
kind: Prompt
name: A
instructions: ${GREETING}
model:
  id: ${MODEL_ID}
  provider: ${PROVIDER:-OpenAI}
";
    let spec = loader.load_agent_spec(yaml).expect("interpolated parse");
    assert_eq!(spec.instructions.as_deref(), Some("Be terse."));
    let model = spec.model.unwrap();
    assert_eq!(model.id.as_deref(), Some("gpt-4o"));
    assert_eq!(model.provider.as_deref(), Some("OpenAI")); // default applied
}

#[test]
fn loader_reports_missing_env_var() {
    let loader = DeclarativeLoader::new().with_env(fixed(&[]));
    let yaml = "kind: Prompt\nname: A\ninstructions: ${REQUIRED_BUT_MISSING}\n";
    let err = loader.load_agent_spec(yaml).unwrap_err();
    assert!(
        matches!(err, DeclarativeError::MissingEnvVar(ref v) if v == "REQUIRED_BUT_MISSING"),
        "got {err:?}"
    );
}
