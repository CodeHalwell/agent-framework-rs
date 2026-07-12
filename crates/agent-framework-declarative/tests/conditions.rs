//! Unit tests for the edge-condition mini-language.

use agent_framework_declarative::condition::parse;
use serde_json::json;

#[tokio::test]
async fn string_equality_quoted_and_bare() {
    let quoted = parse("status == \"done\"").unwrap();
    assert!(quoted(&json!({"status": "done"})).await);
    assert!(!quoted(&json!({"status": "pending"})).await);

    // A bare (unquoted) literal is accepted as a string.
    let bare = parse("status == done").unwrap();
    assert!(bare(&json!({"status": "done"})).await);
}

#[tokio::test]
async fn inequality() {
    let ne = parse("status != done").unwrap();
    assert!(ne(&json!({"status": "pending"})).await);
    assert!(!ne(&json!({"status": "done"})).await);
}

#[tokio::test]
async fn numeric_ordering_and_cross_type_equality() {
    let ge = parse("n >= 3").unwrap();
    assert!(ge(&json!({"n": 5})).await);
    assert!(ge(&json!({"n": 3})).await);
    assert!(!ge(&json!({"n": 2})).await);

    // Integer 5 equals float 5.0 across JSON number types.
    let eq = parse("n == 5").unwrap();
    assert!(eq(&json!({"n": 5.0})).await);

    // Ordering against a non-numeric operand is false, not an error.
    assert!(!ge(&json!({"n": "lots"})).await);
}

#[tokio::test]
async fn boolean_and_null_literals() {
    let is_true = parse("flag == true").unwrap();
    assert!(is_true(&json!({"flag": true})).await);
    assert!(!is_true(&json!({"flag": false})).await);

    let is_null = parse("value == null").unwrap();
    assert!(is_null(&json!({"value": null})).await);
}

#[tokio::test]
async fn nested_path_and_array_index() {
    let nested = parse("data.kind == alert").unwrap();
    assert!(nested(&json!({"data": {"kind": "alert"}})).await);

    let indexed = parse("items.0 == first").unwrap();
    assert!(indexed(&json!({"items": ["first", "second"]})).await);
}

#[tokio::test]
async fn missing_path_is_false() {
    let cond = parse("missing.field == x").unwrap();
    assert!(!cond(&json!({"present": 1})).await);
}

#[test]
fn malformed_expression_errors() {
    assert!(parse("no operator here").is_err());
    assert!(parse("== rightonly").is_err());
}
