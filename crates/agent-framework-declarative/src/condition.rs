//! A tiny, documented condition mini-language for workflow edges.
//!
//! An edge/case `condition` string has the form `PATH OP LITERAL`, where:
//!
//! * `PATH` is a dot-separated path into the JSON message
//!   (e.g. `status`, `data.kind`, `items.0`). Numeric segments index arrays.
//! * `OP` is one of `==`, `!=`, `<`, `<=`, `>`, `>=`.
//! * `LITERAL` is a JSON literal (`"done"`, `42`, `true`, `null`) — or, as a
//!   convenience, a bare unquoted string (`done`).
//!
//! Equality (`==` / `!=`) compares JSON values (numbers compare numerically
//! across integer/float). Ordering (`<`, `<=`, `>`, `>=`) applies to numbers
//! only; a non-numeric operand makes the comparison `false`.
//!
//! For anything beyond this grammar, register a named predicate in the
//! [`PredicateRegistry`](crate::PredicateRegistry) and reference it with
//! `predicate:` instead of `condition:`.

use serde_json::Value;
use std::sync::Arc;

use agent_framework_core::workflow::Condition;

use crate::error::{DeclarativeError, Result};

/// The comparison operators, longest-first so `>=` is matched before `>`.
const OPERATORS: &[&str] = &["==", "!=", ">=", "<=", ">", "<"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

impl Op {
    fn parse(s: &str) -> Option<Op> {
        Some(match s {
            "==" => Op::Eq,
            "!=" => Op::Ne,
            ">" => Op::Gt,
            ">=" => Op::Ge,
            "<" => Op::Lt,
            "<=" => Op::Le,
            _ => return None,
        })
    }
}

/// Parse a condition mini-expression into a reusable [`Condition`] predicate.
pub fn parse(expr: &str) -> Result<Condition> {
    let (path_str, op, literal_str) = split(expr)?;
    let path: Vec<String> = path_str
        .split('.')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if path.is_empty() {
        return Err(DeclarativeError::InvalidCondition {
            expr: expr.to_string(),
            reason: "empty left-hand path".to_string(),
        });
    }
    let expected = parse_literal(literal_str);

    Ok(Arc::new(move |msg: &Value| {
        let actual = lookup(msg, &path);
        match actual {
            Some(actual) => compare(actual, op, &expected),
            None => false,
        }
    }))
}

/// Split `expr` into `(path, op, literal)` around the first operator found.
fn split(expr: &str) -> Result<(&str, Op, &str)> {
    // Scan left-to-right; at each position test the multi-char operators first.
    let bytes = expr.as_bytes();
    for i in 0..bytes.len() {
        for op_str in OPERATORS {
            if expr[i..].starts_with(op_str) {
                let left = expr[..i].trim();
                let right = expr[i + op_str.len()..].trim();
                if left.is_empty() {
                    break;
                }
                let op = Op::parse(op_str).expect("operator table is consistent");
                return Ok((left, op, right));
            }
        }
    }
    Err(DeclarativeError::InvalidCondition {
        expr: expr.to_string(),
        reason: "expected an operator (==, !=, <, <=, >, >=)".to_string(),
    })
}

/// Parse the right-hand literal: a JSON value, else a bare string.
fn parse_literal(s: &str) -> Value {
    if let Ok(v) = serde_json::from_str::<Value>(s) {
        v
    } else {
        Value::String(s.to_string())
    }
}

/// Navigate a dot-path into a JSON value.
fn lookup<'a>(value: &'a Value, path: &[String]) -> Option<&'a Value> {
    let mut cur = value;
    for seg in path {
        cur = match cur {
            Value::Object(map) => map.get(seg)?,
            Value::Array(arr) => {
                let idx: usize = seg.parse().ok()?;
                arr.get(idx)?
            }
            _ => return None,
        };
    }
    Some(cur)
}

/// Compare `actual OP expected`.
fn compare(actual: &Value, op: Op, expected: &Value) -> bool {
    match op {
        Op::Eq => values_equal(actual, expected),
        Op::Ne => !values_equal(actual, expected),
        Op::Gt | Op::Ge | Op::Lt | Op::Le => match (actual.as_f64(), expected.as_f64()) {
            (Some(a), Some(b)) => match op {
                Op::Gt => a > b,
                Op::Ge => a >= b,
                Op::Lt => a < b,
                Op::Le => a <= b,
                _ => unreachable!(),
            },
            _ => false,
        },
    }
}

/// Equality that treats integer and float numbers with the same value as equal.
fn values_equal(a: &Value, b: &Value) -> bool {
    if let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) {
        return x == y;
    }
    a == b
}
