//! Small shared helpers: timestamps and id generation.

use std::time::{SystemTime, UNIX_EPOCH};

use uuid::Uuid;

/// Unix time in fractional seconds (OpenAI `created_at` convention).
pub(crate) fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// A short hex fragment for synthesized ids.
pub(crate) fn short_hex() -> String {
    Uuid::new_v4().simple().to_string()[..8].to_string()
}

/// A `resp_…` id (OpenAI response id convention).
pub(crate) fn resp_id() -> String {
    format!("resp_{}", short_hex())
}

/// A `msg_…` id (OpenAI message-item id convention).
pub(crate) fn msg_id() -> String {
    format!("msg_{}", short_hex())
}
