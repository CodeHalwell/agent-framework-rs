//! A decoder for the AWS event-stream binary framing
//! (`application/vnd.amazon.eventstream`), which Bedrock uses for every
//! streaming response: `ConverseStream` and `InvokeModelWithResponseStream`.
//!
//! Each message is
//!
//! ```text
//! total length (u32 BE) | headers length (u32 BE) | prelude CRC32
//! headers | payload | message CRC32
//! ```
//!
//! where both CRCs are CRC-32 (IEEE) — the prelude's over the first eight
//! bytes, the message's over everything before it. A header is a one-byte
//! name length, the name, a one-byte value type and the value. Feed the
//! response body to [`EventStreamDecoder::push`] chunk by chunk; complete
//! messages come out as they are framed, and a message split across chunks
//! is held until the rest arrives.
//!
//! Hand-rolled like [`crate::sigv4`], so the crate needs no AWS SDK.

use std::collections::HashMap;

use agent_framework_core::error::{Error, Result};

/// The smallest message: prelude (12) plus message CRC (4).
const MIN_MESSAGE_LEN: usize = 16;

/// A header value. Only the types Bedrock sends need to be inspected; the
/// rest are kept so a message still decodes.
#[derive(Debug, Clone, PartialEq)]
pub enum HeaderValue {
    Bool(bool),
    Byte(i8),
    Short(i16),
    Int(i32),
    Long(i64),
    Bytes(Vec<u8>),
    String(String),
    Timestamp(i64),
    Uuid([u8; 16]),
}

/// One decoded event-stream message.
#[derive(Debug, Clone, PartialEq)]
pub struct EventMessage {
    pub headers: HashMap<String, HeaderValue>,
    pub payload: Vec<u8>,
}

impl EventMessage {
    /// A string header's value, if present and a string.
    pub fn header_str(&self, name: &str) -> Option<&str> {
        match self.headers.get(name) {
            Some(HeaderValue::String(s)) => Some(s),
            _ => None,
        }
    }

    /// `:message-type` — `event`, `exception` or `error`.
    pub fn message_type(&self) -> Option<&str> {
        self.header_str(":message-type")
    }

    /// `:event-type` for an event message (e.g. `chunk`, `contentBlockDelta`).
    pub fn event_type(&self) -> Option<&str> {
        self.header_str(":event-type")
    }

    /// The error a non-`event` message carries, or `None` for an event.
    ///
    /// An `exception` names its type in `:exception-type` and explains in the
    /// payload's `message`; an `error` uses `:error-code` / `:error-message`.
    pub fn as_error(&self) -> Option<Error> {
        match self.message_type() {
            Some("event") | None => None,
            Some("exception") => {
                let kind = self.header_str(":exception-type").unwrap_or("exception");
                let detail = serde_json::from_slice::<serde_json::Value>(&self.payload)
                    .ok()
                    .and_then(|v| v.get("message").and_then(|m| m.as_str()).map(String::from))
                    .unwrap_or_else(|| String::from_utf8_lossy(&self.payload).into_owned());
                Some(Error::service(format!("Bedrock stream {kind}: {detail}")))
            }
            Some(other) => {
                let code = self.header_str(":error-code").unwrap_or(other);
                let msg = self.header_str(":error-message").unwrap_or_default();
                Some(Error::service(format!(
                    "Bedrock stream error {code}: {msg}"
                )))
            }
        }
    }
}

/// Incremental event-stream decoder. See the [module docs](self).
#[derive(Debug, Default)]
pub struct EventStreamDecoder {
    buf: Vec<u8>,
}

impl EventStreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `bytes` and return every message now complete, in order.
    ///
    /// # Errors
    /// A CRC mismatch or a malformed length/header. The stream cannot be
    /// resynchronised after one, so the caller should end it.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<EventMessage>> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        loop {
            if self.buf.len() < 12 {
                break;
            }
            let total = be_u32(&self.buf[0..4]) as usize;
            let headers_len = be_u32(&self.buf[4..8]) as usize;
            if total < MIN_MESSAGE_LEN || headers_len > total - MIN_MESSAGE_LEN {
                return Err(malformed(format!(
                    "lengths total={total} headers={headers_len}"
                )));
            }
            if crc32(&self.buf[0..8]) != be_u32(&self.buf[8..12]) {
                return Err(malformed("prelude CRC mismatch".into()));
            }
            if self.buf.len() < total {
                break;
            }
            let frame: Vec<u8> = self.buf.drain(..total).collect();
            if crc32(&frame[..total - 4]) != be_u32(&frame[total - 4..]) {
                return Err(malformed("message CRC mismatch".into()));
            }
            let headers = parse_headers(&frame[12..12 + headers_len])?;
            let payload = frame[12 + headers_len..total - 4].to_vec();
            out.push(EventMessage { headers, payload });
        }
        Ok(out)
    }

    /// Whether bytes of an incomplete message are still buffered — at the
    /// end of a body, a truncated final message.
    pub fn has_partial(&self) -> bool {
        !self.buf.is_empty()
    }
}

fn malformed(what: String) -> Error {
    Error::service(format!("malformed Bedrock event stream: {what}"))
}

fn be_u32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

fn parse_headers(mut b: &[u8]) -> Result<HashMap<String, HeaderValue>> {
    fn take<'a>(b: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
        if b.len() < n {
            return Err(malformed("truncated header".into()));
        }
        let (head, tail) = b.split_at(n);
        *b = tail;
        Ok(head)
    }
    let mut headers = HashMap::new();
    while !b.is_empty() {
        let name_len = take(&mut b, 1)?[0] as usize;
        let name = String::from_utf8_lossy(take(&mut b, name_len)?).into_owned();
        let value = match take(&mut b, 1)?[0] {
            0 => HeaderValue::Bool(true),
            1 => HeaderValue::Bool(false),
            2 => HeaderValue::Byte(take(&mut b, 1)?[0] as i8),
            3 => HeaderValue::Short(i16::from_be_bytes(take(&mut b, 2)?.try_into().unwrap())),
            4 => HeaderValue::Int(i32::from_be_bytes(take(&mut b, 4)?.try_into().unwrap())),
            5 => HeaderValue::Long(i64::from_be_bytes(take(&mut b, 8)?.try_into().unwrap())),
            6 => {
                let len = u16::from_be_bytes(take(&mut b, 2)?.try_into().unwrap()) as usize;
                HeaderValue::Bytes(take(&mut b, len)?.to_vec())
            }
            7 => {
                let len = u16::from_be_bytes(take(&mut b, 2)?.try_into().unwrap()) as usize;
                HeaderValue::String(String::from_utf8_lossy(take(&mut b, len)?).into_owned())
            }
            8 => HeaderValue::Timestamp(i64::from_be_bytes(take(&mut b, 8)?.try_into().unwrap())),
            9 => HeaderValue::Uuid(take(&mut b, 16)?.try_into().unwrap()),
            t => return Err(malformed(format!("unknown header value type {t}"))),
        };
        headers.insert(name, value);
    }
    Ok(headers)
}

/// Decode a streaming response body into its event-stream messages, as they
/// arrive. A transport error, a framing error, or a body that ends partway
/// through a message ends the stream with an error.
pub fn decode_body(
    resp: reqwest::Response,
) -> futures::stream::BoxStream<'static, Result<EventMessage>> {
    use futures::StreamExt;
    let bytes = resp.bytes_stream();
    futures::stream::unfold(
        (
            bytes,
            EventStreamDecoder::new(),
            std::collections::VecDeque::new(),
            false,
        ),
        |(mut bytes, mut decoder, mut queued, done)| async move {
            loop {
                if let Some(msg) = queued.pop_front() {
                    return Some((Ok(msg), (bytes, decoder, queued, done)));
                }
                if done {
                    return None;
                }
                match bytes.next().await {
                    Some(Ok(chunk)) => match decoder.push(&chunk) {
                        Ok(msgs) => queued.extend(msgs),
                        Err(e) => return Some((Err(e), (bytes, decoder, queued, true))),
                    },
                    Some(Err(e)) => {
                        let err = Error::service(format!(
                            "Bedrock stream failed: {}",
                            agent_framework_core::error::error_chain(&e)
                        ));
                        return Some((Err(err), (bytes, decoder, queued, true)));
                    }
                    None if decoder.has_partial() => {
                        let err = malformed("body ended inside a message".into());
                        return Some((Err(err), (bytes, decoder, queued, true)));
                    }
                    None => return None,
                }
            }
        },
    )
    .boxed()
}

/// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320).
fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, entry) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *entry = c;
        }
        t
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc = table[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// Encode one message — for tests in this crate and in crates that decode
/// Bedrock streams (a fake server has to produce real frames).
pub fn encode_message(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
    let mut hbytes = Vec::new();
    for (name, value) in headers {
        hbytes.push(name.len() as u8);
        hbytes.extend_from_slice(name.as_bytes());
        hbytes.push(7);
        hbytes.extend_from_slice(&(value.len() as u16).to_be_bytes());
        hbytes.extend_from_slice(value.as_bytes());
    }
    let total = 12 + hbytes.len() + payload.len() + 4;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&(total as u32).to_be_bytes());
    out.extend_from_slice(&(hbytes.len() as u32).to_be_bytes());
    let prelude_crc = crc32(&out);
    out.extend_from_slice(&prelude_crc.to_be_bytes());
    out.extend_from_slice(&hbytes);
    out.extend_from_slice(payload);
    let msg_crc = crc32(&out);
    out.extend_from_slice(&msg_crc.to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_standard_check_value() {
        // The CRC-32/ISO-HDLC check value for "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn decodes_a_known_aws_test_vector() {
        // From the AWS event-stream spec's test suite ("all_headers"-free
        // "int32_header" vector is longer; this is the minimal empty message):
        // total=16, headers=0, prelude CRC, message CRC.
        let msg = encode_message(&[], b"");
        assert_eq!(msg, hex("000000100000000005c248eb7d98c8ff"));
        let mut d = EventStreamDecoder::new();
        let out = d.push(&msg).unwrap();
        assert_eq!(
            out,
            vec![EventMessage {
                headers: HashMap::new(),
                payload: vec![]
            }]
        );
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn round_trips_headers_and_payload() {
        let msg = encode_message(
            &[(":message-type", "event"), (":event-type", "chunk")],
            br#"{"bytes":"e30="}"#,
        );
        let out = EventStreamDecoder::new().push(&msg).unwrap();
        assert_eq!(out[0].message_type(), Some("event"));
        assert_eq!(out[0].event_type(), Some("chunk"));
        assert_eq!(out[0].payload, br#"{"bytes":"e30="}"#);
        assert!(out[0].as_error().is_none());
    }

    #[test]
    fn a_message_split_across_chunks_is_held_until_complete() {
        let mut bytes = encode_message(&[(":event-type", "a")], b"one");
        bytes.extend(encode_message(&[(":event-type", "b")], b"two"));
        let mut d = EventStreamDecoder::new();
        let mut seen = Vec::new();
        for chunk in bytes.chunks(5) {
            seen.extend(d.push(chunk).unwrap());
        }
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[1].payload, b"two");
        assert!(!d.has_partial());
    }

    #[test]
    fn a_corrupted_message_is_rejected() {
        let mut msg = encode_message(&[(":event-type", "chunk")], b"payload");
        let last = msg.len() - 6;
        msg[last] ^= 0xFF;
        assert!(EventStreamDecoder::new().push(&msg).is_err());
    }

    #[test]
    fn an_exception_message_becomes_an_error() {
        let msg = encode_message(
            &[
                (":message-type", "exception"),
                (":exception-type", "throttlingException"),
            ],
            br#"{"message":"Too many requests"}"#,
        );
        let out = EventStreamDecoder::new().push(&msg).unwrap();
        let err = out[0].as_error().expect("an error").to_string();
        assert!(
            err.contains("throttlingException") && err.contains("Too many requests"),
            "{err}"
        );
    }
}
