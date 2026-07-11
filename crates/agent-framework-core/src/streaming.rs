//! Utilities for decoding streamed bytes.

/// An incremental UTF-8 decoder for byte streams.
///
/// Network streams (SSE and friends) deliver bytes in arbitrary chunks, so a
/// multi-byte UTF-8 character can be split across two chunks. Decoding each
/// chunk independently with `String::from_utf8_lossy` corrupts such
/// characters: the truncated head becomes U+FFFD and the continuation bytes
/// in the next chunk become more U+FFFD. This decoder holds back an
/// incomplete trailing sequence until the rest of it arrives.
///
/// Genuinely invalid bytes (not a truncated tail) are replaced with U+FFFD,
/// matching lossy semantics.
#[derive(Debug, Default)]
pub struct Utf8StreamDecoder {
    pending: Vec<u8>,
}

impl Utf8StreamDecoder {
    /// Create an empty decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed `bytes`, returning all completely-decodable text.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // Safety of unwrap: `valid_up_to` guarantees validity.
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    match e.error_len() {
                        // Definitely-invalid sequence: replace and continue.
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + n);
                        }
                        // Incomplete trailing sequence: keep it for the next
                        // chunk.
                        None => {
                            self.pending.drain(..valid);
                            break;
                        }
                    }
                }
            }
        }
        out
    }

    /// Drain any held-back bytes at end of stream (lossy).
    pub fn flush(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_ascii_through() {
        let mut d = Utf8StreamDecoder::new();
        assert_eq!(d.push(b"hello"), "hello");
        assert_eq!(d.flush(), "");
    }

    #[test]
    fn reassembles_multibyte_char_split_across_chunks() {
        let mut d = Utf8StreamDecoder::new();
        let euro = "€".as_bytes(); // 3 bytes: E2 82 AC
        assert_eq!(d.push(&euro[..1]), "");
        assert_eq!(d.push(&euro[1..2]), "");
        assert_eq!(d.push(&euro[2..]), "€");
    }

    #[test]
    fn mixed_text_and_split_emoji() {
        let mut d = Utf8StreamDecoder::new();
        let s = "ok 🚀 done".as_bytes();
        let cut = 5; // inside the 4-byte emoji (starts at index 3)
        let first = d.push(&s[..cut]);
        let second = d.push(&s[cut..]);
        assert_eq!(format!("{first}{second}"), "ok 🚀 done");
    }

    #[test]
    fn replaces_genuinely_invalid_bytes() {
        let mut d = Utf8StreamDecoder::new();
        assert_eq!(d.push(&[b'a', 0xFF, b'b']), "a\u{FFFD}b");
    }

    #[test]
    fn flush_lossily_drains_incomplete_tail() {
        let mut d = Utf8StreamDecoder::new();
        assert_eq!(d.push("€".as_bytes()[..2].as_ref()), "");
        assert_eq!(d.flush(), "\u{FFFD}");
    }
}
