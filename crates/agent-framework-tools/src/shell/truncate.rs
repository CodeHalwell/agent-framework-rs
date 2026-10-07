//! Head/tail byte truncation shared by every executor.
//!
//! [`truncate_head_tail`] keeps the first `cap / 2` bytes and the last
//! `cap - cap / 2`, joined by a marker naming how many bytes were dropped.
//! [`HeadTailBuffer`] produces the same text while streaming, holding at most
//! `cap` bytes, so a command that prints gigabytes cannot exhaust memory
//! before its timeout fires.

use std::collections::VecDeque;

/// Truncate `data` to `cap` bytes, keeping a head and a tail slice.
///
/// Returns the decoded text (invalid UTF-8 is replaced) and whether anything
/// was dropped. A `cap` of zero is clamped to one: there is no consistent
/// meaning for "keep nothing", and treating it as unlimited would defeat
/// the limit. Upstream raises instead; the builders here reject zero up
/// front.
pub fn truncate_head_tail(data: &[u8], cap: usize) -> (String, bool) {
    let cap = cap.max(1);
    if data.len() <= cap {
        return (String::from_utf8_lossy(data).into_owned(), false);
    }
    let head_cap = cap / 2;
    let tail_cap = cap - head_cap;
    let head = String::from_utf8_lossy(&data[..head_cap]);
    let tail = String::from_utf8_lossy(&data[data.len() - tail_cap..]);
    let dropped = data.len() - cap;
    (
        format!("{head}\n[... truncated {dropped} bytes ...]\n{tail}"),
        true,
    )
}

/// [`truncate_head_tail`] for already-decoded text, budgeted in UTF-8 bytes.
pub fn truncate_text_head_tail(text: &str, cap: usize) -> (String, bool) {
    truncate_head_tail(text.as_bytes(), cap)
}

/// A bounded byte sink whose [`finish`](Self::finish) equals
/// [`truncate_head_tail`] over everything written to it.
#[derive(Debug)]
pub(crate) struct HeadTailBuffer {
    cap: usize,
    head: Vec<u8>,
    tail: VecDeque<u8>,
    total: usize,
}

impl HeadTailBuffer {
    pub(crate) fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        Self {
            cap,
            head: Vec::new(),
            tail: VecDeque::new(),
            total: 0,
        }
    }

    pub(crate) fn push(&mut self, mut chunk: &[u8]) {
        self.total += chunk.len();
        let head_cap = self.cap / 2;
        // Until the cap is reached nothing is dropped, so keep everything in
        // order: the head fills first, the rest queues in the tail.
        if self.head.len() < head_cap {
            let take = (head_cap - self.head.len()).min(chunk.len());
            self.head.extend_from_slice(&chunk[..take]);
            chunk = &chunk[take..];
        }
        let tail_cap = self.cap - head_cap;
        self.tail.extend(chunk);
        while self.tail.len() > tail_cap {
            self.tail.pop_front();
        }
    }

    pub(crate) fn finish(self) -> (String, bool) {
        if self.total <= self.cap {
            let mut all = self.head;
            all.extend(self.tail);
            return (String::from_utf8_lossy(&all).into_owned(), false);
        }
        let tail: Vec<u8> = self.tail.into_iter().collect();
        let dropped = self.total - self.cap;
        (
            format!(
                "{}\n[... truncated {dropped} bytes ...]\n{}",
                String::from_utf8_lossy(&self.head),
                String::from_utf8_lossy(&tail)
            ),
            true,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn under_cap_returns_original() {
        assert_eq!(truncate_head_tail(b"hello", 100), ("hello".into(), false));
    }

    #[test]
    fn at_cap_returns_original() {
        assert_eq!(truncate_head_tail(b"abcde", 5), ("abcde".into(), false));
    }

    #[test]
    fn over_cap_marks_truncated_and_reports_bytes() {
        let (out, truncated) = truncate_head_tail(&[b'A'; 10], 4);
        assert!(truncated);
        assert!(out.contains("truncated 6 bytes"));
        assert_eq!(out.matches('A').count(), 4);
    }

    #[test]
    fn odd_cap_keeps_extra_byte_in_tail() {
        let (out, truncated) = truncate_head_tail(b"ABCDEFGHIJ", 5);
        assert!(truncated);
        assert!(out.starts_with("AB\n["));
        assert!(out.ends_with("]\nHIJ"));
    }

    #[test]
    fn text_uses_utf8_byte_budget() {
        let (out, truncated) = truncate_text_head_tail(&"😀".repeat(10), 20);
        assert!(truncated);
        assert!(out.contains("truncated 20 bytes"));
    }

    #[test]
    fn zero_cap_is_clamped_not_unlimited() {
        let (out, truncated) = truncate_head_tail(b"abc", 0);
        assert!(truncated);
        assert!(out.contains("truncated 2 bytes"));
    }

    #[test]
    fn streaming_buffer_matches_one_shot_truncation() {
        let data: Vec<u8> = (0..=255u8).cycle().take(5000).collect();
        for cap in [1usize, 2, 7, 64, 4999, 5000, 6000] {
            for chunk in [1usize, 3, 100, 5000] {
                let mut buf = HeadTailBuffer::new(cap);
                for piece in data.chunks(chunk) {
                    buf.push(piece);
                }
                assert_eq!(
                    buf.finish(),
                    truncate_head_tail(&data, cap),
                    "cap {cap} chunk {chunk}"
                );
            }
        }
    }
}
