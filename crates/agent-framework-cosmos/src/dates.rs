//! RFC 1123 ("HTTP-date") formatting, with no date/time crate dependency.
//!
//! Cosmos DB's REST API requires the `x-ms-date` header — and the identical
//! string embedded in the master-key auth signature payload (see
//! [`crate::auth`]) — in RFC 1123 form, e.g. `"Tue, 29 Mar 2016 02:28:29
//! GMT"` (RFC 7231 §7.1.1.1's `IMF-fixdate`). This crate's dependency
//! allowlist is limited to `hmac`/`sha2`/`base64` plus `reqwest`/workspace
//! deps — no `chrono`/`time` — so the two small algorithms needed
//! (Unix-days → proleptic-Gregorian civil date, and weekday-from-days) are
//! implemented directly. Both are the well-known, widely-ported
//! constant-time integer formulas from Howard Hinnant's public-domain
//! "chrono-Compatible Low-Level Date Algorithms"
//! (<https://howardhinnant.github.io/date_algorithms.html>), valid for the
//! entire range of `i64` Unix timestamps this crate will ever see.

const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Convert a day count `z` (days since the Unix epoch, 1970-01-01) into a
/// proleptic-Gregorian `(year, month, day)` triple (`month`/`day` are
/// 1-based). See the module docs for the algorithm's provenance.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Format `unix_seconds` (UTC) as an RFC 1123 HTTP-date, e.g.
/// `"Tue, 29 Mar 2016 02:28:29 GMT"`.
pub(crate) fn format_rfc1123(unix_seconds: i64) -> String {
    let days = unix_seconds.div_euclid(86_400);
    let secs_of_day = unix_seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;
    // The Unix epoch (days == 0) was a Thursday (index 4).
    let weekday = WEEKDAYS[(days + 4).rem_euclid(7) as usize];
    let month_name = MONTHS[(month - 1) as usize];
    format!("{weekday}, {day:02} {month_name} {year} {hour:02}:{minute:02}:{second:02} GMT")
}

/// The current wall-clock time as Unix seconds (UTC). A thin wrapper around
/// [`std::time::SystemTime`] so [`format_rfc1123`] itself stays a pure,
/// hermetically-testable function of an explicit timestamp.
pub(crate) fn now_unix_seconds() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_is_thursday_january_first_1970() {
        assert_eq!(format_rfc1123(0), "Thu, 01 Jan 1970 00:00:00 GMT");
    }

    #[test]
    fn matches_official_cosmos_docs_example() {
        // From https://learn.microsoft.com/en-us/rest/api/cosmos-db/query-documents
        // ("x-ms-date: Tue, 29 Mar 2016 02:28:32 GMT").
        // 1459218512 == 2016-03-29T02:28:32Z.
        assert_eq!(
            format_rfc1123(1_459_218_512),
            "Tue, 29 Mar 2016 02:28:32 GMT"
        );
    }

    #[test]
    fn handles_leap_day() {
        // 2000-02-29T00:00:00Z was a Tuesday.
        assert_eq!(format_rfc1123(951_782_400), "Tue, 29 Feb 2000 00:00:00 GMT");
    }

    #[test]
    fn handles_year_boundary() {
        // 2020-12-31T23:59:59Z / 2021-01-01T00:00:00Z.
        assert_eq!(
            format_rfc1123(1_609_459_199),
            "Thu, 31 Dec 2020 23:59:59 GMT"
        );
        assert_eq!(
            format_rfc1123(1_609_459_200),
            "Fri, 01 Jan 2021 00:00:00 GMT"
        );
    }

    #[test]
    fn pads_single_digit_day_and_time_components() {
        // 2023-01-01T01:02:03Z.
        assert_eq!(
            format_rfc1123(1_672_534_923),
            "Sun, 01 Jan 2023 01:02:03 GMT"
        );
    }

    #[test]
    fn now_unix_seconds_is_plausible() {
        // Sanity bound, not a real assertion about "now": must be after this
        // code was written and comfortably before any i64 overflow concern.
        let now = now_unix_seconds();
        assert!(
            now > 1_700_000_000,
            "now_unix_seconds looks too small: {now}"
        );
    }
}
