//! [AWS Signature Version 4](https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html)
//! request signing for the Bedrock Runtime `Converse` API.
//!
//! No `chrono`/`time`/`hex` crate is used: hex encoding is a tiny
//! hand-rolled helper ([`hex_lower`]), and the `X-Amz-Date`/date-stamp
//! formatting ([`amz_dates_from_unix`]) is a hand-rolled implementation of
//! Howard Hinnant's `civil_from_days` algorithm over a `SystemTime`-derived
//! unix timestamp. Hashing/HMAC are the real primitives, from `sha2` and
//! `hmac`.
//!
//! The signing entry point, [`authorization_header`], is a *pure* function:
//! it takes the timestamp as an explicit [`SigV4Params::amz_date`] /
//! [`SigV4Params::date_stamp`] pair rather than reading the clock itself, so
//! it is fully deterministic and unit-testable (see the tests below, which
//! reproduce AWS's own published SigV4 test-suite vector). The live client
//! ([`crate::BedrockChatClient`]) is the only caller that reads the clock,
//! via [`amz_dates_from_unix`] fed from `SystemTime::now()`.

/// Lowercase hex-encode `bytes` (`hex` crate substitute).
pub fn hex_lower(bytes: &[u8]) -> String {
    const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX_DIGITS[(b >> 4) as usize] as char);
        out.push(HEX_DIGITS[(b & 0x0f) as usize] as char);
    }
    out
}

/// Lowercase hex SHA256 digest of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex_lower(&Sha256::digest(bytes))
}

/// HMAC-SHA256 of `msg` under `key`. `key` may be any length (HMAC pads or
/// hashes it internally), so this never fails.
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> Vec<u8> {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    let mut mac =
        Hmac::<Sha256>::new_from_slice(key).expect("HMAC-SHA256 accepts a key of any length");
    mac.update(msg);
    mac.finalize().into_bytes().to_vec()
}

/// Build the SigV4 canonical request string (steps 1-5 of the
/// [canonical-request algorithm](https://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html)).
///
/// `canonical_headers` must already be the sorted `name:value\n` block (one
/// line per signed header, including the trailing newline on the *last*
/// header too — that blank line before `signed_headers` is what separates
/// the two sections) and `signed_headers` the matching `;`-joined, sorted,
/// lowercase header name list.
fn canonical_request(
    method: &str,
    canonical_uri: &str,
    canonical_query: &str,
    canonical_headers: &str,
    signed_headers: &str,
    payload_hash: &str,
) -> String {
    format!(
        "{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )
}

/// Build the SigV4 string-to-sign from an already-hashed canonical request.
fn string_to_sign(amz_date: &str, credential_scope: &str, canonical_request_hash: &str) -> String {
    format!("AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{canonical_request_hash}")
}

/// Derive the final signing key: `HMAC(HMAC(HMAC(HMAC("AWS4"+secret,
/// date), region), service), "aws4_request")`.
fn derive_signing_key(secret_key: &str, date_stamp: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_sha256(
        format!("AWS4{secret_key}").as_bytes(),
        date_stamp.as_bytes(),
    );
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

/// The hex-encoded HMAC-SHA256 signature of `string_to_sign` under
/// `signing_key`.
fn sign(signing_key: &[u8], string_to_sign: &str) -> String {
    hex_lower(&hmac_sha256(signing_key, string_to_sign.as_bytes()))
}

/// Inputs to [`authorization_header`]. Every field is explicit — notably the
/// timestamp (`amz_date`/`date_stamp`) — so signing is a pure function of its
/// arguments and testable without touching the system clock.
pub struct SigV4Params<'a> {
    pub access_key: &'a str,
    pub secret_key: &'a str,
    /// Present when signing with temporary/STS credentials; adds the
    /// `x-amz-security-token` header to the signed set.
    pub session_token: Option<&'a str>,
    pub region: &'a str,
    pub service: &'a str,
    pub host: &'a str,
    pub method: &'a str,
    pub canonical_uri: &'a str,
    /// The canonical query string (already sorted-and-encoded `k=v&k=v`, or
    /// empty). Bedrock's `converse`/`converse-stream` calls take no query
    /// parameters, so callers in this crate always pass `""`.
    pub canonical_query: &'a str,
    pub payload: &'a [u8],
    /// `YYYYMMDDTHHMMSSZ`, e.g. `20150830T123600Z`.
    pub amz_date: &'a str,
    /// `YYYYMMDD`, e.g. `20150830` (the date portion of `amz_date`).
    pub date_stamp: &'a str,
}

/// Compute the `Authorization` header value for a SigV4-signed request, plus
/// the extra headers (`x-amz-date`, `x-amz-content-sha256`, and
/// `x-amz-security-token` when a session token is present) that must also be
/// set on the request — they are part of the signed header set, so the
/// server-side re-computation fails unless they are sent verbatim.
///
/// `host` is deliberately signed but NOT included in the returned header
/// list: callers set it implicitly by connecting to that host (an explicit
/// `Host` header is redundant with what an HTTP client already sends from
/// the request URL).
pub fn authorization_header(p: &SigV4Params) -> (String, Vec<(String, String)>) {
    let payload_hash = sha256_hex(p.payload);

    let mut headers: Vec<(String, String)> = vec![
        ("host".to_string(), p.host.to_string()),
        ("x-amz-content-sha256".to_string(), payload_hash.clone()),
        ("x-amz-date".to_string(), p.amz_date.to_string()),
    ];
    if let Some(token) = p.session_token {
        headers.push(("x-amz-security-token".to_string(), token.to_string()));
    }
    // SignedHeaders must be sorted lowercase header names; header names above
    // are already lowercase, so a plain lexicographic sort suffices.
    headers.sort_by(|a, b| a.0.cmp(&b.0));

    let signed_headers = headers
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let canonical_headers: String = headers.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();

    let creq = canonical_request(
        p.method,
        p.canonical_uri,
        p.canonical_query,
        &canonical_headers,
        &signed_headers,
        &payload_hash,
    );
    let creq_hash = sha256_hex(creq.as_bytes());

    let credential_scope = format!("{}/{}/{}/aws4_request", p.date_stamp, p.region, p.service);
    let sts = string_to_sign(p.amz_date, &credential_scope, &creq_hash);

    let signing_key = derive_signing_key(p.secret_key, p.date_stamp, p.region, p.service);
    let signature = sign(&signing_key, &sts);

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}",
        p.access_key
    );

    let mut extra_headers = vec![
        ("x-amz-date".to_string(), p.amz_date.to_string()),
        ("x-amz-content-sha256".to_string(), payload_hash),
    ];
    if let Some(token) = p.session_token {
        extra_headers.push(("x-amz-security-token".to_string(), token.to_string()));
    }
    (authorization, extra_headers)
}

/// Convert days-since-epoch (`1970-01-01` = `0`) to a proleptic-Gregorian
/// `(year, month, day)` triple.
///
/// This is Howard Hinnant's `civil_from_days` algorithm
/// (<http://howardhinnant.github.io/date_algorithms.html#civil_from_days>),
/// valid for the full `i32`-representable range and exact (no floating
/// point). `days` may be negative (dates before the epoch); this crate only
/// ever feeds it non-negative values derived from a unix timestamp, but the
/// algorithm is correct either way.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Convert a unix timestamp (seconds since `1970-01-01T00:00:00Z`) into the
/// `(amz_date, date_stamp)` pair SigV4 needs: `amz_date` is
/// `YYYYMMDDTHHMMSSZ` (the `X-Amz-Date` header value), `date_stamp` is its
/// `YYYYMMDD` date portion (used in the credential scope).
///
/// This is what [`crate::BedrockChatClient`] calls with
/// `SystemTime::now()`; [`authorization_header`] itself never touches the
/// clock, which is what keeps it unit-testable.
pub fn amz_dates_from_unix(secs: u64) -> (String, String) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = rem / 3600;
    let minute = (rem % 3600) / 60;
    let second = rem % 60;
    let (year, month, day) = civil_from_days(days);
    let amz_date = format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z");
    let date_stamp = format!("{year:04}{month:02}{day:02}");
    (amz_date, date_stamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    // region: amz_dates_from_unix

    #[test]
    fn amz_dates_from_unix_known_timestamp() {
        // 2015-08-30T12:36:00Z, the timestamp used by AWS's own published
        // SigV4 examples/test suite.
        assert_eq!(
            amz_dates_from_unix(1_440_938_160),
            ("20150830T123600Z".to_string(), "20150830".to_string())
        );
    }

    #[test]
    fn amz_dates_from_unix_epoch() {
        assert_eq!(
            amz_dates_from_unix(0),
            ("19700101T000000Z".to_string(), "19700101".to_string())
        );
    }

    #[test]
    fn amz_dates_from_unix_end_of_month_rollover() {
        // 2024-02-29T23:59:59Z (leap day, last second of the month) -> next
        // day is 2024-03-01, exercising the civil_from_days month/leap-year
        // boundary.
        let (amz_date, date_stamp) = amz_dates_from_unix(1_709_251_199);
        assert_eq!(amz_date, "20240229T235959Z");
        assert_eq!(date_stamp, "20240229");
    }

    // endregion

    // region: low-level primitives

    #[test]
    fn sha256_hex_of_empty_string() {
        // Well-known SHA256("") value, also the payload hash AWS's
        // `get-vanilla` test vector expects for a bodyless request.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn hex_lower_encodes_all_byte_values() {
        assert_eq!(hex_lower(&[0x00, 0x0f, 0xab, 0xff]), "000fabff");
    }

    // endregion

    // region: AWS SigV4 known-answer test vector ("get-vanilla")
    //
    // Reproduces AWS's own published SigV4 test-suite vector (credentials,
    // date, and expected canonical request / signature are all from AWS's
    // `aws4_testsuite`, `get-vanilla` case — a bare `GET /` with only `Host`
    // and `X-Amz-Date` signed and an empty body). Verified independently
    // against botocore's copy of the same fixture
    // (`tests/unit/auth/aws4_testsuite/get-vanilla/get-vanilla.{creq,authz}`)
    // before being hardcoded here, so this is an authoritative known-answer
    // test for `sha256_hex` + `hmac_sha256` + the canonical-request /
    // string-to-sign / signing-key derivation, independent of this crate's
    // own header choices in [`authorization_header`] (which additionally
    // signs `x-amz-content-sha256`, not exercised by this specific vector).

    const TEST_ACCESS_KEY: &str = "AKIDEXAMPLE";
    const TEST_SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
    const TEST_REGION: &str = "us-east-1";
    const TEST_SERVICE: &str = "service";
    const TEST_HOST: &str = "example.amazonaws.com";
    const TEST_AMZ_DATE: &str = "20150830T123600Z";
    const TEST_DATE_STAMP: &str = "20150830";

    #[test]
    fn sigv4_get_vanilla_canonical_request_matches_aws_fixture() {
        let canonical_headers = format!("host:{TEST_HOST}\nx-amz-date:{TEST_AMZ_DATE}\n");
        let payload_hash = sha256_hex(b"");
        let creq = canonical_request(
            "GET",
            "/",
            "",
            &canonical_headers,
            "host;x-amz-date",
            &payload_hash,
        );
        assert_eq!(
            creq,
            "GET\n/\n\nhost:example.amazonaws.com\nx-amz-date:20150830T123600Z\n\n\
             host;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sigv4_get_vanilla_signature_matches_aws_test_suite_vector() {
        let canonical_headers = format!("host:{TEST_HOST}\nx-amz-date:{TEST_AMZ_DATE}\n");
        let signed_headers = "host;x-amz-date";
        let payload_hash = sha256_hex(b"");

        let creq = canonical_request(
            "GET",
            "/",
            "",
            &canonical_headers,
            signed_headers,
            &payload_hash,
        );
        let creq_hash = sha256_hex(creq.as_bytes());
        let credential_scope =
            format!("{TEST_DATE_STAMP}/{TEST_REGION}/{TEST_SERVICE}/aws4_request");
        let sts = string_to_sign(TEST_AMZ_DATE, &credential_scope, &creq_hash);
        let signing_key =
            derive_signing_key(TEST_SECRET_KEY, TEST_DATE_STAMP, TEST_REGION, TEST_SERVICE);
        let signature = sign(&signing_key, &sts);

        // AWS `aws4_testsuite`/`get-vanilla.authz`:
        // "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request,
        //  SignedHeaders=host;x-amz-date,
        //  Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        assert_eq!(
            signature,
            "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );

        let expected_authz = format!(
            "AWS4-HMAC-SHA256 Credential={TEST_ACCESS_KEY}/{credential_scope}, \
             SignedHeaders={signed_headers}, Signature={signature}"
        );
        assert_eq!(
            expected_authz,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-date, \
             Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
    }

    // endregion

    // region: authorization_header (this crate's actual signed-header set)

    #[test]
    fn authorization_header_signs_host_content_sha256_and_date() {
        let params = SigV4Params {
            access_key: TEST_ACCESS_KEY,
            secret_key: TEST_SECRET_KEY,
            session_token: None,
            region: TEST_REGION,
            service: TEST_SERVICE,
            host: TEST_HOST,
            method: "GET",
            canonical_uri: "/",
            canonical_query: "",
            payload: b"",
            amz_date: TEST_AMZ_DATE,
            date_stamp: TEST_DATE_STAMP,
        };
        let (authorization, extra_headers) = authorization_header(&params);

        assert!(authorization.starts_with(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, "
        ));
        assert!(authorization.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date, "));
        assert!(authorization.contains("Signature="));

        let names: Vec<&str> = extra_headers.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(names, vec!["x-amz-date", "x-amz-content-sha256"]);
        assert_eq!(
            extra_headers
                .iter()
                .find(|(k, _)| k == "x-amz-date")
                .unwrap()
                .1,
            TEST_AMZ_DATE
        );
        assert_eq!(
            extra_headers
                .iter()
                .find(|(k, _)| k == "x-amz-content-sha256")
                .unwrap()
                .1,
            sha256_hex(b"")
        );
    }

    #[test]
    fn authorization_header_includes_session_token_when_present() {
        let params = SigV4Params {
            access_key: TEST_ACCESS_KEY,
            secret_key: TEST_SECRET_KEY,
            session_token: Some("test-session-token"),
            region: TEST_REGION,
            service: "bedrock",
            host: "bedrock-runtime.us-east-1.amazonaws.com",
            method: "POST",
            canonical_uri: "/model/anthropic.claude-3-haiku/converse",
            canonical_query: "",
            payload: b"{}",
            amz_date: TEST_AMZ_DATE,
            date_stamp: TEST_DATE_STAMP,
        };
        let (authorization, extra_headers) = authorization_header(&params);

        assert!(authorization
            .contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token, "));
        let names: Vec<&str> = extra_headers.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            names,
            vec!["x-amz-date", "x-amz-content-sha256", "x-amz-security-token"]
        );
        assert_eq!(
            extra_headers
                .iter()
                .find(|(k, _)| k == "x-amz-security-token")
                .unwrap()
                .1,
            "test-session-token"
        );
    }

    #[test]
    fn authorization_header_is_deterministic_given_explicit_timestamp() {
        // Same explicit timestamp in -> byte-identical signature out (no
        // hidden clock read).
        let make = || SigV4Params {
            access_key: TEST_ACCESS_KEY,
            secret_key: TEST_SECRET_KEY,
            session_token: None,
            region: TEST_REGION,
            service: TEST_SERVICE,
            host: TEST_HOST,
            method: "GET",
            canonical_uri: "/",
            canonical_query: "",
            payload: b"",
            amz_date: TEST_AMZ_DATE,
            date_stamp: TEST_DATE_STAMP,
        };
        let (a1, _) = authorization_header(&make());
        let (a2, _) = authorization_header(&make());
        assert_eq!(a1, a2);
    }

    // endregion
}
