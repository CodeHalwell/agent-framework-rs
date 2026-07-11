//! Cosmos DB REST API master-key request signing.
//!
//! Implements the algorithm documented at
//! <https://learn.microsoft.com/en-us/rest/api/cosmos-db/access-control-on-cosmosdb-resources>:
//! for a request with HTTP verb `V`, resource type `T` (`dbs`/`colls`/
//! `docs`/...), resource link `L`, and `x-ms-date` value `D`, the canonical
//! string `"{v}\n{t}\n{l}\n{d}\n\n"` — verb/resource-type/date lowercased;
//! the resource link's case is preserved verbatim, since resource ids are
//! case-sensitive — is HMAC-SHA256'd with the base64-decoded master key,
//! the digest is base64-encoded, and substituted into
//! `"type=master&ver=1.0&sig={signature}"`. The **entire** resulting string
//! (not just the signature) is then percent-encoded to produce the
//! `Authorization` header value.
//!
//! Only master-key authentication is implemented. Entra ID / AAD
//! (`TokenCredential`-based) authentication, which the .NET
//! `Microsoft.Agents.AI.CosmosNoSql` package also supports, is **not**
//! ported — see the crate root docs and `PARITY.md`.

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use hmac::{Hmac, Mac};
use sha2::Sha256;

use agent_framework_core::error::{Error, Result};

type HmacSha256 = Hmac<Sha256>;

/// Base64-decode a Cosmos DB master/primary key (as copy-pasted from the
/// Azure portal) into the raw key bytes used for HMAC signing. Decoded once
/// at [`crate::client::CosmosRestClient`] construction rather than per
/// request.
pub(crate) fn decode_master_key(key: &str) -> Result<Vec<u8>> {
    BASE64.decode(key.trim()).map_err(|e| {
        Error::Configuration(format!(
            "invalid Cosmos DB master key: not valid base64: {e}"
        ))
    })
}

/// Compute the canonical signing payload: `"{verb}\n{resource_type}\n{resource_link}\n{date}\n\n"`,
/// with `verb`/`resource_type`/`date` lowercased and `resource_link`
/// preserved verbatim. Split out from [`authorization_header`] purely so
/// tests can assert on the exact string being signed, independent of the
/// HMAC/base64/percent-encoding steps.
fn signing_payload(verb: &str, resource_type: &str, resource_link: &str, date: &str) -> String {
    format!(
        "{}\n{}\n{}\n{}\n\n",
        verb.to_lowercase(),
        resource_type.to_lowercase(),
        resource_link,
        date.to_lowercase(),
    )
}

/// Percent-encode `input`, leaving the RFC 3986 "unreserved" characters
/// (`A-Z a-z 0-9 - _ . ~`) untouched and escaping every other byte as
/// `%XX` (uppercase hex). Cosmos DB expects the *entire*
/// `type=master&ver=1.0&sig=...` authorization string encoded this way
/// (not just the `sig` component) before being sent as the `Authorization`
/// header value — confirmed against the reference C#/Node.js examples in
/// Microsoft's docs, which wrap the whole string in
/// `WebUtility.UrlEncode`/`encodeURIComponent`.
fn percent_encode(input: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push(HEX[(byte >> 4) as usize] as char);
                out.push(HEX[(byte & 0x0f) as usize] as char);
            }
        }
    }
    out
}

/// Compute the `Authorization` header value for one Cosmos DB REST request,
/// signed with a master key. `master_key_bytes` is the already
/// base64-decoded key (see [`decode_master_key`]); `date` must be the same
/// RFC 1123 string also sent as the `x-ms-date` header.
pub(crate) fn authorization_header(
    verb: &str,
    resource_type: &str,
    resource_link: &str,
    date: &str,
    master_key_bytes: &[u8],
) -> Result<String> {
    let payload = signing_payload(verb, resource_type, resource_link, date);

    let mut mac = HmacSha256::new_from_slice(master_key_bytes)
        .map_err(|e| Error::Configuration(format!("invalid Cosmos DB master key: {e}")))?;
    mac.update(payload.as_bytes());
    let signature = BASE64.encode(mac.finalize().into_bytes());

    let raw = format!("type=master&ver=1.0&sig={signature}");
    Ok(percent_encode(&raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Well-known, intentionally-public Cosmos DB Emulator default key (also
    // hardcoded in Microsoft's own `Microsoft.Agents.AI.CosmosNoSql`
    // reference test suite) — used here purely as a fixed test key, not a
    // real secret.
    const EMULATOR_KEY: &str =
        "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==";

    // region: signing_payload (pure)

    #[test]
    fn signing_payload_lowercases_verb_type_and_date_but_not_resource_link() {
        let payload = signing_payload(
            "POST",
            "DOCS",
            "dbs/MyDb/colls/MyColl",
            "Tue, 29 Mar 2016 02:28:29 GMT",
        );
        assert_eq!(
            payload,
            "post\ndocs\ndbs/MyDb/colls/MyColl\ntue, 29 mar 2016 02:28:29 gmt\n\n"
        );
    }

    #[test]
    fn signing_payload_allows_empty_resource_link() {
        // Create Database's resourceLink is the empty string.
        let payload = signing_payload("post", "dbs", "", "Fri, 08 Apr 2015 03:52:31 GMT");
        assert_eq!(payload, "post\ndbs\n\nfri, 08 apr 2015 03:52:31 gmt\n\n");
    }

    // endregion

    // region: percent_encode (pure)

    #[test]
    fn percent_encode_leaves_unreserved_characters_untouched() {
        assert_eq!(percent_encode("abcXYZ019-_.~"), "abcXYZ019-_.~".to_string());
    }

    #[test]
    fn percent_encode_escapes_base64_alphabet_specials() {
        assert_eq!(percent_encode("+/="), "%2B%2F%3D");
    }

    #[test]
    fn percent_encode_escapes_ampersand_and_equals_in_kv_string() {
        assert_eq!(
            percent_encode("type=master&ver=1.0"),
            "type%3Dmaster%26ver%3D1.0"
        );
    }

    // endregion

    // region: authorization_header — exact signature test vectors
    //
    // The first vector is copied verbatim from Microsoft's own published
    // example at
    // https://learn.microsoft.com/en-us/rest/api/cosmos-db/access-control-on-cosmosdb-resources
    // ("Example Encoding" table) — an authoritative ground truth, not just a
    // self-consistency check. The rest are additional vectors covering the
    // other verbs/resource types this crate actually issues, independently
    // computed (Python `hmac`/`hashlib`/`base64`, matching this file's
    // algorithm line-for-line) against the same well-known emulator key so
    // a regression in any of `dbs`/`colls`/`docs`, GET/POST/DELETE, or an
    // empty resource link would be caught.

    #[test]
    fn matches_official_microsoft_docs_example() {
        let key = decode_master_key(
            "dsZQi3KtZmCv1ljt3VNWNm7sQUF1y5rJfC6kv5JiwvW0EndXdDku/dkKBp8/ufDToSxLzR4y+O/0H/t4bQtVNw==",
        )
        .unwrap();
        let header = authorization_header(
            "GET",
            "dbs",
            "dbs/ToDoList",
            "Thu, 27 Apr 2017 00:51:12 GMT",
            &key,
        )
        .unwrap();
        assert_eq!(
            header,
            "type%3Dmaster%26ver%3D1.0%26sig%3Dc09PEVJrgp2uQRkr934kFbTqhByc7TVr3OHyqlu%2Bc%2Bc%3D"
        );
    }

    #[test]
    fn signature_vector_post_create_document() {
        let key = decode_master_key(EMULATOR_KEY).unwrap();
        let header = authorization_header(
            "post",
            "docs",
            "dbs/testdb/colls/testcoll",
            "Tue, 29 Mar 2016 02:28:29 GMT",
            &key,
        )
        .unwrap();
        assert_eq!(
            header,
            "type%3Dmaster%26ver%3D1.0%26sig%3DPyKtcMcAgNBOMAVmuYu6rVetvqMktcOpQvnLwOVdxSQ%3D"
        );
    }

    #[test]
    fn signature_vector_delete_document() {
        let key = decode_master_key(EMULATOR_KEY).unwrap();
        let header = authorization_header(
            "delete",
            "docs",
            "dbs/testdb/colls/testcoll/docs/msg-1",
            "Wed, 30 Mar 2016 10:00:00 GMT",
            &key,
        )
        .unwrap();
        assert_eq!(
            header,
            "type%3Dmaster%26ver%3D1.0%26sig%3D2sY%2Flq9w4SbHARZJW9t%2BHs4Pjm1R7cgz83DFDwox8Hc%3D"
        );
    }

    #[test]
    fn signature_vector_post_create_database_has_empty_resource_link() {
        let key = decode_master_key(EMULATOR_KEY).unwrap();
        let header =
            authorization_header("post", "dbs", "", "Fri, 08 Apr 2015 03:52:31 GMT", &key).unwrap();
        assert_eq!(
            header,
            "type%3Dmaster%26ver%3D1.0%26sig%3DSClx71uG6Jjxhn09aTkuVPB3bcrPzwB8UGUn7Mb%2B8iY%3D"
        );
    }

    #[test]
    fn signature_vector_post_create_container() {
        let key = decode_master_key(EMULATOR_KEY).unwrap();
        let header = authorization_header(
            "post",
            "colls",
            "dbs/testdb",
            "Fri, 08 Apr 2015 03:52:31 GMT",
            &key,
        )
        .unwrap();
        assert_eq!(
            header,
            "type%3Dmaster%26ver%3D1.0%26sig%3DL3m60OT87NOuqRD8UmMp31%2BaeYiGGoXiK6qebyUx3B4%3D"
        );
    }

    #[test]
    fn authorization_header_is_deterministic_for_same_inputs() {
        let key = decode_master_key(EMULATOR_KEY).unwrap();
        let a = authorization_header("get", "docs", "dbs/d/colls/c", "date", &key).unwrap();
        let b = authorization_header("get", "docs", "dbs/d/colls/c", "date", &key).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn authorization_header_changes_with_verb() {
        let key = decode_master_key(EMULATOR_KEY).unwrap();
        let get = authorization_header("get", "docs", "dbs/d/colls/c", "date", &key).unwrap();
        let post = authorization_header("post", "docs", "dbs/d/colls/c", "date", &key).unwrap();
        assert_ne!(get, post);
    }

    // endregion

    // region: decode_master_key

    #[test]
    fn decode_master_key_accepts_valid_base64() {
        assert!(decode_master_key(EMULATOR_KEY).is_ok());
    }

    #[test]
    fn decode_master_key_rejects_invalid_base64() {
        let err = decode_master_key("not base64!!! ***").unwrap_err();
        assert!(err.to_string().contains("master key"));
    }

    // endregion
}
