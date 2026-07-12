//! A minimal Cosmos DB NoSQL (SQL API) REST client: just enough of the
//! `dbs`/`colls`/`docs` resource surface for
//! [`crate::CosmosChatMessageStore`] — create-if-missing database/container,
//! create/query/delete documents — signed with a master key (see
//! [`crate::auth`]).
//!
//! Every request carries `x-ms-version: 2018-12-31` (the version that
//! requires an explicit `partitionKey` on container creation, matching this
//! crate's always-partitioned containers) and the master-key
//! `Authorization`/`x-ms-date` headers computed per request (the signed
//! date must match the header exactly).

use serde_json::Value;

use agent_framework_core::error::{Error, Result};

use crate::auth::{authorization_header, decode_master_key};
use crate::dates::{format_rfc1123, now_unix_seconds};

/// Cosmos DB REST API version this crate speaks. `2018-12-31` is the first
/// version that *requires* a `partitionKey` on `Create Collection` — this
/// crate never creates a legacy non-partitioned container, so there's no
/// reason to support anything older.
pub const DEFAULT_API_VERSION: &str = "2018-12-31";

/// Partition key path used for every container this crate creates —
/// `threadId`, matching [`crate::CosmosChatMessageStore`]'s partitioning
/// (one partition per conversation thread).
pub(crate) const PARTITION_KEY_PATH: &str = "/threadId";

fn db_link(database_id: &str) -> String {
    format!("dbs/{database_id}")
}

fn coll_link(database_id: &str, container_id: &str) -> String {
    format!("dbs/{database_id}/colls/{container_id}")
}

fn docs_link(database_id: &str, container_id: &str) -> String {
    format!("{}/docs", coll_link(database_id, container_id))
}

fn doc_link(database_id: &str, container_id: &str, doc_id: &str) -> String {
    format!("{}/{doc_id}", docs_link(database_id, container_id))
}

/// The `x-ms-documentdb-partitionkey` header value for a single-value
/// (non-hierarchical) partition key: a JSON array containing just that one
/// value, e.g. `["thread-42"]`. `serde_json::to_string` both quotes and
/// escapes the value correctly for any thread id, including one containing
/// `"` or other characters that would otherwise corrupt the header.
pub(crate) fn partition_key_header_value(partition_key: &str) -> Result<String> {
    Ok(serde_json::to_string(&[partition_key])?)
}

/// Build the `POST .../docs` query-request body:
/// `{"query": ..., "parameters": [{"name": ..., "value": ...}, ...]}`.
pub(crate) fn build_query_body(query: &str, parameters: &[(&str, Value)]) -> Value {
    let params: Vec<Value> = parameters
        .iter()
        .map(|(name, value)| serde_json::json!({ "name": name, "value": value }))
        .collect();
    serde_json::json!({ "query": query, "parameters": params })
}

/// Extract the `Documents` array from a query-response body
/// (`{"_rid": ..., "Documents": [...], "_count": N}`), tolerating a missing
/// or non-array field by returning an empty list rather than erroring —
/// this is purely a defensive fallback for a response shape that
/// [`CosmosRestClient::query_documents`] otherwise always sends a
/// well-formed query to produce.
pub(crate) fn parse_query_response(body: &Value) -> Vec<Value> {
    body.get("Documents")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn map_error_response(status: reqwest::StatusCode, body: &str) -> Error {
    Error::service(format!("Cosmos DB API error {status}: {body}"))
}

/// Everything [`CosmosRestClient::send`] needs for one request, bundled into
/// a struct (rather than half a dozen positional parameters) purely to keep
/// clippy's `too_many_arguments` happy — this has no behavior of its own.
struct RequestSpec<'a> {
    method: reqwest::Method,
    /// Feeds the auth signature (see [`crate::auth`]) — RediSearch's
    /// `dbs`/`colls`/`docs`.
    resource_type: &'a str,
    /// Feeds the auth signature; the resource (or, for list/create
    /// operations, its parent) this request addresses.
    resource_link: &'a str,
    /// Request URI path relative to the account endpoint. Usually equal to
    /// `resource_link`, except where the REST API distinguishes a resource
    /// from its "list/create children at this path" endpoint (e.g.
    /// `dbs/{db}/colls` vs. the `dbs/{db}` resource link used to sign it).
    url_path: &'a str,
    body: Option<&'a Value>,
    /// Defaults to `application/json` when `body` is `Some` and this is
    /// `None`.
    content_type: Option<&'a str>,
    extra_headers: &'a [(&'a str, String)],
}

/// Low-level Cosmos DB REST client: owns the account endpoint, master key,
/// and `reqwest::Client`. Crate-private plumbing — [`crate::CosmosChatMessageStore`]
/// is the public surface.
pub(crate) struct CosmosRestClient {
    http: reqwest::Client,
    account_endpoint: String,
    master_key: String,
    master_key_bytes: Vec<u8>,
    api_version: String,
}

impl CosmosRestClient {
    /// `account_endpoint` is the Cosmos DB account URI, e.g.
    /// `https://my-account.documents.azure.com:443/` (a trailing slash, if
    /// present, is trimmed). `key` is the base64-encoded master/primary key
    /// from the Azure portal; it's decoded once here (fails fast on
    /// malformed input) rather than on every request.
    pub(crate) fn new(account_endpoint: impl Into<String>, key: impl Into<String>) -> Result<Self> {
        let account_endpoint = account_endpoint.into();
        let trimmed = account_endpoint.trim_end_matches('/').to_string();
        if trimmed.is_empty() {
            return Err(Error::Configuration(
                "Cosmos DB account_endpoint must not be empty".into(),
            ));
        }
        let master_key = key.into();
        let master_key_bytes = decode_master_key(&master_key)?;
        Ok(Self {
            http: reqwest::Client::new(),
            account_endpoint: trimmed,
            master_key,
            master_key_bytes,
            api_version: DEFAULT_API_VERSION.to_string(),
        })
    }

    pub(crate) fn account_endpoint(&self) -> &str {
        &self.account_endpoint
    }

    pub(crate) fn master_key(&self) -> &str {
        &self.master_key
    }

    /// Issue one signed HTTP request per `spec`.
    async fn send(&self, spec: RequestSpec<'_>) -> Result<reqwest::Response> {
        let date = format_rfc1123(now_unix_seconds());
        let auth = authorization_header(
            spec.method.as_str(),
            spec.resource_type,
            spec.resource_link,
            &date,
            &self.master_key_bytes,
        )?;

        let url = format!("{}/{}", self.account_endpoint, spec.url_path);
        let mut builder = self
            .http
            .request(spec.method, url)
            .header("Authorization", auth)
            .header("x-ms-date", date)
            .header("x-ms-version", self.api_version.as_str())
            .header("Accept", "application/json");

        for (name, value) in spec.extra_headers {
            builder = builder.header(*name, value.as_str());
        }

        if let Some(b) = spec.body {
            let ct = spec.content_type.unwrap_or("application/json");
            builder = builder
                .header("Content-Type", ct)
                .body(serde_json::to_vec(b)?);
        }

        builder
            .send()
            .await
            .map_err(|e| Error::service(format!("Cosmos DB request failed: {e}")))
    }

    /// `POST /dbs` with `{"id": database_id}`. Tolerates `409 Conflict`
    /// ("already exists") as success.
    pub(crate) async fn create_database_if_not_exists(&self, database_id: &str) -> Result<()> {
        let body = serde_json::json!({ "id": database_id });
        let resp = self
            .send(RequestSpec {
                method: reqwest::Method::POST,
                resource_type: "dbs",
                resource_link: "",
                url_path: "dbs",
                body: Some(&body),
                content_type: None,
                extra_headers: &[],
            })
            .await?;
        let status = resp.status();
        if status.is_success() || status == reqwest::StatusCode::CONFLICT {
            Ok(())
        } else {
            let text = resp.text().await.unwrap_or_default();
            Err(map_error_response(status, &text))
        }
    }

    /// `POST /dbs/{database_id}/colls` with `{"id": container_id,
    /// "partitionKey": {"paths": [partition_key_path], "kind": "Hash"}}`.
    /// Tolerates `409 Conflict` as success. `partition_key_path` lets each
    /// store pick its own partitioning scheme (e.g.
    /// [`crate::CosmosChatMessageStore`]'s `/threadId` vs.
    /// [`crate::checkpoint_storage::CosmosCheckpointStorage`]'s `/id`)
    /// rather than hard-coding one path for every container this crate
    /// creates.
    pub(crate) async fn create_container_if_not_exists(
        &self,
        database_id: &str,
        container_id: &str,
        partition_key_path: &str,
    ) -> Result<()> {
        let body = serde_json::json!({
            "id": container_id,
            "partitionKey": {
                "paths": [partition_key_path],
                "kind": "Hash",
            },
        });
        let resource_link = db_link(database_id);
        let url_path = format!("{resource_link}/colls");
        let resp = self
            .send(RequestSpec {
                method: reqwest::Method::POST,
                resource_type: "colls",
                resource_link: &resource_link,
                url_path: &url_path,
                body: Some(&body),
                content_type: None,
                extra_headers: &[],
            })
            .await?;
        let status = resp.status();
        if status.is_success() || status == reqwest::StatusCode::CONFLICT {
            Ok(())
        } else {
            let text = resp.text().await.unwrap_or_default();
            Err(map_error_response(status, &text))
        }
    }

    /// `POST /dbs/{db}/colls/{coll}/docs` with `document` as the body,
    /// scoped to partition `partition_key`. Returns the created document
    /// (as returned by Cosmos DB, including its system-generated
    /// `_rid`/`_ts`/... properties).
    pub(crate) async fn create_document(
        &self,
        database_id: &str,
        container_id: &str,
        partition_key: &str,
        document: &Value,
    ) -> Result<Value> {
        let resource_link = coll_link(database_id, container_id);
        let url_path = docs_link(database_id, container_id);
        let pk_header = partition_key_header_value(partition_key)?;
        let resp = self
            .send(RequestSpec {
                method: reqwest::Method::POST,
                resource_type: "docs",
                resource_link: &resource_link,
                url_path: &url_path,
                body: Some(document),
                content_type: None,
                extra_headers: &[("x-ms-documentdb-partitionkey", pk_header)],
            })
            .await?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::service(format!("failed reading Cosmos DB response body: {e}")))?;
        if !status.is_success() {
            return Err(map_error_response(status, &text));
        }
        serde_json::from_str(&text).map_err(|e| {
            Error::service(format!(
                "invalid Cosmos DB response JSON: {e} (body: {text})"
            ))
        })
    }

    /// Run a parameterized SQL query against `dbs/{db}/colls/{coll}/docs`,
    /// scoped to partition `partition_key` (single-partition query — no
    /// cross-partition fan-out header is needed or sent). Transparently
    /// follows `x-ms-continuation` pagination until exhausted, returning
    /// every matched document's raw JSON.
    pub(crate) async fn query_documents(
        &self,
        database_id: &str,
        container_id: &str,
        partition_key: &str,
        query: &str,
        parameters: &[(&str, Value)],
    ) -> Result<Vec<Value>> {
        let pk_header = partition_key_header_value(partition_key)?;
        self.query_documents_with_headers(
            database_id,
            container_id,
            query,
            parameters,
            vec![
                ("x-ms-documentdb-isquery", "True".to_string()),
                ("x-ms-documentdb-partitionkey", pk_header),
            ],
        )
        .await
    }

    /// Run a parameterized SQL query against `dbs/{db}/colls/{coll}/docs`
    /// fanned out across *every* partition (`x-ms-documentdb-
    /// query-enablecrosspartition: True`, no partition-key header) — for
    /// stores like [`crate::checkpoint_storage::CosmosCheckpointStorage`]
    /// whose partition key (`/id`) isn't a useful query-time filter.
    /// Transparently follows `x-ms-continuation` pagination until
    /// exhausted, same as [`Self::query_documents`].
    pub(crate) async fn query_documents_cross_partition(
        &self,
        database_id: &str,
        container_id: &str,
        query: &str,
        parameters: &[(&str, Value)],
    ) -> Result<Vec<Value>> {
        self.query_documents_with_headers(
            database_id,
            container_id,
            query,
            parameters,
            vec![
                ("x-ms-documentdb-isquery", "True".to_string()),
                (
                    "x-ms-documentdb-query-enablecrosspartition",
                    "True".to_string(),
                ),
            ],
        )
        .await
    }

    /// Shared pagination loop for [`Self::query_documents`] and
    /// [`Self::query_documents_cross_partition`]; `base_headers` supplies
    /// whichever of the two mutually-exclusive scoping headers (partition
    /// key vs. cross-partition) the caller wants, plus `x-ms-documentdb-
    /// isquery`. `x-ms-continuation` is appended per-iteration.
    async fn query_documents_with_headers(
        &self,
        database_id: &str,
        container_id: &str,
        query: &str,
        parameters: &[(&str, Value)],
        base_headers: Vec<(&str, String)>,
    ) -> Result<Vec<Value>> {
        let resource_link = coll_link(database_id, container_id);
        let url_path = docs_link(database_id, container_id);
        let body = build_query_body(query, parameters);

        let mut out = Vec::new();
        let mut continuation: Option<String> = None;
        loop {
            let mut headers = base_headers.clone();
            if let Some(token) = &continuation {
                headers.push(("x-ms-continuation", token.clone()));
            }

            let resp = self
                .send(RequestSpec {
                    method: reqwest::Method::POST,
                    resource_type: "docs",
                    resource_link: &resource_link,
                    url_path: &url_path,
                    body: Some(&body),
                    content_type: Some("application/query+json"),
                    extra_headers: &headers,
                })
                .await?;
            let status = resp.status();
            continuation = resp
                .headers()
                .get("x-ms-continuation")
                .and_then(|v| v.to_str().ok())
                .map(String::from);
            let text = resp.text().await.map_err(|e| {
                Error::service(format!("failed reading Cosmos DB response body: {e}"))
            })?;
            if !status.is_success() {
                return Err(map_error_response(status, &text));
            }
            let value: Value = serde_json::from_str(&text).map_err(|e| {
                Error::service(format!(
                    "invalid Cosmos DB response JSON: {e} (body: {text})"
                ))
            })?;
            out.extend(parse_query_response(&value));

            if continuation.is_none() {
                break;
            }
        }
        Ok(out)
    }

    /// `GET /dbs/{db}/colls/{coll}/docs/{doc_id}`, scoped to partition
    /// `partition_key` — a point read. Returns `Ok(None)` on `404 Not
    /// Found` rather than erroring (mirrors [`Self::delete_document`]'s
    /// "not found is not a failure" treatment), `Ok(Some(document))` on
    /// success.
    pub(crate) async fn get_document(
        &self,
        database_id: &str,
        container_id: &str,
        partition_key: &str,
        doc_id: &str,
    ) -> Result<Option<Value>> {
        let resource_link = doc_link(database_id, container_id, doc_id);
        let pk_header = partition_key_header_value(partition_key)?;
        let resp = self
            .send(RequestSpec {
                method: reqwest::Method::GET,
                resource_type: "docs",
                resource_link: &resource_link,
                url_path: &resource_link,
                body: None,
                content_type: None,
                extra_headers: &[("x-ms-documentdb-partitionkey", pk_header)],
            })
            .await?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let text = resp
            .text()
            .await
            .map_err(|e| Error::service(format!("failed reading Cosmos DB response body: {e}")))?;
        if !status.is_success() {
            return Err(map_error_response(status, &text));
        }
        let value: Value = serde_json::from_str(&text).map_err(|e| {
            Error::service(format!(
                "invalid Cosmos DB response JSON: {e} (body: {text})"
            ))
        })?;
        Ok(Some(value))
    }

    /// `POST /dbs/{db}/colls/{coll}/docs` with `x-ms-documentdb-is-upsert:
    /// True`, scoped to partition `partition_key` — creates `document` if
    /// its `id` doesn't yet exist in this partition, else replaces it in
    /// place. Returns the resulting document as returned by Cosmos DB.
    pub(crate) async fn upsert_document(
        &self,
        database_id: &str,
        container_id: &str,
        partition_key: &str,
        document: &Value,
    ) -> Result<Value> {
        let resource_link = coll_link(database_id, container_id);
        let url_path = docs_link(database_id, container_id);
        let pk_header = partition_key_header_value(partition_key)?;
        let resp = self
            .send(RequestSpec {
                method: reqwest::Method::POST,
                resource_type: "docs",
                resource_link: &resource_link,
                url_path: &url_path,
                body: Some(document),
                content_type: None,
                extra_headers: &[
                    ("x-ms-documentdb-partitionkey", pk_header),
                    ("x-ms-documentdb-is-upsert", "True".to_string()),
                ],
            })
            .await?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::service(format!("failed reading Cosmos DB response body: {e}")))?;
        if !status.is_success() {
            return Err(map_error_response(status, &text));
        }
        serde_json::from_str(&text).map_err(|e| {
            Error::service(format!(
                "invalid Cosmos DB response JSON: {e} (body: {text})"
            ))
        })
    }

    /// `DELETE /dbs/{db}/colls/{coll}/docs/{doc_id}`, scoped to partition
    /// `partition_key`. `404 Not Found` (already gone) is treated as
    /// success.
    pub(crate) async fn delete_document(
        &self,
        database_id: &str,
        container_id: &str,
        partition_key: &str,
        doc_id: &str,
    ) -> Result<()> {
        let resource_link = doc_link(database_id, container_id, doc_id);
        let pk_header = partition_key_header_value(partition_key)?;
        let resp = self
            .send(RequestSpec {
                method: reqwest::Method::DELETE,
                resource_type: "docs",
                resource_link: &resource_link,
                url_path: &resource_link,
                body: None,
                content_type: None,
                extra_headers: &[("x-ms-documentdb-partitionkey", pk_header)],
            })
            .await?;
        let status = resp.status();
        if status.is_success() || status == reqwest::StatusCode::NOT_FOUND {
            Ok(())
        } else {
            let text = resp.text().await.unwrap_or_default();
            Err(map_error_response(status, &text))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // region: resource link / path helpers (pure)

    #[test]
    fn db_link_format() {
        assert_eq!(db_link("mydb"), "dbs/mydb");
    }

    #[test]
    fn coll_link_format() {
        assert_eq!(coll_link("mydb", "mycoll"), "dbs/mydb/colls/mycoll");
    }

    #[test]
    fn docs_link_format() {
        assert_eq!(docs_link("mydb", "mycoll"), "dbs/mydb/colls/mycoll/docs");
    }

    #[test]
    fn doc_link_format() {
        assert_eq!(
            doc_link("mydb", "mycoll", "msg-1"),
            "dbs/mydb/colls/mycoll/docs/msg-1"
        );
    }

    // endregion

    // region: partition_key_header_value (pure)

    #[test]
    fn partition_key_header_value_wraps_in_json_array() {
        assert_eq!(
            partition_key_header_value("thread-42").unwrap(),
            "[\"thread-42\"]"
        );
    }

    #[test]
    fn partition_key_header_value_escapes_embedded_quotes() {
        assert_eq!(partition_key_header_value("a\"b").unwrap(), "[\"a\\\"b\"]");
    }

    // endregion

    // region: build_query_body (pure)

    #[test]
    fn build_query_body_shape() {
        let body = build_query_body(
            "SELECT * FROM c WHERE c.threadId = @t",
            &[("@t", Value::String("thread-1".to_string()))],
        );
        assert_eq!(
            body,
            serde_json::json!({
                "query": "SELECT * FROM c WHERE c.threadId = @t",
                "parameters": [{"name": "@t", "value": "thread-1"}],
            })
        );
    }

    #[test]
    fn build_query_body_no_parameters() {
        let body = build_query_body("SELECT * FROM c", &[]);
        assert_eq!(body["parameters"], serde_json::json!([]));
    }

    #[test]
    fn build_query_body_multiple_parameters_preserve_order() {
        let body = build_query_body(
            "q",
            &[("@a", Value::String("1".into())), ("@b", Value::from(2))],
        );
        assert_eq!(
            body["parameters"],
            serde_json::json!([{"name": "@a", "value": "1"}, {"name": "@b", "value": 2}])
        );
    }

    // endregion

    // region: parse_query_response (pure)

    #[test]
    fn parse_query_response_extracts_documents_array() {
        let body = serde_json::json!({
            "_rid": "abc",
            "Documents": [{"id": "1"}, {"id": "2"}],
            "_count": 2,
        });
        let docs = parse_query_response(&body);
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0]["id"], "1");
    }

    #[test]
    fn parse_query_response_missing_documents_field_yields_empty() {
        let body = serde_json::json!({"_rid": "abc", "_count": 0});
        assert!(parse_query_response(&body).is_empty());
    }

    #[test]
    fn parse_query_response_non_array_documents_field_yields_empty() {
        let body = serde_json::json!({"Documents": "not-an-array"});
        assert!(parse_query_response(&body).is_empty());
    }

    // endregion

    // region: map_error_response (pure)

    #[test]
    fn map_error_response_includes_status_and_body() {
        let err = map_error_response(reqwest::StatusCode::UNAUTHORIZED, "bad key");
        let msg = err.to_string();
        assert!(msg.contains("401"));
        assert!(msg.contains("bad key"));
    }

    // endregion

    // region: CosmosRestClient::new (pure-ish: no network, just validation)

    #[test]
    fn new_trims_trailing_slash_from_endpoint() {
        let client = CosmosRestClient::new(
            "https://acct.documents.azure.com/",
            "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==",
        )
        .unwrap();
        assert_eq!(
            client.account_endpoint(),
            "https://acct.documents.azure.com"
        );
    }

    #[test]
    fn new_rejects_empty_endpoint() {
        assert!(CosmosRestClient::new("", "key").is_err());
    }

    #[test]
    fn new_rejects_invalid_base64_key() {
        assert!(
            CosmosRestClient::new("https://acct.documents.azure.com", "not-base64!!!").is_err()
        );
    }

    #[test]
    fn default_api_version_is_2018_12_31() {
        let client = CosmosRestClient::new(
            "https://acct.documents.azure.com",
            "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==",
        )
        .unwrap();
        assert_eq!(client.api_version, DEFAULT_API_VERSION);
    }

    // endregion
}
