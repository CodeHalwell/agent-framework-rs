//! A file-backed [`ResponseStore`] for local development (the agentserver's
//! `FileResponseStore`): responses and conversation transcripts survive a
//! restart.
//!
//! Layout under the storage directory:
//!
//! ```text
//! responses/{sha256(response id)}.json
//! conversations/{sha256(conversation id)}.json
//! ```
//!
//! Single-process only: writes are serialized in process and replace files
//! atomically (write, then rename); concurrent writers in several
//! processes race.

use std::path::{Path, PathBuf};

use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::Message;
use async_trait::async_trait;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::responses::{ResponseStore, StoredResponse, WorkflowPause};

/// The state root: `AGENTSERVER_STATE_ROOT`, else `~/.agentserver`.
pub fn state_root() -> PathBuf {
    match std::env::var("AGENTSERVER_STATE_ROOT") {
        Ok(root) if !root.is_empty() => PathBuf::from(root),
        _ => std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".agentserver"),
    }
}

/// Responses and conversations as JSON files under one directory.
#[derive(Debug)]
pub struct FileResponseStore {
    dir: PathBuf,
    lock: tokio::sync::Mutex<()>,
}

impl FileResponseStore {
    /// A store under `dir` (created on first write).
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            lock: tokio::sync::Mutex::new(()),
        }
    }

    /// The agentserver default: `{state root}/responses`.
    pub fn from_env() -> Self {
        Self::new(state_root().join("responses"))
    }

    /// The storage directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, kind: &str, id: &str) -> PathBuf {
        let digest = Sha256::digest(id.as_bytes());
        let name: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        self.dir.join(kind).join(format!("{name}.json"))
    }

    async fn read(&self, path: &Path) -> Result<Option<Value>> {
        match tokio::fs::read(path).await {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io(e)),
        }
    }

    /// A conversation's transcript and state. Reads the combined record,
    /// and the earlier layout (a bare transcript array, with the state in a
    /// separate `conversation_states` file).
    async fn read_conversation(&self, id: &str) -> Result<Option<(Vec<Message>, Option<Value>)>> {
        match self.read(&self.path("conversations", id)).await? {
            None => Ok(None),
            Some(Value::Array(items)) => {
                let transcript = serde_json::from_value(Value::Array(items))?;
                let state = self.read(&self.path("conversation_states", id)).await?;
                Ok(Some((transcript, state)))
            }
            Some(record) => {
                let transcript = serde_json::from_value(record["transcript"].clone())?;
                let state = match &record["session_state"] {
                    Value::Null => None,
                    state => Some(state.clone()),
                };
                Ok(Some((transcript, state)))
            }
        }
    }

    async fn write_conversation(
        &self,
        id: &str,
        transcript: Vec<Message>,
        state: Option<Value>,
    ) -> Result<()> {
        let record = serde_json::json!({"transcript": transcript, "session_state": state});
        self.write(&self.path("conversations", id), &record).await
    }

    async fn write(&self, path: &Path, value: &Value) -> Result<()> {
        let parent = path.parent().unwrap_or(&self.dir);
        tokio::fs::create_dir_all(parent).await.map_err(io)?;
        let tmp = path.with_extension(format!("tmp-{}", uuid::Uuid::new_v4().simple()));
        tokio::fs::write(&tmp, serde_json::to_vec(value)?)
            .await
            .map_err(io)?;
        tokio::fs::rename(&tmp, path).await.map_err(io)
    }
}

fn io(e: std::io::Error) -> Error {
    Error::Other(format!("file response store: {e}"))
}

fn to_json(stored: &StoredResponse) -> Value {
    json!({
        "response": stored.response,
        "input_items": stored.input_items,
        "transcript": stored.transcript,
        "pause": stored.pause.as_ref().map(|p| json!({
            "checkpoint_id": p.checkpoint_id,
            "requests": p.requests,
            "request_data": p.request_data,
        })),
        "session_state": stored.session_state,
    })
}

fn from_json(value: Value) -> Result<StoredResponse> {
    let transcript: Vec<Message> = serde_json::from_value(value["transcript"].clone())?;
    let input_items = value["input_items"].as_array().cloned().unwrap_or_default();
    let pause = match &value["pause"] {
        Value::Null => None,
        p => Some(WorkflowPause {
            checkpoint_id: p["checkpoint_id"].as_str().unwrap_or_default().to_string(),
            requests: serde_json::from_value(p["requests"].clone())?,
            request_data: p["request_data"].as_array().cloned().unwrap_or_default(),
        }),
    };
    Ok(StoredResponse {
        response: value["response"].clone(),
        input_items,
        transcript,
        pause,
        session_state: match &value["session_state"] {
            Value::Null => None,
            state => Some(state.clone()),
        },
    })
}

#[async_trait]
impl ResponseStore for FileResponseStore {
    async fn get(&self, id: &str) -> Result<Option<StoredResponse>> {
        match self.read(&self.path("responses", id)).await? {
            Some(value) => Ok(Some(from_json(value)?)),
            None => Ok(None),
        }
    }

    async fn put(&self, id: &str, response: StoredResponse) -> Result<()> {
        let _guard = self.lock.lock().await;
        self.write(&self.path("responses", id), &to_json(&response))
            .await
    }

    async fn response_ids(&self) -> Result<Vec<String>> {
        let mut entries = match tokio::fs::read_dir(self.dir.join("responses")).await {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io(e)),
        };
        // Files are named by digest: the id is read from each record.
        let mut ids = Vec::new();
        while let Some(entry) = entries.next_entry().await.map_err(io)? {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match self.read(&path).await {
                Ok(Some(value)) => {
                    if let Some(id) = value["response"]["id"].as_str() {
                        ids.push(id.to_string());
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "skipping an unreadable response record")
                }
            }
        }
        Ok(ids)
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let _guard = self.lock.lock().await;
        match tokio::fs::remove_file(self.path("responses", id)).await {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(io(e)),
        }
    }

    async fn conversation(&self, id: &str) -> Result<Option<Vec<Message>>> {
        Ok(self
            .read_conversation(id)
            .await?
            .map(|(transcript, _)| transcript))
    }

    async fn set_conversation(&self, id: &str, transcript: Vec<Message>) -> Result<()> {
        let _guard = self.lock.lock().await;
        let state = self.read_conversation(id).await?.and_then(|(_, s)| s);
        self.write_conversation(id, transcript, state).await
    }

    async fn conversation_state(&self, id: &str) -> Result<Option<Value>> {
        Ok(self
            .read_conversation(id)
            .await?
            .and_then(|(_, state)| state))
    }

    async fn set_conversation_state(&self, id: &str, state: Value) -> Result<()> {
        let _guard = self.lock.lock().await;
        let transcript = self
            .read_conversation(id)
            .await?
            .map(|(t, _)| t)
            .unwrap_or_default();
        self.write_conversation(id, transcript, Some(state)).await
    }

    /// One file, written by atomic rename: transcript and state together.
    async fn update_conversation(
        &self,
        id: &str,
        transcript: Vec<Message>,
        state: Option<Value>,
    ) -> Result<()> {
        let _guard = self.lock.lock().await;
        self.write_conversation(id, transcript, state).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn conversations_keep_transcript_and_state_in_one_record() {
        let dir = std::env::temp_dir().join(format!("af-fs-{}", uuid::Uuid::new_v4().simple()));
        let store = FileResponseStore::new(&dir);
        store
            .update_conversation("c", vec![Message::user("a")], Some(json!({"turns": 1})))
            .await
            .unwrap();
        assert_eq!(store.conversation("c").await.unwrap().unwrap().len(), 1);
        assert_eq!(
            store.conversation_state("c").await.unwrap(),
            Some(json!({"turns": 1}))
        );
        // The separate setters keep the other half.
        store
            .set_conversation("c", vec![Message::user("a"), Message::user("b")])
            .await
            .unwrap();
        assert_eq!(
            store.conversation_state("c").await.unwrap(),
            Some(json!({"turns": 1}))
        );
        // The earlier layout (a bare transcript array and a separate state
        // file) still reads.
        store
            .write(
                &store.path("conversations", "old"),
                &json!([{"role": "user", "contents": [{"type": "text", "text": "x"}]}]),
            )
            .await
            .unwrap();
        store
            .write(
                &store.path("conversation_states", "old"),
                &json!({"turns": 7}),
            )
            .await
            .unwrap();
        assert_eq!(store.conversation("old").await.unwrap().unwrap().len(), 1);
        assert_eq!(
            store.conversation_state("old").await.unwrap(),
            Some(json!({"turns": 7}))
        );
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }
}
