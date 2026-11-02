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

    async fn delete(&self, id: &str) -> Result<bool> {
        let _guard = self.lock.lock().await;
        match tokio::fs::remove_file(self.path("responses", id)).await {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(io(e)),
        }
    }

    async fn conversation(&self, id: &str) -> Result<Option<Vec<Message>>> {
        match self.read(&self.path("conversations", id)).await? {
            Some(value) => Ok(Some(serde_json::from_value(value)?)),
            None => Ok(None),
        }
    }

    async fn set_conversation(&self, id: &str, transcript: Vec<Message>) -> Result<()> {
        let _guard = self.lock.lock().await;
        self.write(
            &self.path("conversations", id),
            &serde_json::to_value(transcript)?,
        )
        .await
    }

    async fn conversation_state(&self, id: &str) -> Result<Option<Value>> {
        self.read(&self.path("conversation_states", id)).await
    }

    async fn set_conversation_state(&self, id: &str, state: Value) -> Result<()> {
        let _guard = self.lock.lock().await;
        self.write(&self.path("conversation_states", id), &state)
            .await
    }
}
