//! The todo list: a [`ContextProvider`] that gives an agent tools to plan and
//! track work items for the current session.
//!
//! Port of upstream `_harness/_todo.py`. [`TodoProvider`] contributes five
//! tools (`todos_add`, `todos_complete`, `todos_remove`,
//! `todos_get_remaining`, `todos_get_all`), its usage instructions, and a
//! `user` message listing the current items. Items persist through a
//! [`TodoStore`]; the default [`TodoSessionStore`] keeps them in the
//! session's state bag as `{"items": [...], "next_id": n}` under the
//! provider's `source_id`.
//!
//! # Divergences
//!
//! - Read-modify-write tool calls are serialized by one lock per provider
//!   rather than upstream's per-session lock map. Tool calls of different
//!   sessions on the same provider therefore also take turns; the critical
//!   section is a state read and write, so the cost is negligible.
//! - `TodoFileStore` is not ported. Implement [`TodoStore`] for another
//!   backing store.
//! - The tools return their JSON result as a JSON value rather than a JSON
//!   string; the model sees the same text.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::memory::{ContextProvider, SessionContext};
use crate::session::AgentSession;
use crate::tools::{FunctionTool, ToolDefinition};
use crate::types::Message;

use super::{json_type_name, read_state_object};

/// The default `source_id` (state key) of a [`TodoProvider`].
pub const DEFAULT_TODO_SOURCE_ID: &str = "todo";

/// The default instructions a [`TodoProvider`] adds to every run.
pub const DEFAULT_TODO_INSTRUCTIONS: &str = "## Todo Items\n\n\
You have access to a todo list for tracking work items.\n\
When a user asks you to perform a task, follow these steps to manage your work:\n\
1. Determine whether the ask requires multiple steps to complete (complex) or can be completed \
using a single step (simple).\n\
2. If complex, turn the task into manageable todo items and add them to the list.\n\
3. If simple, don't add a todo item, but rather just complete the task directly.\n\n\
### General TODO Guidelines\n\
Ask questions from the user where clarification is needed to create effective todos.\n\
If the user provides feedback on your plan, adjust your todos accordingly by adding new items \
or removing irrelevant ones.\n\
During execution, use the todo list to keep track of what needs to be done, \
mark items as complete when finished, and remove any items that are no longer needed.\n\
When a user changes the topic, changes their mind or switches to a new request, ensure that you update \
the todo list accordingly by removing irrelevant/old items, clearing the list, or adding new ones as needed.\n\n\
Use these tools to manage your tasks:\n\
- Use todos_add to break down complex work into trackable items (supports adding one or many at once).\n\
- Use todos_complete to mark items as done when finished (supports one or many at once). \
Include a reason describing how the items were completed.\n\
- Use todos_get_remaining to check what work is still pending.\n\
- Use todos_get_all to review the full list including completed items.\n\
- Use todos_remove to remove items that are no longer needed (supports one or many at once).";

/// One todo item tracked for a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoItem {
    /// The item's id, unique within the session's list.
    pub id: u64,
    /// A short title.
    pub title: String,
    /// An optional longer description.
    pub description: Option<String>,
    /// Whether the item has been completed.
    #[serde(default)]
    pub is_complete: bool,
}

impl TodoItem {
    /// A new, open item.
    pub fn new(id: u64, title: impl Into<String>, description: Option<String>) -> Self {
        Self {
            id,
            title: title.into(),
            description,
            is_complete: false,
        }
    }

    /// Parse one persisted item, rejecting the shapes upstream rejects (a
    /// missing or non-integer id, an empty title, a non-string description,
    /// a non-boolean completion flag).
    pub fn from_value(value: &Value) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            Error::Serialization(format!(
                "todo item must be a JSON object; got {}",
                json_type_name(value)
            ))
        })?;
        let id = obj
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::Serialization("Todo item id must be an integer.".into()))?;
        let title = obj
            .get("title")
            .and_then(Value::as_str)
            .filter(|t| !t.trim().is_empty())
            .ok_or_else(|| {
                Error::Serialization("Todo item title must be a non-empty string.".into())
            })?;
        let description = match obj.get("description") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(_) => {
                return Err(Error::Serialization(
                    "Todo item description must be a string or null.".into(),
                ))
            }
        };
        let is_complete = match obj.get("is_complete") {
            None => false,
            Some(Value::Bool(b)) => *b,
            Some(_) => {
                return Err(Error::Serialization(
                    "Todo item is_complete must be a boolean.".into(),
                ))
            }
        };
        Ok(Self {
            id,
            title: title.to_string(),
            description,
            is_complete,
        })
    }

    /// The item as stored and as returned by the tools (`description` is
    /// written as `null` when absent).
    pub fn to_value(&self) -> Value {
        json!({
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "is_complete": self.is_complete,
        })
    }
}

/// One item to create via `todos_add`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TodoInput {
    /// The item's title.
    pub title: String,
    /// An optional longer description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl TodoInput {
    /// A validated input: the title is trimmed and must not be empty.
    pub fn new(title: impl Into<String>, description: Option<String>) -> Result<Self> {
        let title = title.into().trim().to_string();
        if title.is_empty() {
            return Err(Error::tool("Todo input title must be a non-empty string."));
        }
        Ok(Self { title, description })
    }
}

/// One item to mark complete via `todos_complete`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TodoCompleteInput {
    /// The id of the item to complete.
    pub id: u64,
    /// How or why the item was completed.
    pub reason: String,
}

impl TodoCompleteInput {
    /// A validated input: the reason is trimmed and must not be empty.
    pub fn new(id: u64, reason: impl Into<String>) -> Result<Self> {
        let reason = reason.into().trim().to_string();
        if reason.is_empty() {
            return Err(Error::tool(
                "Todo complete input reason must be a non-empty string.",
            ));
        }
        Ok(Self { id, reason })
    }
}

/// Clamp `next_id` so it can never collide with a persisted item id.
fn safe_next_id(items: &[TodoItem], next_id: u64) -> u64 {
    next_id.max(items.iter().map(|i| i.id).max().unwrap_or(0) + 1)
}

/// The backing store for a session's todo items.
///
/// Mirrors upstream's `TodoStore` ABC. Implement it to keep todos somewhere
/// other than the session's state bag.
#[async_trait]
pub trait TodoStore: Send + Sync {
    /// Load the persisted items and the next id to assign.
    async fn load_state(
        &self,
        session: &AgentSession,
        source_id: &str,
    ) -> Result<(Vec<TodoItem>, u64)>;

    /// Persist the items and the next id to assign.
    async fn save_state(
        &self,
        session: &AgentSession,
        items: &[TodoItem],
        next_id: u64,
        source_id: &str,
    ) -> Result<()>;

    /// Load only the items.
    async fn load_items(&self, session: &AgentSession, source_id: &str) -> Result<Vec<TodoItem>> {
        Ok(self.load_state(session, source_id).await?.0)
    }
}

/// The default [`TodoStore`]: keeps todos in the session's state bag.
#[derive(Debug, Clone, Copy, Default)]
pub struct TodoSessionStore;

#[async_trait]
impl TodoStore for TodoSessionStore {
    async fn load_state(
        &self,
        session: &AgentSession,
        source_id: &str,
    ) -> Result<(Vec<TodoItem>, u64)> {
        if !session.state.contains_key(source_id) {
            session.state.insert(source_id, json!({}));
        }
        let state = read_state_object(session, source_id)?;
        let items = match state.get("items") {
            None => Vec::new(),
            Some(Value::Array(raw)) => raw
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    if !item.is_object() {
                        return Err(Error::Serialization(format!(
                            "Todo item at index {index} in session todo state must be a \
                             mapping; got {}.",
                            json_type_name(item)
                        )));
                    }
                    TodoItem::from_value(item)
                })
                .collect::<Result<Vec<_>>>()?,
            Some(other) => {
                return Err(Error::Serialization(format!(
                    "Session state for source_id {source_id:?} has a non-list 'items' field; \
                     got {}.",
                    json_type_name(other)
                )))
            }
        };
        let next_id = match state.get("next_id") {
            None => 1,
            Some(v) => v.as_u64().ok_or_else(|| {
                Error::Serialization(format!(
                    "Session state for source_id {source_id:?} has a non-integer 'next_id' \
                     field; got {}.",
                    json_type_name(v)
                ))
            })?,
        };
        let next_id = safe_next_id(&items, next_id);
        Ok((items, next_id))
    }

    async fn save_state(
        &self,
        session: &AgentSession,
        items: &[TodoItem],
        next_id: u64,
        source_id: &str,
    ) -> Result<()> {
        // Keep any other keys a caller stored alongside ours.
        let mut state = match session.state.get(source_id) {
            Some(Value::Object(map)) => map,
            _ => serde_json::Map::new(),
        };
        state.insert(
            "items".into(),
            Value::Array(items.iter().map(TodoItem::to_value).collect()),
        );
        state.insert("next_id".into(), json!(safe_next_id(items, next_id)));
        session.state.insert(source_id, Value::Object(state));
        Ok(())
    }
}

struct TodoInner {
    source_id: String,
    instructions: String,
    store: Arc<dyn TodoStore>,
    mutation_lock: tokio::sync::Mutex<()>,
}

/// Gives an agent a session-scoped todo list.
///
/// Register it as a context provider on the agent
/// ([`AgentBuilder::context_provider`](crate::agent::AgentBuilder::context_provider)).
/// On each run it adds its instructions, five todo tools, and a `user`
/// message listing the current items. Keep the `Arc` you register to read the
/// list from application code ([`TodoProvider::items`],
/// [`TodoProvider::remaining`]) or to drive a [`LoopAgent`](super::LoopAgent)
/// with [`todos_remaining`](super::todos_remaining).
///
/// Cloning yields a handle onto the same provider.
#[derive(Clone)]
pub struct TodoProvider {
    inner: Arc<TodoInner>,
}

impl std::fmt::Debug for TodoProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TodoProvider")
            .field("source_id", &self.inner.source_id)
            .finish_non_exhaustive()
    }
}

impl Default for TodoProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl TodoProvider {
    /// A provider with the default source id, instructions and
    /// [`TodoSessionStore`].
    pub fn new() -> Self {
        Self::build(
            DEFAULT_TODO_SOURCE_ID.to_string(),
            DEFAULT_TODO_INSTRUCTIONS.to_string(),
            Arc::new(TodoSessionStore),
        )
    }

    fn build(source_id: String, instructions: String, store: Arc<dyn TodoStore>) -> Self {
        Self {
            inner: Arc::new(TodoInner {
                source_id,
                instructions,
                store,
                mutation_lock: tokio::sync::Mutex::new(()),
            }),
        }
    }

    /// Builder: use `source_id` as the state key (default `"todo"`).
    pub fn with_source_id(self, source_id: impl Into<String>) -> Self {
        Self::build(
            source_id.into(),
            self.inner.instructions.clone(),
            self.inner.store.clone(),
        )
    }

    /// Builder: replace the default instructions.
    pub fn with_instructions(self, instructions: impl Into<String>) -> Self {
        Self::build(
            self.inner.source_id.clone(),
            instructions.into(),
            self.inner.store.clone(),
        )
    }

    /// Builder: persist todos through `store` instead of the session state.
    pub fn with_store(self, store: Arc<dyn TodoStore>) -> Self {
        Self::build(
            self.inner.source_id.clone(),
            self.inner.instructions.clone(),
            store,
        )
    }

    /// The state key this provider uses.
    pub fn source_id(&self) -> &str {
        &self.inner.source_id
    }

    /// The instructions this provider adds to every run.
    pub fn instructions(&self) -> &str {
        &self.inner.instructions
    }

    /// The backing store.
    pub fn store(&self) -> &Arc<dyn TodoStore> {
        &self.inner.store
    }

    /// Every todo item of `session`, complete and open.
    pub async fn items(&self, session: &AgentSession) -> Result<Vec<TodoItem>> {
        self.inner
            .store
            .load_items(session, &self.inner.source_id)
            .await
    }

    /// The open todo items of `session`.
    pub async fn remaining(&self, session: &AgentSession) -> Result<Vec<TodoItem>> {
        Ok(self
            .items(session)
            .await?
            .into_iter()
            .filter(|i| !i.is_complete)
            .collect())
    }

    /// The five todo tools, bound to `session`.
    pub fn tools(&self, session: &AgentSession) -> Vec<ToolDefinition> {
        vec![
            self.add_tool(session),
            self.complete_tool(session),
            self.remove_tool(session),
            self.get_remaining_tool(session),
            self.get_all_tool(session),
        ]
    }

    fn add_tool(&self, session: &AgentSession) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// The todo items to add.
            todos: Vec<TodoInput>,
        }
        let inner = self.inner.clone();
        let session = session.clone();
        FunctionTool::typed(
            "todos_add",
            "Add one or more todo items for the current session.",
            move |args: Args| {
                let inner = inner.clone();
                let session = session.clone();
                async move {
                    if args.todos.is_empty() {
                        return Err(Error::tool("todos must contain at least one item."));
                    }
                    let todos = args
                        .todos
                        .into_iter()
                        .map(|t| TodoInput::new(t.title, t.description))
                        .collect::<Result<Vec<_>>>()?;
                    let _guard = inner.mutation_lock.lock().await;
                    let (mut items, mut next_id) =
                        inner.store.load_state(&session, &inner.source_id).await?;
                    let mut created = Vec::with_capacity(todos.len());
                    for todo in todos {
                        let item = TodoItem::new(
                            next_id,
                            todo.title,
                            todo.description.map(|d| d.trim().to_string()),
                        );
                        created.push(item.to_value());
                        items.push(item);
                        next_id += 1;
                    }
                    inner
                        .store
                        .save_state(&session, &items, next_id, &inner.source_id)
                        .await?;
                    Ok(Value::Array(created))
                }
            },
        )
        .into_definition()
    }

    fn complete_tool(&self, session: &AgentSession) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// The items to mark complete, each with the id and a reason
            /// describing how or why it was completed.
            items: Vec<TodoCompleteInput>,
        }
        let inner = self.inner.clone();
        let session = session.clone();
        FunctionTool::typed(
            "todos_complete",
            "Mark one or more todo items as complete. Each entry has an id (int) and a reason \
             (string) describing how/why the item was completed.",
            move |args: Args| {
                let inner = inner.clone();
                let session = session.clone();
                async move {
                    if args.items.is_empty() {
                        return Err(Error::tool("items must contain at least one entry."));
                    }
                    let ids = args
                        .items
                        .into_iter()
                        .map(|i| TodoCompleteInput::new(i.id, i.reason).map(|i| i.id))
                        .collect::<Result<std::collections::HashSet<_>>>()?;
                    let _guard = inner.mutation_lock.lock().await;
                    let (mut items, next_id) =
                        inner.store.load_state(&session, &inner.source_id).await?;
                    let mut completed = 0;
                    for item in &mut items {
                        if !item.is_complete && ids.contains(&item.id) {
                            item.is_complete = true;
                            completed += 1;
                        }
                    }
                    if completed > 0 {
                        inner
                            .store
                            .save_state(&session, &items, next_id, &inner.source_id)
                            .await?;
                    }
                    Ok(json!({ "completed": completed }))
                }
            },
        )
        .into_definition()
    }

    fn remove_tool(&self, session: &AgentSession) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// The ids of the items to remove.
            ids: Vec<u64>,
        }
        let inner = self.inner.clone();
        let session = session.clone();
        FunctionTool::typed(
            "todos_remove",
            "Remove one or more todo items by ID.",
            move |args: Args| {
                let inner = inner.clone();
                let session = session.clone();
                async move {
                    if args.ids.is_empty() {
                        return Err(Error::tool("ids must contain at least one todo ID."));
                    }
                    let _guard = inner.mutation_lock.lock().await;
                    let (items, next_id) =
                        inner.store.load_state(&session, &inner.source_id).await?;
                    let before = items.len();
                    let remaining: Vec<TodoItem> = items
                        .into_iter()
                        .filter(|i| !args.ids.contains(&i.id))
                        .collect();
                    let removed = before - remaining.len();
                    if removed > 0 {
                        inner
                            .store
                            .save_state(&session, &remaining, next_id, &inner.source_id)
                            .await?;
                    }
                    Ok(json!({ "removed": removed }))
                }
            },
        )
        .into_definition()
    }

    fn get_remaining_tool(&self, session: &AgentSession) -> ToolDefinition {
        let provider = self.clone();
        let session = session.clone();
        FunctionTool::new(
            "todos_get_remaining",
            "Retrieve only incomplete todo items for the current session.",
            crate::tools::empty_schema(),
            move |_args| {
                let provider = provider.clone();
                let session = session.clone();
                async move {
                    let items = provider.remaining(&session).await?;
                    Ok(Value::Array(items.iter().map(TodoItem::to_value).collect()))
                }
            },
        )
        .into_definition()
    }

    fn get_all_tool(&self, session: &AgentSession) -> ToolDefinition {
        let provider = self.clone();
        let session = session.clone();
        FunctionTool::new(
            "todos_get_all",
            "Retrieve all todo items for the current session.",
            crate::tools::empty_schema(),
            move |_args| {
                let provider = provider.clone();
                let session = session.clone();
                async move {
                    let items = provider.items(&session).await?;
                    Ok(Value::Array(items.iter().map(TodoItem::to_value).collect()))
                }
            },
        )
        .into_definition()
    }
}

/// Render the "Current todo list" message body.
fn render_todo_list(items: &[TodoItem]) -> String {
    let lines: Vec<String> = items
        .iter()
        .map(|item| {
            let status = if item.is_complete { "done" } else { "open" };
            let mut line = format!("- {} [{status}] {}", item.id, item.title);
            if let Some(d) = item.description.as_deref().filter(|d| !d.is_empty()) {
                line.push_str(": ");
                line.push_str(d);
            }
            line
        })
        .collect();
    let body = if lines.is_empty() {
        "- none yet".to_string()
    } else {
        lines.join("\n")
    };
    format!("### Current todo list\n{body}")
}

#[async_trait]
impl ContextProvider for TodoProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let session = ctx.session.clone().ok_or_else(|| {
            Error::Configuration("TodoProvider requires an agent session.".into())
        })?;
        ctx.add_instructions(self.inner.instructions.clone());
        ctx.tools.extend(self.tools(&session));
        let items = self.items(&session).await?;
        ctx.messages.push(Message::user(render_todo_list(&items)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool<'a>(tools: &'a [ToolDefinition], name: &str) -> &'a ToolDefinition {
        tools.iter().find(|t| t.name == name).unwrap()
    }

    async fn call(tools: &[ToolDefinition], name: &str, args: Value) -> Result<Value> {
        tool(tools, name)
            .executor
            .as_ref()
            .unwrap()
            .invoke(args)
            .await
    }

    #[test]
    fn todo_item_round_trips_with_value_equality() {
        let item = TodoItem::new(3, "Write tests", Some("cover the store".into()));
        let restored = TodoItem::from_value(&item.to_value()).unwrap();
        assert_eq!(restored, item);
        assert_eq!(item.to_value()["description"], "cover the store");
        let bare = TodoItem::new(4, "No description", None);
        assert_eq!(bare.to_value()["description"], Value::Null);
    }

    #[test]
    fn todo_inputs_validate() {
        assert_eq!(TodoInput::new("  title  ", None).unwrap().title, "title");
        assert!(TodoInput::new("   ", None).is_err());
        assert_eq!(TodoCompleteInput::new(1, " done ").unwrap().reason, "done");
        assert!(TodoCompleteInput::new(1, "  ").is_err());
    }

    #[test]
    fn todo_item_rejects_malformed_values() {
        assert!(TodoItem::from_value(&json!({"id": "1", "title": "x"})).is_err());
        assert!(TodoItem::from_value(&json!({"id": 1, "title": " "})).is_err());
        assert!(TodoItem::from_value(&json!({"id": 1, "title": "x", "description": 3})).is_err());
        assert!(
            TodoItem::from_value(&json!({"id": 1, "title": "x", "is_complete": "no"})).is_err()
        );
        assert!(TodoItem::from_value(&json!(["not", "an", "object"])).is_err());
    }

    #[tokio::test]
    async fn session_store_initializes_and_round_trips_state() {
        let session = AgentSession::new();
        let store = TodoSessionStore;
        let (items, next_id) = store.load_state(&session, "todo").await.unwrap();
        assert!(items.is_empty());
        assert_eq!(next_id, 1);
        assert_eq!(session.state.get("todo"), Some(json!({})));

        let items = vec![TodoItem::new(1, "a", None), TodoItem::new(2, "b", None)];
        store.save_state(&session, &items, 3, "todo").await.unwrap();
        let (loaded, next_id) = store.load_state(&session, "todo").await.unwrap();
        assert_eq!(loaded, items);
        assert_eq!(next_id, 3);
    }

    #[tokio::test]
    async fn session_store_rejects_malformed_state() {
        let session = AgentSession::new();
        let store = TodoSessionStore;
        session.state.insert("todo", json!({"items": [1]}));
        assert!(store.load_state(&session, "todo").await.is_err());
        session.state.insert("todo", json!({"items": {}}));
        assert!(store.load_state(&session, "todo").await.is_err());
        session
            .state
            .insert("todo", json!({"items": [], "next_id": "2"}));
        assert!(store.load_state(&session, "todo").await.is_err());
        session.state.insert("todo", json!("not an object"));
        assert!(store.load_state(&session, "todo").await.is_err());
    }

    #[tokio::test]
    async fn session_store_clamps_next_id_to_avoid_collisions() {
        let session = AgentSession::new();
        session.state.insert(
            "todo",
            json!({"items": [{"id": 7, "title": "x", "description": null, "is_complete": false}],
                   "next_id": 2}),
        );
        let (_, next_id) = TodoSessionStore.load_state(&session, "todo").await.unwrap();
        assert_eq!(next_id, 8);
    }

    #[tokio::test]
    async fn tools_manage_session_state() {
        let provider = TodoProvider::new();
        let session = AgentSession::new();
        let tools = provider.tools(&session);
        assert_eq!(
            tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            [
                "todos_add",
                "todos_complete",
                "todos_remove",
                "todos_get_remaining",
                "todos_get_all"
            ]
        );

        let created = call(
            &tools,
            "todos_add",
            json!({"todos": [{"title": "First", "description": " details "}, {"title": "Second"}]}),
        )
        .await
        .unwrap();
        assert_eq!(
            created,
            json!([
                {"id": 1, "title": "First", "description": "details", "is_complete": false},
                {"id": 2, "title": "Second", "description": null, "is_complete": false}
            ])
        );

        let completed = call(
            &tools,
            "todos_complete",
            json!({"items": [{"id": 1, "reason": "done"}, {"id": 99, "reason": "missing"}]}),
        )
        .await
        .unwrap();
        assert_eq!(completed, json!({"completed": 1}));

        let remaining = call(&tools, "todos_get_remaining", json!({}))
            .await
            .unwrap();
        assert_eq!(remaining.as_array().unwrap().len(), 1);
        assert_eq!(remaining[0]["title"], "Second");

        let removed = call(&tools, "todos_remove", json!({"ids": [2, 5]}))
            .await
            .unwrap();
        assert_eq!(removed, json!({"removed": 1}));

        let all = call(&tools, "todos_get_all", json!({})).await.unwrap();
        assert_eq!(
            all,
            json!([{"id": 1, "title": "First", "description": "details", "is_complete": true}])
        );

        // Ids keep increasing after removals.
        let created = call(&tools, "todos_add", json!({"todos": [{"title": "Third"}]}))
            .await
            .unwrap();
        assert_eq!(created[0]["id"], 3);

        // Empty batches and blank titles are tool errors.
        assert!(call(&tools, "todos_add", json!({"todos": []}))
            .await
            .is_err());
        assert!(
            call(&tools, "todos_add", json!({"todos": [{"title": " "}]}))
                .await
                .is_err()
        );
        assert!(call(&tools, "todos_complete", json!({"items": []}))
            .await
            .is_err());
        assert!(call(&tools, "todos_remove", json!({"ids": []}))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn concurrent_adds_are_serialized() {
        let provider = TodoProvider::new();
        let session = AgentSession::new();
        let tools = Arc::new(provider.tools(&session));
        let mut handles = Vec::new();
        for i in 0..10 {
            let tools = tools.clone();
            handles.push(tokio::spawn(async move {
                call(
                    &tools,
                    "todos_add",
                    json!({"todos": [{"title": format!("item {i}")}]}),
                )
                .await
                .unwrap()
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        let items = provider.items(&session).await.unwrap();
        let mut ids: Vec<u64> = items.iter().map(|i| i.id).collect();
        ids.sort_unstable();
        assert_eq!(ids, (1..=10).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn before_run_injects_instructions_tools_and_current_list() {
        let provider = TodoProvider::new().with_source_id("plan");
        let session = AgentSession::new();
        session.state.insert(
            "plan",
            json!({"items": [
                {"id": 1, "title": "Draft", "description": "outline", "is_complete": true},
                {"id": 2, "title": "Review", "description": null, "is_complete": false}
            ], "next_id": 3}),
        );
        let mut ctx = SessionContext::new(vec![]);
        ctx.session = Some(session.clone());
        provider.before_run(&mut ctx).await.unwrap();
        assert_eq!(ctx.instructions.as_deref(), Some(DEFAULT_TODO_INSTRUCTIONS));
        assert_eq!(ctx.tools.len(), 5);
        assert_eq!(
            ctx.messages[0].text(),
            "### Current todo list\n- 1 [done] Draft: outline\n- 2 [open] Review"
        );

        let empty = AgentSession::new();
        let mut ctx = SessionContext::new(vec![]);
        ctx.session = Some(empty);
        provider.before_run(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages[0].text(), "### Current todo list\n- none yet");

        // A session is required.
        let mut ctx = SessionContext::new(vec![]);
        assert!(provider.before_run(&mut ctx).await.is_err());
    }
}
