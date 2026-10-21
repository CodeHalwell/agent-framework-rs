//! The harness file store and file memory (`experimental-harness`).
//!
//! Ported from upstream `test_harness_file_memory.py` and the store tests in
//! `test_harness_file_access.py`.
#![cfg(feature = "experimental-harness")]

use std::sync::{Arc, Mutex};

use agent_framework_core::harness::*;
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};

/// Runs `before_run` for `session` and returns the context it filled.
async fn prepare(provider: &FileMemoryProvider, session: &AgentSession) -> SessionContext {
    let mut ctx = SessionContext::new(vec![Message::user("hi")]);
    ctx.session = Some(session.clone());
    provider.before_run(&mut ctx).await.unwrap();
    ctx
}

async fn call(ctx: &SessionContext, name: &str, args: Value) -> Value {
    ctx.tools
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("no tool {name}"))
        .executor
        .as_ref()
        .unwrap()
        .invoke(args)
        .await
        .unwrap()
}

fn memory() -> (Arc<InMemoryAgentFileStore>, FileMemoryProvider) {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let provider = FileMemoryProvider::new(store.clone());
    (store, provider)
}

#[tokio::test]
async fn before_run_adds_instructions_and_the_seven_tools() {
    let (_, provider) = memory();
    let ctx = prepare(&provider, &AgentSession::new()).await;
    assert_eq!(
        ctx.instructions.as_deref(),
        Some(DEFAULT_FILE_MEMORY_INSTRUCTIONS)
    );
    let names: Vec<_> = ctx.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "file_memory_write",
            "file_memory_read",
            "file_memory_delete",
            "file_memory_ls",
            "file_memory_grep",
            "file_memory_replace",
            "file_memory_replace_lines"
        ]
    );
    // No index yet, so no index message.
    assert!(ctx.messages.is_empty());
}

#[tokio::test]
async fn write_read_delete_round_trip() {
    let (_, provider) = memory();
    let ctx = prepare(&provider, &AgentSession::new()).await;

    let written = call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "notes.md", "content": "hello"}),
    )
    .await;
    assert_eq!(written, "File 'notes.md' written.");
    assert_eq!(
        call(&ctx, "file_memory_read", json!({"file_name": "notes.md"})).await,
        "hello"
    );
    assert_eq!(
        call(&ctx, "file_memory_delete", json!({"file_name": "notes.md"})).await,
        "File 'notes.md' deleted."
    );
    assert_eq!(
        call(&ctx, "file_memory_read", json!({"file_name": "notes.md"})).await,
        "File 'notes.md' not found."
    );
    assert_eq!(
        call(&ctx, "file_memory_delete", json!({"file_name": "notes.md"})).await,
        "File 'notes.md' not found."
    );
}

#[tokio::test]
async fn descriptions_are_listed_and_removed_with_their_file() {
    let (store, provider) = memory();
    let session = AgentSession::new();
    let ctx = prepare(&provider, &session).await;
    let folder = provider.working_folder(Some(&session)).unwrap();

    let written = call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "plan.md", "content": "steps", "description": "The plan"}),
    )
    .await;
    assert_eq!(written, "File 'plan.md' written with description.");
    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "raw.txt", "content": "data"}),
    )
    .await;
    assert_eq!(
        call(&ctx, "file_memory_ls", json!({})).await,
        json!([
            {"name": "plan.md", "type": "file", "description": "The plan"},
            {"name": "raw.txt", "type": "file", "description": null}
        ])
    );
    assert_eq!(
        call(&ctx, "file_memory_ls", json!({"glob_pattern": "*.MD"})).await,
        json!([{"name": "plan.md", "type": "file", "description": "The plan"}])
    );

    // Rewriting without a description drops the stale one.
    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "plan.md", "content": "v2"}),
    )
    .await;
    assert!(!store
        .file_exists(&format!("{folder}/plan.md_description.md"))
        .await
        .unwrap());

    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "plan.md", "content": "v3", "description": "Again"}),
    )
    .await;
    call(&ctx, "file_memory_delete", json!({"file_name": "plan.md"})).await;
    assert!(!store
        .file_exists(&format!("{folder}/plan.md_description.md"))
        .await
        .unwrap());
}

#[tokio::test]
async fn the_index_is_rebuilt_and_given_to_the_next_run() {
    let (store, provider) = memory();
    let session = AgentSession::new();
    let ctx = prepare(&provider, &session).await;
    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "b.md", "content": "x", "description": "  Second  "}),
    )
    .await;
    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "A.md", "content": "y"}),
    )
    .await;

    let folder = provider.working_folder(Some(&session)).unwrap();
    let index = store
        .read(&format!("{folder}/memories.md"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(index, "# Memory Index\n\n- **A.md**\n- **b.md**: Second\n");

    let next = prepare(&provider, &session).await;
    assert_eq!(next.messages.len(), 1);
    let text = next.messages[0].text();
    assert!(text.starts_with("The following is your memory index \u{2014} a list of files"));
    assert!(text.ends_with(&index));
}

#[tokio::test]
async fn the_index_holds_at_most_fifty_files() {
    let (store, provider) = memory();
    let session = AgentSession::new();
    let ctx = prepare(&provider, &session).await;
    for i in 0..55 {
        call(
            &ctx,
            "file_memory_write",
            json!({"file_name": format!("f{i:02}.md"), "content": "x"}),
        )
        .await;
    }
    let folder = provider.working_folder(Some(&session)).unwrap();
    let index = store
        .read(&format!("{folder}/memories.md"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(index.matches("- **").count(), 50);
    assert!(index.contains("f49.md") && !index.contains("f50.md"));
}

#[tokio::test]
async fn listings_and_searches_hide_internal_files() {
    let (_, provider) = memory();
    let ctx = prepare(&provider, &AgentSession::new()).await;
    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "notes.md", "content": "secret plan", "description": "secret summary"}),
    )
    .await;

    let listed = call(&ctx, "file_memory_ls", json!({})).await;
    assert_eq!(listed.as_array().unwrap().len(), 1);

    // "secret" is in the file, its description and the index; only the file
    // is reported.
    let found = call(&ctx, "file_memory_grep", json!({"regex_pattern": "SECRET"})).await;
    assert_eq!(
        found,
        json!([{
            "file_name": "notes.md",
            "snippet": "secret plan",
            "matching_lines": [{"line_number": 1, "line": "secret plan"}]
        }])
    );
}

#[tokio::test]
async fn reserved_and_nested_names_are_refused() {
    let (_, provider) = memory();
    let ctx = prepare(&provider, &AgentSession::new()).await;
    for name in ["memories.md", "MEMORIES.md", "x_description.md"] {
        let result = call(
            &ctx,
            "file_memory_write",
            json!({"file_name": name, "content": "x"}),
        )
        .await;
        assert!(
            result
                .as_str()
                .unwrap()
                .contains("reserved for internal use"),
            "{result}"
        );
    }
    let nested = call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "dir/notes.md", "content": "x"}),
    )
    .await;
    assert!(nested
        .as_str()
        .unwrap()
        .contains("must not be written into a subdirectory"));
    assert_eq!(
        call(
            &ctx,
            "file_memory_read",
            json!({"file_name": "dir/notes.md"})
        )
        .await,
        "File 'dir/notes.md' not found."
    );
    let traversal = call(&ctx, "file_memory_read", json!({"file_name": "../x.md"})).await;
    assert!(traversal
        .as_str()
        .unwrap()
        .starts_with("Could not read file '../x.md': Invalid path"));
    let replace = call(
        &ctx,
        "file_memory_replace",
        json!({"file_name": "memories.md", "old_string": "a", "new_string": "b"}),
    )
    .await;
    assert!(replace
        .as_str()
        .unwrap()
        .contains("reserved for internal use"));
}

#[tokio::test]
async fn replace_edits_a_unique_match_or_reports_why_not() {
    let (_, provider) = memory();
    let ctx = prepare(&provider, &AgentSession::new()).await;
    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "n.md", "content": "a b a"}),
    )
    .await;

    let refused = call(
        &ctx,
        "file_memory_replace",
        json!({"file_name": "n.md", "old_string": "a", "new_string": "c"}),
    )
    .await;
    assert!(refused.as_str().unwrap().contains("occurs 2 times"));
    assert_eq!(
        call(
            &ctx,
            "file_memory_replace",
            json!({"file_name": "n.md", "old_string": "a", "new_string": "c", "replace_all": true}),
        )
        .await,
        "Replaced 2 occurrence(s) in 'n.md'."
    );
    assert_eq!(
        call(&ctx, "file_memory_read", json!({"file_name": "n.md"})).await,
        "c b c"
    );
    assert_eq!(
        call(
            &ctx,
            "file_memory_replace",
            json!({"file_name": "missing.md", "old_string": "a", "new_string": "c"}),
        )
        .await,
        "File 'missing.md' not found."
    );
}

#[tokio::test]
async fn grep_line_numbers_address_replace_lines() {
    let (_, provider) = memory();
    let ctx = prepare(&provider, &AgentSession::new()).await;
    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "todo.md", "content": "- buy milk\r\n- call bob\n- file taxes\n"}),
    )
    .await;

    let found = call(&ctx, "file_memory_grep", json!({"regex_pattern": "bob$"})).await;
    let hit = &found[0]["matching_lines"][0];
    assert_eq!(hit["line_number"], 2);
    assert_eq!(hit["line"], "- call bob\n");

    let edited = call(
        &ctx,
        "file_memory_replace_lines",
        json!({"file_name": "todo.md", "edits": [
            {"line_number": hit["line_number"], "new_line": "- call alice\n", "expected_line": hit["line"]},
            {"line_number": 1, "new_line": ""}
        ]}),
    )
    .await;
    assert_eq!(edited, "Replaced 2 line(s) in 'todo.md'.");
    assert_eq!(
        call(&ctx, "file_memory_read", json!({"file_name": "todo.md"})).await,
        "- call alice\n- file taxes\n"
    );

    let stale = call(
        &ctx,
        "file_memory_replace_lines",
        json!({"file_name": "todo.md", "edits": [
            {"line_number": 1, "new_line": "x\n", "expected_line": "- call bob"}
        ]}),
    )
    .await;
    assert!(stale
        .as_str()
        .unwrap()
        .contains("does not match the expected text"));
}

#[tokio::test]
async fn an_invalid_or_long_pattern_is_reported_to_the_model() {
    let (_, provider) = memory();
    let ctx = prepare(&provider, &AgentSession::new()).await;
    let invalid = call(&ctx, "file_memory_grep", json!({"regex_pattern": "("})).await;
    assert!(invalid
        .as_str()
        .unwrap()
        .starts_with("Could not search memory files: Invalid regex pattern"));
    let long = call(
        &ctx,
        "file_memory_grep",
        json!({"regex_pattern": "a".repeat(300)}),
    )
    .await;
    assert!(long.as_str().unwrap().contains("too long"));
}

#[tokio::test]
async fn sessions_are_isolated_unless_they_share_a_scope() {
    let (_, provider) = memory();
    let one = prepare(&provider, &AgentSession::new()).await;
    let two = prepare(&provider, &AgentSession::new()).await;
    call(
        &one,
        "file_memory_write",
        json!({"file_name": "n.md", "content": "mine"}),
    )
    .await;
    assert_eq!(
        call(&two, "file_memory_read", json!({"file_name": "n.md"})).await,
        "File 'n.md' not found."
    );

    let store = Arc::new(InMemoryAgentFileStore::new());
    let shared = FileMemoryProvider::new(store).with_scope("user-1");
    let a = prepare(&shared, &AgentSession::new()).await;
    let b = prepare(&shared, &AgentSession::new()).await;
    call(
        &a,
        "file_memory_write",
        json!({"file_name": "n.md", "content": "ours"}),
    )
    .await;
    assert_eq!(
        call(&b, "file_memory_read", json!({"file_name": "n.md"})).await,
        "ours"
    );
}

#[tokio::test]
async fn scopes_map_to_distinct_single_folders() {
    let store: Arc<dyn AgentFileStore> = Arc::new(InMemoryAgentFileStore::new());
    let folder = |scope: &str| {
        FileMemoryProvider::new(store.clone())
            .with_scope(scope)
            .working_folder(None)
            .unwrap()
    };
    assert_eq!(folder("user-1"), "user-1");
    assert_ne!(folder("a/b"), folder("a_b"));
    assert_ne!(folder("Alice"), folder("alice"));
    assert!(!folder("tenants/alice").contains('/'));
    assert!(!folder("..").is_empty());
    assert!(FileMemoryProvider::new(store.clone())
        .with_scope("x".repeat(300))
        .working_folder(None)
        .is_err());
}

#[tokio::test]
async fn without_a_scope_or_session_the_run_fails() {
    let (_, provider) = memory();
    let mut ctx = SessionContext::new(vec![Message::user("hi")]);
    assert!(matches!(
        provider.before_run(&mut ctx).await,
        Err(Error::Configuration(_))
    ));
}

#[tokio::test]
async fn custom_instructions_replace_the_default() {
    let (_, provider) = memory();
    let provider = provider.with_instructions("Use memory.");
    let ctx = prepare(&provider, &AgentSession::new()).await;
    assert_eq!(ctx.instructions.as_deref(), Some("Use memory."));
}

/// A store whose reads always fail.
struct UnreadableStore(InMemoryAgentFileStore);

#[async_trait]
impl AgentFileStore for UnreadableStore {
    async fn write(&self, path: &str, content: &str) -> Result<()> {
        self.0.write(path, content).await
    }
    async fn create_new(&self, path: &str, content: &str) -> Result<()> {
        self.0.create_new(path, content).await
    }
    async fn read(&self, _path: &str) -> Result<Option<String>> {
        Err(Error::tool("disk on fire"))
    }
    async fn delete(&self, path: &str) -> Result<bool> {
        self.0.delete(path).await
    }
    async fn list_children(&self, directory: &str) -> Result<Vec<FileStoreEntry>> {
        self.0.list_children(directory).await
    }
    async fn file_exists(&self, path: &str) -> Result<bool> {
        self.0.file_exists(path).await
    }
    async fn create_directory(&self, path: &str) -> Result<()> {
        self.0.create_directory(path).await
    }
}

#[tokio::test]
async fn store_errors_become_tool_results_and_an_unreadable_index_is_skipped() {
    let provider =
        FileMemoryProvider::new(Arc::new(UnreadableStore(InMemoryAgentFileStore::new())));
    let ctx = prepare(&provider, &AgentSession::new()).await;
    assert!(ctx.messages.is_empty());
    assert_eq!(
        call(&ctx, "file_memory_read", json!({"file_name": "n.md"})).await,
        "Could not read file 'n.md': disk on fire"
    );
    // The index rebuild reads descriptions, so the write reports the failure.
    let written = call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "n.md", "content": "x"}),
    )
    .await;
    assert_eq!(written, "Could not write file 'n.md': disk on fire");
    // A search skips files it cannot read rather than failing.
    assert_eq!(
        call(&ctx, "file_memory_grep", json!({"regex_pattern": "x"})).await,
        json!([])
    );
}

#[tokio::test]
async fn an_agent_run_sends_the_tools_and_instructions() {
    type Calls = Arc<Mutex<Vec<(Vec<Message>, ChatOptions)>>>;

    #[derive(Clone, Default)]
    struct Recorder(Calls);

    #[async_trait]
    impl ChatClient for Recorder {
        async fn get_response(
            &self,
            messages: Vec<Message>,
            options: ChatOptions,
        ) -> Result<ChatResponse> {
            self.0.lock().unwrap().push((messages, options));
            Ok(ChatResponse::from_text("ok"))
        }
        async fn get_streaming_response(
            &self,
            _messages: Vec<Message>,
            _options: ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    let client = Recorder::default();
    let (_, provider) = memory();
    let agent = Agent::builder(client.clone())
        .context_provider(Arc::new(provider))
        .build();
    let mut session = agent.create_session();
    agent
        .run(vec![Message::user("hi")], Some(&mut session))
        .await
        .unwrap();

    let (messages, options) = client.0.lock().unwrap()[0].clone();
    assert_eq!(options.tools.len(), 7);
    assert!(messages[0].text().contains("## File Based Memory"));
}

// region: stores

#[tokio::test]
async fn the_in_memory_store_ignores_case_and_keeps_written_names() {
    let store = InMemoryAgentFileStore::new();
    store.write("Docs/Plan.MD", "a").await.unwrap();
    store.write("Docs/sub/x.txt", "b").await.unwrap();
    store.write("top.txt", "c").await.unwrap();
    assert_eq!(
        store.read("docs/plan.md").await.unwrap().as_deref(),
        Some("a")
    );
    assert_eq!(
        store.list_children("").await.unwrap(),
        [
            FileStoreEntry::directory("Docs"),
            FileStoreEntry::file("top.txt")
        ]
    );
    assert_eq!(
        store.list_children("DOCS").await.unwrap(),
        [
            FileStoreEntry::directory("sub"),
            FileStoreEntry::file("Plan.MD")
        ]
    );
    assert!(store.create_new("docs/plan.md", "z").await.is_err());
    store.create_new("new.md", "z").await.unwrap();
    assert!(store.delete("DOCS/PLAN.md").await.unwrap());
    assert!(!store.delete("docs/plan.md").await.unwrap());
    assert!(store.read("../escape").await.is_err());
}

#[tokio::test]
async fn store_search_respects_scope_glob_and_recursion() {
    let store = InMemoryAgentFileStore::new();
    store.write("a.md", "needle").await.unwrap();
    store.write("b.txt", "needle").await.unwrap();
    store.write("sub/c.md", "Needle").await.unwrap();

    let names = |results: Vec<FileSearchResult>| {
        let mut names: Vec<_> = results.into_iter().map(|r| r.file_name).collect();
        names.sort();
        names
    };
    assert_eq!(
        names(store.search("", "needle", None, false).await.unwrap()),
        ["a.md", "b.txt"]
    );
    assert_eq!(
        names(
            store
                .search("", "needle", Some("*.md"), true)
                .await
                .unwrap()
        ),
        ["a.md", "sub/c.md"]
    );
    assert_eq!(
        names(store.search("sub", "needle", None, false).await.unwrap()),
        ["c.md"]
    );
}

#[tokio::test]
async fn the_file_system_store_round_trips_and_lists() {
    let dir = tempdir();
    let store = FileSystemAgentFileStore::new(dir.join("root")).unwrap();
    // Nothing is created until the first write.
    assert!(!dir.join("root").exists());
    assert!(store.list_children("").await.unwrap().is_empty());

    store.write("notes/a.md", "one\ntwo").await.unwrap();
    store.write("b.md", "three").await.unwrap();
    assert_eq!(
        store.read("notes/a.md").await.unwrap().as_deref(),
        Some("one\ntwo")
    );
    assert_eq!(store.read("missing.md").await.unwrap(), None);
    assert_eq!(
        store.list_children("").await.unwrap(),
        [
            FileStoreEntry::directory("notes"),
            FileStoreEntry::file("b.md")
        ]
    );
    assert!(store.create_new("b.md", "x").await.is_err());
    let found = store.search("", "TWO", None, true).await.unwrap();
    assert_eq!(found[0].file_name, "notes/a.md");
    assert_eq!(found[0].matching_lines[0].line_number, 2);
    assert!(store.delete("b.md").await.unwrap());
    assert!(!store.delete("b.md").await.unwrap());
    assert!(store.write("../escape.md", "x").await.is_err());
    assert!(store.write("/abs.md", "x").await.is_err());

    // A file blocking a parent directory is reported as such.
    store.write("blocker", "x").await.unwrap();
    let err = store.write("blocker/inner.md", "x").await.unwrap_err();
    assert!(err.to_string().contains("not a directory"), "{err}");

    // Non-UTF-8 content is an error to read and skipped by search.
    std::fs::write(dir.join("root/bin.dat"), [0xff, 0xfe, b'x']).unwrap();
    assert!(store.read("bin.dat").await.is_err());
    assert!(store
        .search("", "x", Some("*.dat"), false)
        .await
        .unwrap()
        .is_empty());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn the_file_system_store_refuses_symbolic_links() {
    let dir = tempdir();
    let root = dir.join("root");
    let outside = dir.join("outside");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("secret.md"), "secret").unwrap();
    std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
    std::os::unix::fs::symlink(outside.join("secret.md"), root.join("file-link.md")).unwrap();

    let store = FileSystemAgentFileStore::new(&root).unwrap();
    assert!(store.read("link/secret.md").await.is_err());
    assert!(store.read("file-link.md").await.is_err());
    assert!(store.write("link/new.md", "x").await.is_err());
    assert!(store.list_children("link").await.is_err());
    assert!(store.list_children("").await.unwrap().is_empty());
    assert!(store
        .search("", "secret", None, true)
        .await
        .unwrap()
        .is_empty());
    assert!(!outside.join("new.md").exists());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn file_memory_works_over_the_file_system_store() {
    let dir = tempdir();
    let store = Arc::new(FileSystemAgentFileStore::new(&dir).unwrap());
    let provider = FileMemoryProvider::new(store).with_scope("user 1");
    let ctx = prepare(&provider, &AgentSession::new()).await;
    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "n.md", "content": "persisted", "description": "Note"}),
    )
    .await;
    let folder = provider.working_folder(None).unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.join(&folder).join("n.md")).unwrap(),
        "persisted"
    );
    assert_eq!(
        std::fs::read_to_string(dir.join(&folder).join("memories.md")).unwrap(),
        "# Memory Index\n\n- **n.md**: Note\n"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("af-file-store-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn delete_refuses_the_index_and_description_files() {
    let (store, provider) = memory();
    let session = AgentSession::new();
    let ctx = prepare(&provider, &session).await;
    let folder = provider.working_folder(Some(&session)).unwrap();
    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "plan.md", "content": "steps", "description": "The plan"}),
    )
    .await;
    for name in ["memories.md", "MEMORIES.md", "plan.md_description.md"] {
        let result = call(&ctx, "file_memory_delete", json!({"file_name": name})).await;
        assert!(
            result
                .as_str()
                .unwrap()
                .contains("reserved for internal use"),
            "{result}"
        );
    }
    for kept in ["memories.md", "plan.md_description.md"] {
        assert!(store
            .file_exists(&format!("{folder}/{kept}"))
            .await
            .unwrap());
    }
}

#[tokio::test]
async fn files_differing_only_by_extension_keep_their_own_descriptions() {
    let (_, provider) = memory();
    let ctx = prepare(&provider, &AgentSession::new()).await;
    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "notes.md", "content": "a", "description": "Markdown notes"}),
    )
    .await;
    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "notes.txt", "content": "b", "description": "Plain notes"}),
    )
    .await;
    assert_eq!(
        call(&ctx, "file_memory_ls", json!({})).await,
        json!([
            {"name": "notes.md", "type": "file", "description": "Markdown notes"},
            {"name": "notes.txt", "type": "file", "description": "Plain notes"}
        ])
    );
    // Deleting one leaves the other's description alone.
    call(&ctx, "file_memory_delete", json!({"file_name": "notes.md"})).await;
    assert_eq!(
        call(&ctx, "file_memory_ls", json!({})).await,
        json!([{"name": "notes.txt", "type": "file", "description": "Plain notes"}])
    );
}

#[tokio::test]
async fn ls_finds_a_description_written_under_different_letter_case() {
    let (store, provider) = memory();
    let session = AgentSession::new();
    let ctx = prepare(&provider, &session).await;
    let folder = provider.working_folder(Some(&session)).unwrap();
    // What a case-insensitive file system leaves after `Plan.md` is rewritten
    // as `plan.md` with a new description: the file keeps its first name and
    // the sidecar takes the new one.
    store
        .write(&format!("{folder}/Plan.md"), "c")
        .await
        .unwrap();
    store
        .write(&format!("{folder}/plan.md_description.md"), "Second")
        .await
        .unwrap();
    assert_eq!(
        call(&ctx, "file_memory_ls", json!({})).await,
        json!([{"name": "Plan.md", "type": "file", "description": "Second"}])
    );
}

#[tokio::test]
async fn a_name_too_long_for_its_description_file_is_refused_before_writing() {
    let dir = tempdir();
    let store = Arc::new(FileSystemAgentFileStore::new(&dir).unwrap());
    let provider = FileMemoryProvider::new(store).with_scope("user 1");
    let ctx = prepare(&provider, &AgentSession::new()).await;
    let folder = dir.join(provider.working_folder(None).unwrap());
    // 245 bytes fits a file name, but not with `_description.md` added.
    let long = format!("{}.md", "n".repeat(242));
    let result = call(
        &ctx,
        "file_memory_write",
        json!({"file_name": long, "content": "x", "description": "Note"}),
    )
    .await;
    assert!(result.as_str().unwrap().contains("too long"), "{result}");
    assert!(!folder.join(&long).exists(), "the file was written anyway");
    // The longest name that leaves room for the description still works.
    let longest = format!("{}.md", "n".repeat(255 - "_description.md".len() - 3));
    assert_eq!(
        call(
            &ctx,
            "file_memory_write",
            json!({"file_name": longest, "content": "x", "description": "Note"}),
        )
        .await,
        format!("File '{longest}' written with description.")
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn a_shared_scope_gets_instructions_that_do_not_promise_isolation() {
    let (_, provider) = memory();
    let ctx = prepare(&provider.with_scope("user-1"), &AgentSession::new()).await;
    assert_eq!(
        ctx.instructions.as_deref(),
        Some(SCOPED_FILE_MEMORY_INSTRUCTIONS)
    );
    assert!(!SCOPED_FILE_MEMORY_INSTRUCTIONS.contains("isolated from other sessions"));
    // Custom instructions still win over either default.
    let (_, provider) = memory();
    let custom = provider
        .with_scope("user-1")
        .with_instructions("Use memory.");
    let ctx = prepare(&custom, &AgentSession::new()).await;
    assert_eq!(ctx.instructions.as_deref(), Some("Use memory."));
}

#[tokio::test]
async fn dot_only_scopes_get_their_own_usable_folders() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let mut folders = Vec::new();
    for scope in [".", ".."] {
        let scoped = FileMemoryProvider::new(store.clone()).with_scope(scope);
        let ctx = prepare(&scoped, &AgentSession::new()).await;
        call(
            &ctx,
            "file_memory_write",
            json!({"file_name": "n.md", "content": scope}),
        )
        .await;
        assert_eq!(
            call(&ctx, "file_memory_read", json!({"file_name": "n.md"})).await,
            scope
        );
        folders.push(scoped.working_folder(None).unwrap());
    }
    assert_eq!(folders, ["~scope-2e", "~scope-2e2e"]);
}

/// A path is never both a file and a directory, in either built-in store.
async fn check_file_directory_invariant(store: &dyn AgentFileStore) {
    store.write("a", "file").await.unwrap();
    assert!(store.write("a/b", "x").await.is_err());
    assert!(store.create_new("a/c", "x").await.is_err());
    assert!(store.create_directory("a").await.is_err());
    assert!(store.create_directory("a/d").await.is_err());

    store.write("dir/child", "x").await.unwrap();
    assert!(store.write("dir", "x").await.is_err());

    let names: Vec<_> = store
        .list_children("")
        .await
        .unwrap()
        .into_iter()
        .map(|e| (e.is_file(), e.name))
        .collect();
    assert_eq!(names, [(false, "dir".to_string()), (true, "a".to_string())]);
}

#[tokio::test]
async fn in_memory_store_keeps_files_and_directories_apart() {
    let store = InMemoryAgentFileStore::new();
    check_file_directory_invariant(&store).await;
    // Paths ignore case here, so `DIR` is the directory `dir`.
    assert!(store.create_new("DIR", "x").await.is_err());
}

#[tokio::test]
async fn file_system_store_keeps_files_and_directories_apart() {
    let dir = tempdir();
    check_file_directory_invariant(&FileSystemAgentFileStore::new(&dir).unwrap()).await;
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn paths_with_a_colon_are_refused() {
    // On Windows these name alternate data streams, which listings never
    // show, and `memories.md::$DATA` would slip past the reserved names.
    for path in ["notes.md:hidden", "memories.md::$DATA", "dir/a:b"] {
        assert!(normalize_relative_path(path, false).is_err(), "{path}");
    }
    let (_, provider) = memory();
    let ctx = prepare(&provider, &AgentSession::new()).await;
    let result = call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "memories.md::$DATA", "content": "x"}),
    )
    .await;
    assert!(result.as_str().unwrap().contains("':'"), "{result}");
}

#[tokio::test]
async fn an_empty_created_directory_exists_in_the_in_memory_store() {
    let store = InMemoryAgentFileStore::new();
    store.create_directory("Empty/Inner").await.unwrap();
    // Listed by its parent even with nothing in it, and empty itself.
    let root: Vec<_> = store
        .list_children("")
        .await
        .unwrap()
        .into_iter()
        .map(|e| (e.is_file(), e.name))
        .collect();
    assert_eq!(root, [(false, "Empty".to_string())]);
    let inner: Vec<_> = store
        .list_children("empty")
        .await
        .unwrap()
        .into_iter()
        .map(|e| e.name)
        .collect();
    assert_eq!(inner, ["Inner"]);
    assert!(store.list_children("empty/inner").await.unwrap().is_empty());
    // And it can't be overwritten by a file, as on disk.
    assert!(store.write("empty", "x").await.is_err());
    assert!(store.write("empty/inner", "x").await.is_err());
    store.write("empty/inner/n.md", "x").await.unwrap();
}

#[tokio::test]
async fn builder_methods_work_on_a_cloned_provider() {
    let (_, provider) = memory();
    let clone = provider.clone();
    let scoped = clone.with_scope("user-1").with_source_id("mine");
    assert_eq!(scoped.working_folder(None).unwrap(), "user-1");
    // The original is unchanged.
    assert!(provider.working_folder(None).is_err());
}

#[tokio::test]
async fn an_oversized_glob_is_a_tool_error_not_a_panic() {
    let (_, provider) = memory();
    let ctx = prepare(&provider, &AgentSession::new()).await;
    call(
        &ctx,
        "file_memory_write",
        json!({"file_name": "n.md", "content": "text"}),
    )
    .await;
    let huge = "[a-z]*".repeat(100_000);
    for (tool, args) in [
        ("file_memory_ls", json!({"glob_pattern": huge})),
        (
            "file_memory_grep",
            json!({"regex_pattern": "text", "glob_pattern": huge}),
        ),
    ] {
        let result = call(&ctx, tool, args).await;
        assert!(
            result.as_str().unwrap().contains("256 characters or fewer"),
            "{tool}: {result}"
        );
    }
    assert!(matches_glob("n.md", Some(&huge)).is_err());
}
