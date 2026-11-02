# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps
may break APIs).

## [Unreleased]

## [0.15.0] — 2026-11-02

Upstream drift from the window ending at `ca65085e` (2026-11-01), recorded
in [`docs/drift/2026-11-02-ca65085e.md`](docs/drift/2026-11-02-ca65085e.md)
and three further rounds against the same baseline in
[`round-2`](docs/drift/2026-11-02-ca65085e-round-2.md),
[`round-3`](docs/drift/2026-11-02-ca65085e-round-3.md) and
[`round-4`](docs/drift/2026-11-02-ca65085e-round-4.md).
Round four takes the two security items earlier rounds had left for a
decision, and clears the rest of the queue. A review of the official Azure
SDK for Rust crates is in [`docs/azure-rust-sdk.md`](docs/azure-rust-sdk.md).

### Added (parity push)

Closing items from `SCOPE.md`'s "Not yet ported" list:

- Agent hooks (`experimental-agent-hooks`), the rest of the control
  contract: the `sequential/run_all`, `parallel/strictest` and
  `parallel/unanimous` composition profiles (`Composition`,
  `AgentHooksOptions::composition`), the approval seam
  (`ApprovalResolver`, `approval_resolver_fn`, `ApprovalRedactor`; a
  liftable deny is consulted, the echo rule enforced, and `on_approval`
  stops or resumes the fold), the `jcs-sha256` identity provider (RFC 8785
  + SHA-256 over the closed projection, now the default, as upstream) and
  custom providers (`IdentityProvider`), and `agent_shutdown` with reason
  `cancelled` when a run's future is dropped. **Breaking (experimental):**
  `CompositionRecord` is now `Composition`, `InterceptionRecord::fold_truncated`
  is `Option<bool>` (defined for the sequential profiles only), records
  gain `resolved_by`, and records carry identities by default (use
  `IdentityProvider::Null` for the previous identity-unbound records).
- Foundry hosting: `/invocations_ws` WebSocket invocations (feature
  `websocket`): the host runs the handler with a session and the request
  context, closes with 1000 or 1011, logs the close, and sends keepalive
  pings when `WS_KEEPALIVE_INTERVAL` is set.
- Core: `OtelExport::with_span_processor` adds span processors to the export
  pipeline. Foundry hosting (`otel`): `FoundryEnrichmentSpanProcessor`,
  installed by `configure_observability`, stamps the agentserver's Foundry
  identity and baggage ids on every span; request spans join the caller's
  W3C baggage.
- Foundry hosting: `ResponsesHostServer::serve` first fails background
  responses a previous process left `queued`/`in_progress`
  (`fail_interrupted_responses`, with upstream's recovery error);
  `ResponseStore::response_ids` lists stored responses (implemented by the
  in-memory and file stores).
- Foundry hosting: with `session-store` and a hosted
  `FoundrySessionStoreProvider`, stored conversation turns under
  `HistorySource::Agent`/`Service` no longer fail with a storage conflict
  when saving their session: the turn keeps one request-scoped store from
  snapshotting its inherited session through the run.
- Foundry hosting (`websocket`): `/invocations_ws` reads client frames on
  a separate task, so a handler that isn't reading no longer blocks its own
  output, keepalive pings or the close frame.
- Foundry hosting: a cancel that arrives after a background turn has stored
  its final response still closes the turn's replay on that response's final
  event and schedules its eviction, so replay streams no longer hang.
- Foundry hosting: `DELETE /responses/{id}` also deletes the agent session
  the response saved under `HistorySource::Agent`/`Service`, keeping the
  conversation's own; session saves share the response's cancel/delete lock,
  so a late write cannot recreate a deleted snapshot.
- Background responses: `ChatOptions` and `AgentRunOptions` gain
  `background` and `continuation_token`; updates gain `operation_state`
  (in progress or terminal), and a terminal update clears the aggregated
  token. `Agent` rejects a continuation token sent with new messages, needs
  a session for background runs, skips history and context providers when
  resuming, and records an unfinished reply only once the operation
  completes. The OpenAI Responses client submits with `background: true`,
  returns `OpenAIContinuationToken` while a response is queued or in
  progress, polls with `GET /responses/{id}`, and resumes a stream with
  `?stream=true&starting_after=N`.
- Foundry hosting: as upstream, `store=false` turns keep no approval state.
  An `mcp_approval_response` sent with `store=false` is rejected with a 400
  that says it needs `store=true` (it was previously resolved from the
  approval store), and the 400 for an unknown approval id says that approval
  requests from `store=false` responses are not kept.
- OpenAI Responses: resuming an interrupted background stream no longer
  truncates function-call arguments. `OpenAIContinuationToken` gains
  `function_calls` (output index → call id), so argument fragments after the
  cut reach their call, and a function call the resumed stream never saw
  announced is taken whole from its `response.output_item.done`.
- Claude agent (review fix): control handlers the CLI started (tool calls,
  approval and hook callbacks) are aborted when the run is dropped or
  cancelled, and a `control_cancel_request` aborts the matching handler, so
  no tool side effect starts or continues after cancellation.
- MCP/Foundry hosting: Agent Skills `archive` (ZIP) entries are loaded in
  memory with upstream's safety limits and digest checks (`ArchiveOptions`,
  `load_skills_with`). `FoundryToolbox::skills_options` passes them through,
  and `ToolboxSkillsProvider` discovers lazily, caches, refreshes on an
  interval (or every run), and keeps its previous skills when a refresh fails.
- Foundry hosting: with `HistorySource::Agent`/`Service`, each stored
  conversation turn records the session it inherited before it runs, so
  continuing from a cancelled or failed conversation turn with
  `previous_response_id` restores that exact session rather than the
  conversation's later state.
- FIDES for MCP (review fix): an MCP error result whose server `_meta.ifc`
  label is untrusted or confidential has its details withheld, instead of
  its text reaching the model unlabeled through a locally trusted/public tool.
- Foundry hosting: with `HistorySource::Agent`/`Service`, a cancelled or
  failed response can be continued, inheriting its predecessor's session (or
  its conversation's, or a fresh one). Cancel/delete coordination entries are
  released once no turn, cancel or delete holds them, instead of being kept
  per cancelled or deleted response.
- Foundry hosting: paused hosted workflows accept `mcp_approval_response`
  replies to approval requests and, with `experimental-computer-use`,
  `computer_call_output` replies to computer calls. Review fix: a cancel or
  delete racing a background turn's persistence now wins deterministically
  (per-response serialization), so a late write can no longer overwrite
  `cancelled`, recreate a deleted response, or update the conversation.
- Workflows (`experimental-computer-use`): a pending computer call can be
  cancelled by replying `{"call_id": …, "cancelled": true}` (alone or in an
  array) or `{"cancelled": true}` to cancel all; cancelled calls become
  `ErrorContent` (`CANCELLED_COMPUTER_CALL`) and the turn ends with the
  conversation as output instead of re-running the agent. `AgentExecutor`
  defers streamed updates while a computer call is pending.
- OpenAI Responses (`experimental-computer-use`): screenshot `detail` and a
  secondary `file_id` / `image_url` survive the round trip
  (`SCREENSHOT_DETAIL_KEY`, `SCREENSHOT_FILE_ID_KEY`,
  `SCREENSHOT_IMAGE_URL_KEY`).
- Foundry hosting (review fixes): a conversation's transcript and session
  state are written together through `ResponseStore::update_conversation`
  (in-memory: one record under one lock; `FileResponseStore`: one file by
  atomic rename, still reading the earlier two-file layout), persistence runs
  in a spawned task so a client disconnect cannot interrupt it, and a failed
  update puts the previous transcript and state back. A turn that produces
  no new session snapshot (a failed run, a workflow) keeps the state it
  inherited, so continuing from a failed response keeps session-scoped state
  such as FIDES labels. Custom `ResponseStore`s should override
  `update_conversation` to write atomically; the default writes the two
  halves in sequence.
- Computer use (`experimental-computer-use`, upstream `COMPUTER_USE`;
  umbrella `experimental-computer-use`, in `experimental`):
  `ComputerToolCallContent` / `ComputerToolResultContent` /
  `ComputerSafetyCheck` (new `Content::ComputerToolCall` /
  `ComputerToolResult` variants and `ToolKind::HostedComputer`, always
  present; **breaking** for exhaustive matches), validated constructors and
  `validate_result`, `tools::hosted_computer()`; the tool loop returns to
  the caller on a pending computer call; `AgentExecutor` pauses with a
  `ComputerUseRequest` and resumes with validated results; the OpenAI
  Responses client maps `computer_call` / `computer_call_output`; the
  Foundry Responses host emits and accepts computer items.
- Foundry hosting and MCP: `FoundryToolbox` fetches its bearer token for
  every request through a new per-request header hook
  (`McpStreamableHttpTool::header_provider`, `HeaderProvider`); with the
  `session-store` feature the `Agent`/`Service` history sources keep their
  agent sessions (history included) in a session store
  (`ResponsesHostServer::with_session_store_provider`) so continuations
  survive a restart, and without it the `Agent` source no longer drops local
  history between turns; Agent Skills served over MCP (SEP-2640:
  `agent-framework-mcp` feature `skills`, `McpClient::read_resource`,
  `skills::load_skills`, and `FoundryToolbox::skills_provider` behind
  foundry-hosting's `skills` feature; umbrella `mcp-skills` /
  `foundry-hosting-skills`); container-file, file and file-path citations in
  Responses output (core `Annotation` gains `additional_properties`,
  **breaking** for struct literals without `..Default::default()`; the
  OpenAI Responses client now keeps `index` and `container_id`). Review
  fixes: cancelling a background response that already finished no longer
  overwrites it with `cancelled`; Foundry approval round trip; unfinished
  predecessors are refused; continuations restore the predecessor's
  session state (so FIDES context labels and hidden variables carry across
  turns) via **breaking** `StoredResponse::session_state` and new
  `ResponseStore::conversation_state` / `set_conversation_state` (no-op
  defaults: custom stores must implement them to keep conversation state);
  host-kept sessions are stored under a stable namespace
  (`foundry-responses/{agent name}`), so factory-built agents resume them
  (sessions saved under the old per-agent-id key are not found).
- `agent-framework-foundry-hosting`: rich Responses output (reasoning,
  refusals, function calls and outputs streamed in order, MCP calls, image
  generation, shell calls, approval and OAuth consent requests, URL
  citations, usage, `incomplete` finishes), checkpoint-resumed workflow
  `request_info` pauses (`with_checkpoint_store`),
  `HistorySource::{AgentServer, Agent, Service}` and `with_prepare_options`,
  background stream replay (`GET /responses/{id}?stream=true&starting_after=N`),
  `FileResponseStore`, after-restore session binding and environment
  overrides (`after_restore_with_context`), Invocations OpenAPI/AsyncAPI
  document routes, and request spans joining the caller's `traceparent`
  (feature `otel`, forwarded by the umbrella's `foundry-hosting-otel`, adds
  OpenTelemetry parenting and `configure_observability`). Review fixes:
  Invocations keeps a service conversation id across turns, responses are
  persisted before their terminal event, and turns on one conversation are
  serialized.
- Agent hooks host declarations (`experimental-agent-hooks`, upstream
  `create_agent_hooks_middleware_from_declaration`): `HostDeclaration`
  (JSON documents `agent-hooks-declaration/1.0`, from text, value or file)
  validated through the SDK's pipeline with its refusal classes
  (`DeclarationError`), resolved against a `HostRegistry` of kinds,
  identity providers, approval resolvers and redactors, and built with
  `AgentHooksOptions::from_declaration` / `AgentHooks::from_declaration`.
  Bindings run only at their `at` points with their own timeouts
  (`InterceptionEmitter::register_at`), the approval resolver has its own
  timeout, and records carry `declaration`. Also (review finding): an
  enforced identity the provider cannot compute after a transform now
  fails the emission closed (`host_error:context_invalid`).
- `agent-framework-mcp` feature `fides` (`security` module), forwarded by
  the umbrella's `experimental-fides`: `SecureMcpToolProxy` /
  `secure_tool_definitions` declare every listed MCP tool in a
  `ToolSecurityRegistry` from restriction-only `ToolAnnotations` hints
  (`openWorldHint`, PUBLIC sink cap, local `annotation_override`s),
  re-declared as the server's tool list changes, and stamp IFC labels a
  server returns in `_meta.ifc` (combined with the local label, or
  authoritative with `trust_server_ifc`). `ToolDescriptor` gains
  `annotations` and `CallToolResult` gains `meta` (**breaking** for struct
  literals); core gains `security::set_authoritative_result_label`.
- FIDES security fix: a hidden result's model-visible variable reference
  carries only the label's integrity and confidentiality, and a tool
  result's embedded `security_label` contributes only its levels; its
  attacker-controlled metadata never reaches the model or the result label.
- `agent-framework-foundry-hosting` (umbrella feature `foundry-hosting`, in
  `full`): host agents and workflows as Azure AI Foundry hosted agents,
  ported from upstream `foundry_hosting` and the agentserver container
  protocol it builds on. `ResponsesHostServer` (sync, streamed and
  background responses, `previous_response_id` / `conversation`
  continuation, get/delete/cancel/input items), `InvocationsHostServer`
  (feature `session-store`: durable per-session agent state, scoped per
  platform user when hosted), `GET /readiness`, the snapshot lifecycle
  routes, Foundry state-store clients with session (ETag-checked),
  checkpoint and function-approval stores, and `FoundryToolbox`. Tested
  in-process against storage and MCP doubles.
- FIDES information-flow control (`experimental-fides`, upstream
  `ExperimentalFeature.FIDES`): `agent_framework_core::security` with
  integrity/confidentiality labels (`ContentLabel`, `combine_labels`,
  `check_confidentiality_allowed`), `LabelTrackingFunctionMiddleware`
  (tiered result labels, hiding untrusted results behind variable
  references, `[var_<id>]` expansion, context-label tracking, standing
  guidance), `PolicyEnforcementFunctionMiddleware` (untrusted-context,
  untrusted-argument, confidentiality and principal checks with an audit
  log; block, warn, inline approval handler, or session-bound pending
  approvals), the `quarantined_llm` and `inspect_variable` tools,
  `SecureAgentConfig`, and MCP annotation mapping. Tool declarations are
  registered by name (`ToolSecurityRegistry`). The umbrella `experimental` feature now also enables
  `experimental-fides` and the `sql-server`, `mongodb` and
  `azure-documentdb` stores.
- `agent-framework-claude` (umbrella feature `claude`, in `full`): the
  Claude Code CLI as an agent (`ClaudeAgent`), ported from upstream's
  `agent_framework_claude`. With no Rust Claude Agent SDK, it drives the
  CLI directly over the SDK's stream-json control protocol: one process per
  run, resuming the session's CLI conversation, streaming text and thinking
  deltas, usage, finish reason, cost and structured output. Function tools
  are served from an in-process MCP server (approval-gated via
  `on_function_approval`), and `can_use_tool` permission prompts, hooks and
  per-run model and permission-mode switches are supported. Tested against
  a scripted fake CLI (Unix); a live test needs `AF_CLAUDE_CLI_TEST`.
- `agent-framework-mongodb` (experimental, `VECTOR_STORES`; umbrella feature
  `mongodb`): a MongoDB Atlas vector store ported from upstream's
  `agent_framework_mongodb` over the `mongodb` 3.9 driver. BSON documents
  keyed by `_id`, one Atlas Vector Search index per vector field (a
  mismatched existing index is refused), `$vectorSearch` approximate or
  exact search, normalized scores, and portable filters compiled to MQL
  prefilters (anything inexpressible is refused, not dropped).
- `agent-framework-azure-documentdb` (experimental, `VECTOR_STORES`;
  umbrella feature `azure-documentdb`): an Azure DocumentDB (MongoDB vCore)
  vector store ported from `agent_framework_azure_documentdb`:
  `cosmosSearch` IVF / HNSW / DiskANN indexes (`IndexParameters`),
  `$search` with `k`, `nProbes`, `efSearch`, `lSearch`, MQL filters
  including list containment, TLS on and retryable writes off.
- `agent-framework-sql-server` (experimental, `VECTOR_STORES`; umbrella
  feature `sql-server`): a SQL Server 2025 vector store ported from
  upstream's `agent_framework_sql_server`, over `tiberius` (rustls). Native
  `VECTOR(n)` columns with exact `VECTOR_DISTANCE` search (cosine,
  Euclidean, dot), typed columns in an existing schema, scalar indexes,
  parameterized two-valued portable filters, one transaction per
  operation. Bytes fields and generated keys are not supported. Tested
  offline (statement building); live tests need `AF_SQL_SERVER_TEST_URL`.

- `agent-framework-duckdb` (experimental, `VECTOR_STORES`; depend on it
  directly: it is not re-exported by the umbrella crate, whose
  `--all-features` builds would otherwise link the bundled DuckDB library
  into every example binary): an embedded DuckDB vector store ported from upstream's
  `agent_framework_duckdb`. Typed columns, `FLOAT[n]`/`DOUBLE[n]` vectors,
  cosine distance/similarity, dot product and Euclidean search, upserts
  with `ON CONFLICT`, and parameterized portable filters (equality, `in`,
  ordered, text). List containment and explicit data indexes are refused,
  as upstream. DuckDB is bundled, so the crate builds without a system
  library.

- Typed workflow protocols: `workflow::protocol` (`ProtocolDescriptor`,
  `ProtocolBuilder`, `RequestPortInfo`), a defaulted
  `Executor::describe_protocol`, `FunctionExecutor::with_protocol`,
  `RequestInfoExecutor::with_types`, build-time refusal of an edge whose
  declared types do not meet (`ValidationType::TypeCompatibility`), and
  `Workflow::describe_protocol`, `reflect_executors`, `reflect_edges` and
  `reflect_ports`. Types are names, since payloads are JSON.
- MCP `2026-07-28` (upstream #9236): `McpProtocolMode` (`Auto` by default,
  `Legacy`, `Modern`) on `McpClient` and the MCP tools. `Auto` probes with
  `server/discover` and, when the server offers the modern era, sends
  version, client info and capabilities in every request's `_meta` (and
  `MCP-Protocol-Version` over HTTP) instead of the `initialize` handshake;
  otherwise it falls back to the handshake.
- `FoundryMemoryProvider`: `with_max_memories`, search and storage message
  filters, `ensure_stored_memories_deleted` and
  `ensure_memory_store_created`.
- Smaller parity options: `AgentResponse::to_updates`, `Agent::chat_client`,
  `memory::with_message_source` / `message_source_type`,
  `SkillsProvider::with_filter`, `with_instructions_prompt`,
  `with_load_skill_approval` and `with_read_skill_resource_approval`,
  `TodoProvider::with_suppressed_todo_list_message` and
  `with_todo_list_message_builder`, and
  `harness::all_tools_auto_approval_rule`.
- Workflow run controls: `WorkflowContext::request_halt` (stop after the
  superstep), `WorkflowRunStream::cancel`, per-run `session_id` on
  `WorkflowRun` and `WorkflowRunStream`, a defaulted `Executor::reset`
  called before each fresh run, `WorkflowContext::add_warning` with the new
  `WorkflowEvent::Warning` / `SubworkflowWarning` variants (forwarded from
  nested workflows), `SequentialBuilder::with_chain_only_agent_responses`,
  `ConcurrentBuilder::with_aggregator` and
  `AgentExecutor::with_reassign_other_agents_as_users`.
- Workflow telemetry: runs emit `workflow.build`, `workflow.run`,
  `executor.process {id}`, `message.send` and `edge_group.process {type}`
  spans (upstream's names and attributes) through `tracing`;
  `WorkflowBuilder::with_telemetry(WorkflowTelemetryOptions)` turns kinds off
  and opts into payload capture.
- Compaction depth: the group model (`compaction::index`:
  `CompactionMessageIndex`, `CompactionMessageGroup`, `CompactionGroupKind`),
  `CompactionTriggers`, `PipelineCompactionStrategy`,
  `TriggeredCompactionStrategy`, the LLM-backed
  `SummarizationCompactionStrategy` over the new `AsyncCompactionStrategy`
  (`CompactionProvider::with_async_strategy`,
  `AgentBuilder::with_async_compaction`), `ToolResultCompactionStrategy`
  with a custom formatter, and the two-phase
  `ContextWindowCompactionStrategy`.
- Skill scripts: `SkillScript` (inline closures run in-process, or files
  discovered in a skill's directory, `.py` by default) and the
  `run_skill_script` tool, which requires approval by default
  (`SkillsProvider::with_script_approval`). File scripts run through the
  application's `SkillsProvider::with_script_runner`; there is no default
  runner. `Skill` gains `scripts`, `path` and the frontmatter fields
  (`license`, `compatibility`, `allowed_tools`, `metadata`);
  `FileSkillsSource` gains `with_script_extensions` and `with_script_filter`.
  `Agent::context_providers` exposes the configured providers.
- Azure Monitor export: `agent-framework-core`'s `azure-monitor` feature
  (umbrella `azure-monitor`) adds `OtelExport::with_azure_monitor` and
  `OtelExport::azure_monitor_from_env` (`APPLICATIONINSIGHTS_CONNECTION_STRING`),
  sending the pipeline's traces and metrics to Application Insights, the
  counterpart of upstream's `configure_azure_monitor`.
- Message source attribution (upstream `_attribution`): context messages
  carry `additional_properties["_attribution"]` with their `source_id`
  (`ContextProvider::source_id`, defaulting to the provider's type name),
  `source_type` and any `origin_session_ids`. `SessionContext` gains
  `extend_messages`, `extend_messages_from` and `get_messages`; the agent
  attributes whatever a provider adds.
- `SessionContext::agent` (`AgentInfo`: id, name, description): context
  providers see the agent they run for in `before_run`, as upstream's
  `agent` argument.
- `AgentBuilder::require_per_service_call_history_persistence` (upstream's
  option of the same name): with a local history provider, each model call
  of the tool loop records what it added, so a run that fails midway keeps
  its completed turns. `create_harness_agent` turns it on, as upstream.
- `agent-framework-redis`: `RedisVectorStore` / `RedisVectorCollection`
  (port of upstream `agent_framework_redis._vector_store`), HASH or JSON
  documents indexed by Redis Search, behind the crate's
  `experimental-vector-stores` feature (turned on by the umbrella crate's
  feature of the same name). Scores are native distances, reported with the
  new `VectorSearchResult::SCORE_KIND_DISTANCE`.
- **Harness complete** (`experimental-harness`): `FileAccessProvider`,
  `TodoFileStore`, `BackgroundAgentsProvider` with the
  `background_tasks_running` loop callbacks, `MemoryContextProvider` with
  `MemoryStore` / `MemoryFileStore`, and `create_harness_agent`
  (`HarnessAgent`). File-access and file-memory edits now share one lock
  per store and folder, as upstream.
- **Evaluation** (`experimental-evals`): `evaluate_agent`,
  `evaluate_workflow`, `LocalEvaluator`, the keyword / tool-call /
  function checks and `EvalResults` CI assertions; `FoundryEvals` (Foundry's
  built-in and generated rubric evaluators over the Evals API) behind
  `agent-framework-foundry`'s `evals` feature.
- **`agent-framework-qdrant`**: the Qdrant vector store (umbrella feature
  `qdrant`, experimental).
- **`agent-framework-postgres`**: the PostgreSQL + pgvector vector store
  (umbrella feature `postgres`, experimental). Its live tests run when
  `AF_POSTGRES_TEST_URL` is set.
- `FoundryAgentChatClient` (in `agent-framework-foundry`): calls a Prompt or
  Hosted Agent defined in a Foundry project by `agent_reference`, dropping
  the model and tool declarations the service owns; `as_agent()` builds an
  `Agent` whose local function tools dispatch the service agent's calls.
  `AzureOpenAIResponsesClient::with_body_transform` is the hook it uses.
- `ValkeyChatHistoryProvider` (in `agent-framework-redis`): port of .NET's
  Valkey history provider, resolving each session's conversation from its
  state bag rather than being pinned to one session.
- **Cloud streaming**: Anthropic on Foundry, Vertex and Bedrock now stream
  (Bedrock via a new AWS event-stream decoder), and the Bedrock Converse
  client streams through `converse-stream` instead of buffering.
- `ContentUnderstandingContextProvider` (`agent-framework-azure`, feature
  `content-understanding`; umbrella `azure-content-understanding`): analyzes
  supported attachments with Azure Content Understanding, strips them from
  the model input, injects the rendered document (or uploads it through a
  `FileSearchBackend`), resumes slow analyses on later turns, and offers
  `list_documents`.
- File-based skills (`file-skills` feature, in `full`): `FileSkillsSource`
  discovers `SKILL.md` packages per the Agent Skills spec (validated
  frontmatter, resources by extension and depth, links refused);
  `SkillsProvider::from_paths`; the harness builder's `skills_paths`.
- Message injection: `session::enqueue_messages` / `pending_messages`
  queue messages for a run's next model call, including from a tool
  mid-run; a queue left at the end of a turn earns another call. Built into
  the function-invocation loop for non-streaming calls.
- Context providers can rewrite `SessionContext::input_messages`; the model
  receives the rewritten input, as upstream allows.
- `ContextProvider::after_run_in_session`: the completion hook with the
  run's session. It defaults to `after_run`, and agents now call it.
- `SupportsAgentRun::description()` (defaulted), reported by `Agent`,
  `WorkflowAgent`, `A2AAgent` and `CopilotStudioAgent`; `Agent::tools()`.

### Changed (parity push)

- **Breaking:** `ValidationType` gains `TypeCompatibility`.
- **Breaking:** `WorkflowEvent` gains `Warning` and `SubworkflowWarning`
  variants; exhaustive matches need arms for them.
- **Behavior change:** mixed declaration-only batches follow upstream
  #9050 and #9203. With a session, a declaration-only call's local siblings
  no longer run on the same turn: the batch pauses, the siblings are held in
  session state, and they run once, under the current tool policy, on the
  request that answers every declaration-only call (a sibling whose tool now
  requires approval pauses for it instead). Without a session they still run
  on the same turn. In an approval batch, a declaration-only call is returned
  for the caller to answer rather than turned into an approval request.
- Message injection now also applies to streamed runs: messages queued on
  the session are streamed as their own updates and sent with the request,
  and a message queued mid-turn earns another call on the same stream
  (upstream `_stream_injected_messages`). To make that follow-up call from
  inside the stream, `FunctionInvokingChatClient` holds its inner client in
  an `Arc`, and its `ChatClient` impl requires `C: 'static`.
- **Behavior change:** the Responses parsers (OpenAI, Azure OpenAI,
  Foundry) return one assistant message per `message` output item, with the
  item's id as `message_id`, instead of a single message whose id was the
  response id (upstream #9768). Streamed text and refusal deltas carry their
  `item_id`, and `ChatResponse::from_updates` names an id-less trailing
  message with the first id it sees, so streamed and non-streamed responses
  agree. `response_id` is unchanged; read it instead of
  `messages[0].message_id`.
- **Breaking:** `WorkflowEvent::AgentRun` gains an `input` field (the
  conversation the agent ran over), which evaluation needs. Patterns that
  name every field need a `..`.
- `Message::text()` and the update text accessors join only plain text, as
  upstream does, so model reasoning no longer leaks into the answer text.

### Security (round four)

- `agent-framework-redis` no longer returns one scope's memories under
  another's. The `FT.SEARCH` path's scope `TAG` query over-matched — a value
  containing a comma is indexed as several tags, and matching is
  case-insensitive — so `user_id = "smith"` received `"smith,john"`'s
  memories. Every hit is now re-checked against the exact scope, and the
  query requires each comma-separated part, which also lets a provider whose
  own id contains a comma find its memories at all. No schema change or data
  migration (upstream's #9461 needed both).
- AG-UI `RUN_ERROR` events no longer carry a failed run's internal error
  text to the client; it is logged against the run id instead, as DevUI's
  were in round one.

### Fixed (round four)

- Transport errors now say what went wrong. A `reqwest` failure's own
  message reads the same for a timeout, a refused connection and a DNS
  failure; every provider crate now includes the error's cause chain, via a
  new `agent_framework_core::error::error_chain`.
- An MCP tool configured with a zero request timeout is refused at
  `connect` instead of failing every call (upstream #9314).
- MCP tools that require the Tasks extension, which this client does not
  implement, are no longer offered to the model (upstream #9746).

### Added (round four)

- Every HTTP-backed crate sends the framework `User-Agent`
  (`agent-framework-rust/<version> (<crate>)`). The format moves to
  `agent_framework_core::user_agent`; the provider crates that sent none,
  and the two that sent a one-off format (Copilot Studio, Purview), now use
  it.
- `Annotation::annotated_regions_coordinate_space` and
  `Annotation::content_regions()`: Anthropic document citations locate the
  quote in the *source*, and are now marked so, so they are not mistaken
  for spans of the answer (upstream #9436).
- `SequentialBuilder::ensure_trailing_user_turn` (upstream #8779), and
  `with_trailing_user_turn` on `AgentExecutor` / `AgentApprovalExecutor`.
- `HandoffBuilder::with_autonomous_mode(AutonomousMode)`: per-agent
  autonomous continuation with per-agent prompts and turn limits, as
  upstream's `with_autonomous_mode`.

### Fixed (round three)

- Every follow-up question after an Anthropic server-side web search was
  refused (upstream #9718). The search's `server_tool_use` and
  `web_search_tool_result` blocks replayed as a client `tool_use` and a
  `tool_result` inside the assistant turn, which the Messages API rejects. A
  `srvtoolu_` call now replays as `server_tool_use` and its result as the
  typed block it arrived as, `content` verbatim (including the
  `encrypted_content` the API validates). A server tool this client cannot
  replay faithfully is omitted with its result, keeping the request valid.
  Covers `web_search` and `web_fetch`, as upstream does.
- A turn the API paused (`stop_reason: "pause_turn"`, during a long server
  tool) was returned to the caller as finished — often a search call with no
  answer. All four Anthropic clients (direct, Foundry, Bedrock, Vertex) now
  continue the turn, up to 10 times and only while the paused content
  replays faithfully, merging the turns into one response with summed usage.
  Streaming continues too, forwarding every update as it arrives.

### Fixed (round two)

- `CheckpointStorage::list` could report a superseded checkpoint as the
  newest one (upstream #9261). Sorting by `timestamp_millis` alone is not a
  total order, and checkpoints are written one per superstep boundary, so two
  routinely share a millisecond; the tie then fell to the backend's
  enumeration order — `HashMap` order for `InMemoryCheckpointStorage`, which
  Rust randomises per process. A caller reading the end of the list as "the
  newest checkpoint" got an arbitrary one of the pair, differently on
  different runs: with the fix reverted the new regression test fails 9 times
  in 20. `list` now sorts with a new public `compare_checkpoints` over
  `(timestamp_millis, iteration_count, checkpoint_id)` — `iteration_count`
  being the superstep counter, which already encodes the supersession
  upstream recovers by walking `previous_checkpoint_id`. Applied to the
  in-memory, file and Cosmos stores, and now part of the trait's documented
  contract.
- Streamed Responses errors dropped the provider's `code` and `param`, and a
  flat `error` event lost its message entirely (upstream #9616). The handler
  read only `message`, and only from `response.error` or a top-level `error`
  key — but a flat `error` event carries `code` / `message` / `param` on the
  event itself, so every such event collapsed to a generic "response failed".
  A streamed `400` arrived as bare text with nothing naming the offending
  field. The event is now the last fallback for the error object, and `code`
  and `param` are formatted into the message as upstream formats its own,
  each clause omitted when absent. Content-filter classification is
  unchanged. `agent-framework-azure` and Foundry reuse the same SSE parser,
  so both get the fix.

### Added (round two)

- `otel-export`'s pipeline marks its telemetry with the
  `microsoft.agent-framework="True"` resource attribute (upstream #9589),
  shared by the tracer and meter providers. Azure's agent telemetry views —
  Foundry's, and the Application Insights experiences built on them — key off
  it to tell Agent Framework telemetry apart from an application's own, so a
  pipeline without it exported telemetry those views did not recognise. The
  value is the string `"True"`, matching upstream's default exactly. Two
  `ObservabilityConfig` doc comments that claimed this crate builds no OTel
  `Resource`, exporter or `MeterProvider` were corrected: they predate
  `otel-export`, and a reader who believed them would wire their own
  `Resource` and omit this attribute.

The first round against this baseline follows.

### Fixed

- Gemini reports which model served a run (upstream #9736).
  `ChatResponse::model` and every streamed update came back `None` whenever
  Gemini omitted `modelVersion` from the body, so a per-request
  `ChatOptions::model` override was invisible on the response. Both the
  non-streaming and streaming parsers now fall back to the model the request
  was sent to, and prefer a `modelVersion` when the body carries one. On the
  streaming path the fallback is applied after the "carries nothing new"
  test, so a keep-alive chunk is still skipped rather than becoming an update
  carrying only a model label. (Upstream fixed only its non-streaming path;
  here neither path had a fallback.)
- A round cap below 1 is coerced up to 1 in both Magentic and group chat
  (upstream #9502). Both check the limit before counting the round, so a cap
  of `0` ended the run on the loop's first pass: no ledger requested, no
  participant spoken, and only the "round limit" placeholder returned.
  Magentic coerces in `MagenticBuilder::build` rather than in the setter,
  because the cap can also come from a custom `MagenticManager`.
- A workflow exposed as an agent no longer loses its token counts
  (upstream #9720). An `AgentResponse` output was not a recognised shape: it
  fell through to the catch-all arm, arrived as one assistant message of raw
  JSON, and its `usage_details` were dropped — and `AgentResponse` was built
  with `..Default::default()` besides, so a workflow-as-agent reported no
  usage from any run. Its caller-facing messages are now taken and its usage
  merged and reported, including when every message of the response was
  filtered out.
- A workflow exposed as an agent no longer returns internal messages to its
  caller (upstream #8729). System prompts and tool results are workflow
  artifacts, and user messages get re-emitted — a group-chat orchestrator
  yields the whole conversation, the caller's own turn included. Only
  assistant messages survive, and an assistant message carrying a
  function-call envelope is dropped whole, since a call without its result is
  an invalid transcript for providers that validate call/result pairing on
  replay. Pending request-info messages are appended after the filter and are
  unaffected.

### Added

- `FoundryChatClient::with_timeout` bounds a single request, body included
  (upstream #9699). `reqwest` applies no timeout by default, so a hung
  Foundry endpoint stalled a run indefinitely with no way to say otherwise.
  Backed by a new `agent_framework_azure::azure_http_client_with_timeout`
  that shares a builder with `azure_http_client`, so a timed client keeps the
  framework `User-Agent` and the whole Azure crate family can reach it. The
  bound spans the response body, so on `get_streaming_response` it caps the
  stream's lifetime rather than the time to first token.
- `McpApprovalMode::PerTool` carries a `never_require` list and a `default`
  (upstream #9828), resolved in upstream's order: always, never, `default`,
  then no approval. `per_tool_except` expresses the policy this enables —
  gate every tool except a known-safe set, so a tool the server adds later is
  approved rather than let through. `with_default` and `with_never_require`
  compose onto a `per_tool` policy and are no-ops on the whole-server
  variants. `per_tool`'s own behaviour is unchanged. Upstream's
  single-string name normalisation and its no-`default` warning do not apply
  — see the pass notes.

### Security

- Recorded, not fixed: `agent-framework-redis`'s context provider can
  retrieve one scope's memory under another's. Its scope `TAG` fields are
  indexed with RediSearch's defaults (separator `,`, case-insensitive) and
  the value is written unescaped, so `user_id = "smith,john"` is indexed as
  two tags and matches a provider scoped to `user_id = "smith"`. Upstream's
  fix (#9461) hashes scope values into opaque tokens and is `[BREAKING]`
  with a migration; so is the smaller schema fix. See `SCOPE.md` and the
  pass notes.

---

Upstream drift from the window ending at `67ba500e` (2026-10-24), recorded
in [`docs/drift/2026-10-26-67ba500e.md`](docs/drift/2026-10-26-67ba500e.md).
Five changes, three of them defects that were live in this port. One
behaviour change is listed under *Changed*.

### Security

- The hosted Responses surface no longer tells a client why a run failed
  (upstream #9460). `devui`'s SSE `error` event and its 500 body carried the
  run error's own text, which can be a provider's raw HTTP error body, a
  tool's error text, a deserialization message naming internal types or a
  filesystem path from a message store — and the router is unauthenticated
  unless `with_bearer_token` was set. The caller now gets a fixed message;
  the detail is logged, correlated by the stream's response id, with the
  error's compact type always logged and its full text gated on
  `ENABLE_SENSITIVE_DATA`. The hosting layer's own validation messages are
  unchanged. AG-UI's `RUN_ERROR` events are **not** covered: upstream has
  not made that call for its AG-UI surface, so it is recorded as a
  divergence to decide rather than changed unilaterally.
- A cross-site `POST` no longer reaches the hosted agents (upstream #9575).
  The `Host` allowlist stops DNS rebinding but not a browser: a page on any
  origin can make the user's browser `POST
  http://localhost:8080/v1/responses` with `Host: localhost`, which the
  allowlist allows. `host_guard` now also rejects `Sec-Fetch-Site:
  cross-site` and a mismatched `Origin` on state-changing methods. See
  *Changed*.
- The bearer token is compared in constant time rather than with `==`, and
  the `Authorization` scheme is matched case-insensitively with the
  credential trimmed, per RFC 7235 and upstream.

### Fixed

- `allow_multiple_tool_calls` is no longer dropped by the Anthropic client
  when no `tool_choice` is set (upstream #8980). Anthropic exposes parallel
  tool use only as `tool_choice.disable_parallel_tool_use`, and the setting
  was a parameter of a conversion that only ran when a `tool_choice` was
  present — so a caller who disabled parallel calls and left `tool_choice`
  alone got them anyway, as did every turn after the function-invocation
  loop stopped pinning `tool_choice`. It is now carried on the default
  `auto` mode, as a derived default an explicit `tool_choice` or an
  `additional_properties["tool_choice"]` still overrides, and only when the
  request declares `tools`.
- The Bedrock client reports which model served a run (upstream #9399).
  `ChatResponse::model` was always `None`: Converse does not echo `modelId`
  and nothing supplied the request's, so a per-request `ChatOptions::model`
  override was invisible on the response. Affected the streaming path too,
  which derives its update from the same parsed response.
- The tool-invocation duration histogram emits the same dimensions as
  upstream (#9592): `gen_ai.operation.name`, `gen_ai.tool.name` and
  `gen_ai.tool.type` alongside `agent_framework.function.name` and
  `error.type`. The three were missing, which split a dashboard reading
  both languages' output. This port never had upstream's underlying bug of
  copying span attributes — and so tool arguments and results — onto the
  histogram; the new test pins the dimension set exhaustively so it cannot
  start.

### Changed

- `host_guard` (applied by `with_allowed_hosts` and `into_secure_router`) is
  now a local-access guard rather than a `Host` allowlist alone: it also
  rejects cross-site state-changing requests, as above. A caller serving the
  router to browsers on another origin should leave the allowlist unset,
  which is upstream's `AllowRemoteAccess`. Two deliberate divergences from
  upstream, both documented at the call site: the peer address is not
  checked (it needs `ConnectInfo<SocketAddr>`, which only exists if the
  caller serves the router with `into_make_service_with_connect_info`), and
  an `Origin`'s scheme is not compared (a server-side request carries no
  trustworthy scheme, and only one process can listen on a host and port).
- **Breaking (minor, pre-1.0):**
  `agent_framework_bedrock::convert::parse_response` now takes the request's
  `modelId` as a second argument, which is what lets it report the model (see
  *Fixed*). It matches
  `agent_framework_openai::responses::parse_response(value, store)`, which
  already takes request state this way. The module is public but the function
  is only called by the crate's own client; a direct caller passes the
  `modelId` it sent.

## [0.14.0] — 2026-10-21

A parity and roadmap release. Four upstream fixes from the window ending
at `02005995` (2026-10-21), recorded in
[`docs/drift/2026-10-21-02005995.md`](docs/drift/2026-10-21-02005995.md),
then four of the gaps that pass confirmed: a **session store**, a
**text-search context provider**, **file memory** for the agent harness,
and Cosmos chat metadata that no longer depends on the partition strategy.
It also fixes a bug where `Agent` sent earlier turns to a service that
already held them.

There are no breaking API changes. Four behaviour changes are listed under
*Changed*.

### Added

- **Session store** (`agent_framework_core::session_store`, behind the new
  `experimental-session-store` feature). Ports .NET's `AgentSessionStore`,
  including `DeleteSessionAsync` (upstream #8801), and Python's
  `SessionStore`/`FileSessionStore`. The `AgentSessionStore` trait saves,
  gets, gets-or-creates and deletes sessions by `AgentSessionStoreKey`, a
  session id with ordered named partitions. `InMemoryAgentSessionStore` and
  `FileAgentSessionStore` are included. The file store encodes each path
  component injectively (so two keys never share a file, even on a
  case-insensitive disk or where Windows reserves a device name) and writes
  through a temporary file and a rename. Snapshots carry an in-memory
  history provider's messages through the new provided method
  `ContextProvider::snapshot_history`, so a restored session carries on its
  conversation.
- **Text-search context provider** (`agent_framework_core::text_search`,
  stable). Ports .NET's `TextSearchProvider` and Go's `textsearch`.
  `TextSearchProvider` wraps any async search function. By default it
  searches the run's input before each call and adds the results ahead of
  it. With `TextSearchBehavior::OnDemandFunctionCalling` it offers a
  `Search` tool instead. Optional recent-message memory searches a
  follow-up together with the messages before it. A failed search does not
  fail the run, and the memory is updated under a lock so runs on clones
  of one session do not lose each other's messages. The default context
  format matches upstream byte for byte.
- **File memory for the agent harness** (`experimental-harness`). Ports
  upstream's file store and `FileMemoryProvider` from Python, .NET and Go:
  - `AgentFileStore`, with `InMemoryAgentFileStore` and
    `FileSystemAgentFileStore`. Paths are normalized and cannot leave the
    root. The disk store refuses symbolic links (Windows junctions
    included). Paths with a colon, a segment ending in `.` or a space, or a
    Windows device name such as `nul` or `COM¹` are refused on every
    platform, so a path means the same thing wherever the store runs.
  - Shared helpers for fnmatch-style globs, case-insensitive regex search
    (patterns capped at 256 characters, the whole search bounded by a
    10-second timeout), and literal and line-based edits.
  - `FileMemoryProvider` provides seven `file_memory_*` tools, keeps
    descriptions in sidecar files, and rebuilds a `memories.md` index that
    is given to the next run. Memories are kept per session by default, or
    per explicit scope with `with_scope`, which gets instructions saying
    the files are shared.
  - `regex` is a new optional dependency of `agent-framework-core`,
    enabled by `experimental-harness`.
- **Cosmos chat metadata** (upstream #9214). `CosmosChatMessageStore`
  gains `with_tenant_id` and `with_user_id`, which write `tenantId` and
  `userId` on every message document whatever the partition strategy. The
  new `CosmosPartitionStrategy` keeps `ThreadId` as the default, and adds
  `Hierarchical`, which partitions on tenant, user and thread with a
  MultiHash key. Both ids survive `serialize` and `from_state`.

### Changed

- `Agent` leaves history providers out of a run when the service stores
  the conversation: when the session has a service id, or the run's
  options carry a conversation id (a default or a per-run override). This
  ports .NET's `ResolveChatHistoryProvider`. It always behaves as .NET's
  `ClearOnChatHistoryProviderConflict` does and never warns or throws.
  The fix for #57 is under *Fixed*.
- `ToolApprovalRule::tool` keeps the tool name exactly as given (upstream
  #9557). It used to trim it, so a standing approval for `run_shell `
  approved `run_shell`. An empty or whitespace-only name is still refused.
- `ToolApprovalAgent` refuses a batch with two live approval requests under
  one id (upstream #9542). It discards the state held under the duplicated
  id and fails the run with `Error::AgentExecution`, so the answer to one
  request can no longer stand in for the other.
- An A2A task's status message is now kept for every state except
  `submitted`, `working` and `completed` (upstream #9534), and comes after
  the artifacts. For `input-required`, a task with artifacts and a question
  now returns both, not only the question.

### Fixed

- `Agent` no longer sends earlier turns twice when the service holds the
  conversation (#57). A reply with a conversation id, which OpenAI
  Responses gives by default, was adopted while the auto-attached
  `InMemoryHistoryProvider` kept loading and recording. The next request
  replayed the earlier turns and also set `conversation_id`. Streaming
  runs were affected in the same way.
- An approval answer repeated after its call had already run executed the
  call again (upstream #9535). The function-invocation loop now tracks
  approval requests as well as responses, and a response to a request that
  is already closed is settled on arrival.
- A failed, cancelled or rejected A2A task no longer drops the reason it
  gave in its status message (upstream #9534).

## [0.13.0] — 2026-10-19

A parity release from the upstream window `a75e803e..9d308951`
(2026-10-12 to 2026-10-19), recorded in
[`docs/drift/2026-10-19-9d308951.md`](docs/drift/2026-10-19-9d308951.md).
The headline is **AG-UI thread snapshots**: the AG-UI router can now
persist threads, replay them to a reconnecting client, and resume a paused
approval across requests, behind an explicit authorization scope. Alongside
it come six fixes, two of them gaps the review surfaced rather than commits
asking to be ported: the hosting Responses route discarded image and file
input, and Foundry OAuth consent requests had no producer anywhere in the
workspace.

**Two public API breaks**, both small and mechanical — see *Changed*. The
AG-UI router's behaviour without a snapshot store is unchanged, apart from
no longer emitting a `TOOL_CALL_END` for a call the run never started.

### Added

- **AG-UI thread snapshots** (`agent_framework_hosting::agui::snapshot`).
  `AgUiRouter::snapshot_store` persists each thread — transcript, shared
  state, pending interrupts, and the agent session's private state — in an
  `AgUiThreadSnapshotStore` (`InMemoryAgUiThreadSnapshotStore` included,
  bounded and process-local). With a store the router emits
  `STATE_SNAPSHOT` at the start of a run and `MESSAGES_SNAPSHOT` at the
  end, replays a thread to a request that sends no messages, records
  pending approvals as `RUN_FINISHED` interrupts, and resumes them from
  `resume` entries. Upstream's eight AG-UI hardening commits this window
  shape the design:
  - **Explicit authorization scope** (#9413, #9316). Snapshots are keyed by
    scope *and* thread id, and a store without an authorization mode does
    not build: `AgUiRouter::authorization_scope` resolves the scope from
    the request head (an `Err(status)` rejects the request before any state
    is read), and `AgUiRouter::single_user` declares one principal.
    `AgUiRouter::try_into_router` reports a bad configuration;
    `into_router` panics on one.
  - **Client state cannot reach server-owned session state** (#9368,
    tightened). A client value shaped like tool-approval state is never
    merged into the session — checking all five `ToolApprovalState`
    fields, where upstream checks three. `held_messages` is replayed into
    the inner run as input, so the two upstream misses let a client plant
    an approval response and run a tool unprompted. A session key the
    server owns, such as an agent's mode, cannot be overwritten either:
    the snapshot tracks which keys the client owns (ones it wrote that the
    server has not since changed), and keys under the framework's reserved
    `__af_` / `__ag_ui` prefixes are refused outright — the
    function-invocation loop reads its parked budget at the start of every
    run, so a client-seeded one would reset `max_function_calls` and
    `max_duration_seconds` on a fresh thread.
  - **Interrupts are bound to their thread** (#9365). A resume must answer
    exactly the interrupts stored for that scope and thread, or the run
    ends with `RUN_ERROR` and an `APPROVAL_RESUME_*` code. A thread with
    interrupts pending refuses any other turn (`APPROVAL_RESUME_REQUIRED`)
    until they are answered or cancelled, rather than saving over them. The approved
    call is rebuilt from the stored interrupt, never from the request, and
    the interrupts are spent before the agent runs, so an approved call
    runs at most once even if the run fails or the client disconnects.
    Turns on one thread are serialized within the process, so a
    double-click cannot resume an approval twice. A run whose final save
    fails ends with `RUN_ERROR` (`SNAPSHOT_SAVE_FAILED`) rather than
    `RUN_FINISHED`, which would advertise interrupts that were never
    stored; upstream logs the failure and finishes.
  - **The stored transcript is the server's record** (port of upstream's
    `_reconstruct_messages_from_thread_snapshot`). A client replaying
    history, with or without ids, does not duplicate it, and only user turns
    and results for server-issued tool calls extend it.
- Foundry **OAuth consent requests** now reach the caller. A hosted tool
  that needs the user to complete a consent flow produces
  `Content::OauthConsentRequest` on the non-streaming path, on
  `response.output_item.added`, and from the top-level
  `response.oauth_consent_requested` event (upstream #8976). The consent
  link must be HTTPS and parse with a real host before it is surfaced;
  anything else is dropped with a warning rather than handed to a human to
  click.
  `OauthConsentRequestContent` gained `id` (the provider's item id, which a
  workflow registers the request under) and the
  `OauthConsentRequestContent::new` / `with_id` constructors.
- The hosting **Responses route accepts multimodal input**. `input_image`
  and `input_file` content parts become data or URI content instead of
  being discarded, and a `refusal` part becomes a marked refusal. A `data:`
  URL keeps the media type it declares rather than being flattened to
  `image/*` (upstream #9298), which is what Anthropic and Gemini need to
  accept the attachment. A part with no URL is skipped and the rest of the
  message survives. `input_file` reads every spelling the
  Responses API accepts — `file_url`, `file_data` as a data URL or as bare
  base64 beside a `filename`, and `file_id` (as hosted-file content). An
  `input_image` given by `file_id` is skipped with a warning, as upstream
  rejects it: a hosted file carries no media type in the core model, so it
  would reach the provider as an `input_file`. A malformed `data:` URL is
  skipped with a warning rather than passed on as a URI no provider can
  fetch.

### Changed

- **Breaking:** `observability::record_error` takes an
  `&ObservabilityConfig`. The error's `Display` text is now recorded as
  `otel.status_message` only with content capture on (upstream #9288): a
  provider's 400 quotes the request back, a tool's failure prints the
  argument it choked on, and a transport error carries the URL, so it is
  sensitive data like any other. `error.type` and the ERROR status are
  still always recorded. `Agent` retains its `ObservabilityConfig` for the
  `invoke_agent` span, defaulting to `ObservabilityConfig::from_env()` —
  the rule `FunctionInvokingChatClient::new` already followed.
- **Breaking:** `OauthConsentRequestContent` gained an `id` field, so a
  struct literal naming every field needs updating. `::new` or
  `..Default::default()` do not.
- The AG-UI router emits `TOOL_CALL_END` only for a call this run
  started and has not closed, tracked per occurrence so a call id the
  provider reuses within a run still gets one end per start. A result or approval for a call started in
  an earlier request — a resumed approval — used to get an end with no
  matching start, which AG-UI clients reject (upstream does the same).
- Switch-case branches may share a target executor. `add_switch` builds one
  edge per distinct target and joins the branch labels of the cases that
  share it (upstream #9300); previously the duplicate reached `targets()`
  and `flat_edges()`, so validation and both graph renderings saw an edge
  that does not exist.

### Fixed

- A streamed update whose `author_name` differs from the message it would
  continue now starts a new message (upstream #9353). Two agents streaming
  in turn — a sequential orchestration, a handoff — emit assistant updates
  with no `message_id`, so their text was folded into one message
  attributed to whoever spoke last. An explicit `message_id` still wins: it
  relabels the message rather than splitting it.
- `add_fan_out` and `add_multi_selection` deduplicate their targets, so a
  broadcast declared with a repeated id delivers once (upstream #9432).
  Delivering twice ran the executor twice on one message and double-counted
  toward a fan-in sink's barrier. The runner's selector-free branch
  deduplicates too, since `EdgeGroup::FanOut` can be built by hand.
- The OpenAI Responses converter sends a remote `application/*` file by
  `file_url`. It put every such URI in `file_data`, which the service reads
  as the file's contents, so `https://example.com/report.pdf` arrived as a
  file whose bytes were that string. Inline `data:` URIs still go in
  `file_data`. `text/*` files (plain text, Markdown, CSV) are sent as
  `input_file` too, with a matching filename extension; the converter had
  no arm for them and dropped them. An inline `application/*` file is now
  named for its type as well (`file.pdf`, `file.docx`) rather than a bare
  `file`, which the service identifies files by.
- A Magentic progress ledger naming a participant that does not exist is
  retried against the existing `progress_ledger_retry_count` budget instead
  of ending the run on the first bad answer (upstream #9181). If every
  attempt names an unknown participant the last ledger is returned, so the
  orchestrator takes the invalid-speaker path it already has rather than
  reading the failure as a parse error and replanning.

## [0.12.0] — 2026-10-14

A parity release: ten fixes from the upstream window `301a43c..a75e803e`
(2026-10-05 to 2026-10-12), recorded in
[`docs/drift/2026-10-12-a75e803e.md`](docs/drift/2026-10-12-a75e803e.md).
Azure clients now identify themselves with a `User-Agent`, and chat spans
record which endpoint served the request.

**No public API breaks**, but two behaviour changes need attention.
`FunctionInvokingChatClient` now runs local tools that share a turn with a
declaration-only call instead of returning them unexecuted, and
`HandoffBuilder::build` rejects two participants whose handoff tool names
are the same once cut to 64 characters.

### Added

- `ChatClient::service_url()` returns the endpoint a client sends requests
  to, and `ObservableChatClient` records its host as `server.address` and
  its port as `server.port` on the `chat` span (upstream #9225, #9222).
  Upstream records the whole URL; this follows the OpenTelemetry semantic
  conventions instead, so these spans group with other network spans. The
  new `observability::server_address_and_port` does the split.
  `service_url()` defaults to `None`, and every wrapper forwards it. Every
  built-in chat client implements it: OpenAI chat completions and Responses,
  Azure OpenAI chat and Responses, Foundry (through its Responses client),
  Anthropic (direct, on Foundry, on Bedrock and on Vertex), Bedrock, Gemini,
  Mistral, Ollama, Foundry Local, GitHub Copilot and the AG-UI client.
- Azure requests now carry a framework `User-Agent`:
  `agent-framework-rust/<version> (<crate>)`. The new
  `agent_framework_azure::{azure_user_agent, azure_http_client,
  USER_AGENT_PRODUCT}` build it, and the Azure OpenAI chat, embeddings and
  Responses clients, Cosmos DB, Azure AI Search and Foundry use them. Azure
  reads this header for telemetry, throttling diagnostics and support cases.
- `AzureOpenAIResponsesClient::with_http_client` replaces the client that
  requests go through. Foundry chat uses it so its requests carry the
  Foundry `User-Agent` rather than the Azure OpenAI one; callers can use it
  for a proxy or timeouts.
- `handoff_tool_name` is exported, so marker tools attached by hand use the
  same name the coordinator resolves.

### Changed

- **Mixed declaration-only tool batches.** When a model turn calls both a
  declaration-only tool and a local tool, the local tool now runs on that
  turn. Its result comes back in a tool message alongside the caller-owned
  call, which is still returned intact. Before, the whole batch came back
  unexecuted, so the replayed turn carried an unanswered call that
  providers reject (upstream #9050).
- **Handoff tool names are capped at 64 characters**, the limit OpenAI,
  Anthropic and Gemini enforce. A long participant name used to make the
  provider reject the whole request. Routing uses the shortened name, and
  two participants that collide once shortened are a build error (upstream
  #9035). Only the offered names are compared, and a participant's offered
  name can no longer be overridden by another participant's alias, which
  used to make routing depend on registration order.

### Fixed

- **Bedrock extended thinking with tools.** `reasoningContent` blocks were
  dropped on the way in and on the way out. Converse rejects an assistant
  turn whose reasoning no longer matches its signature, so the turn after a
  tool result failed. Signed reasoning and redacted blocks now round-trip
  exactly (upstream #8936).
- **Magentic progress ledger.** Braces inside JSON string values ended the
  ledger object early, so a ledger answer that mentioned a brace failed to
  parse and stalled the round (upstream #9089).
- **Azure OpenAI `api-version`.** A blank `AZURE_OPENAI_API_VERSION` was
  sent as `?api-version=`, which the chat and embeddings routes answer with
  a 400. A blank value now keeps the dated default (upstream #9098).
- **Data URIs.** An omitted media type now defaults to `text/plain`, as
  RFC 2397 specifies. A second encoding marker such as
  `data:text/plain;gzip;base64,...` is refused; before, it was decoded as
  plain base64 and returned corrupted bytes (upstream #8918).
- **Workflow diagrams.** Executor ids that sanitize to the same Mermaid id
  (`step-a` and `step_a`, or any two non-Latin ids) rendered as one node,
  and a fan-in node could take an executor's id in Mermaid and DOT. Node
  ids are now unique (upstream #9082).
- **Failed workflow supersteps.** A failed superstep's staged shared-state
  writes stayed readable, and the next superstep committed them if the
  caller answered another pending request on the same run. They are now
  discarded on failure (upstream #8819).

## [0.11.0] — 2026-10-09

Three experimental ports and a new crate. The **agent harness** (loop, todo,
tool approval, agent mode) and **agent hooks** (AGENT-HOOKS-0.1 interception)
land behind `experimental-harness` and `experimental-agent-hooks`, and the
**local and Docker shell tools** ship stable in the new
`agent-framework-tools` crate. Hosting gains an **AG-UI client**, and Gemini
and Foundry can embed images.

**Breaking, in four places.** `EmbeddingClient::get_embeddings` takes
`Vec<EmbeddingInput>` (most callers change `text.to_string()` to
`text.into()`). APIs upstream still marks experimental move behind
`experimental-*` features, and the umbrella crate's `declarative` feature is
now `experimental-declarative-agents` and has left `full`. `SessionContext`
gains a public `session` field and `McpServerToolResultContent` an `error`
field, so code building either with a struct literal must set them. Pre-1.0,
minor bumps may break APIs; the sections below give the details.

### Agent hooks (experimental)

New `experimental-agent-hooks` feature (upstream `AGENT_HOOKS`): the
`agent_framework_core::agent_hooks` module implements the AGENT-HOOKS-0.1
control contract. `AgentHooks::build_agent` (or
`AgentBuilder::build_with_agent_hooks`) installs all eight interception points
as one unit — `agent_startup`, `input` and `agent_shutdown` in an agent
decorator, `output` as the first agent middleware, `pre_model_call` /
`post_model_call` around each model service call, and `pre_tool_call` /
`post_tool_call` as the first function middleware — and enforces the combined
verdicts fail closed. A run-level deny fails the run with the new
`Error::InterceptionBlocked`; a tool-level deny blocks that call and tells the
model why; a `host_error:*` deny anywhere halts the run. Transforms are
written back into messages, arguments and results, streaming is buffered
behind the verdicts, and denied output is never persisted to history. The
protocol layer (contexts, verdicts, `Interceptor`, `InterceptionEmitter`,
payload-free records) is implemented in Rust since no agent-hooks crate
exists; it supports the `sequential/first_deny` profile without an approval
resolver or identity provider.

The function-invocation loop now puts the model's `call_id` in
`FunctionInvocationContext::metadata["call_id"]`, as upstream does. A new
`Error::ToolRejected` lets function middleware fail a call with a message
written for the model: the loop records it as an error result (counted toward
`max_consecutive_errors_per_request`) and shows the model that message even
with `include_detailed_errors` off. The tool seam uses it for blocked calls
and for a `post_tool_call` transform of a failed call.

### Experimental agent harness

A new `experimental-harness` feature on `agent-framework-core` (forwarded by
the umbrella crate and included in `experimental`) adds the `harness`
module, a port of upstream's experimental harness:

- **Agent loop**: `LoopAgent` wraps an agent and re-invokes it until a
  condition says stop, with `max_iterations`, `next_message`,
  `record_feedback`, `inject_progress`, `fresh_context` and
  `return_final_only`. Conditions are closures or `LoopCallback`s. `Judge`
  asks a chat client whether the criteria are met, and `todos_remaining`
  (optionally limited to some agent modes) keeps going while todos are open.
  Upstream's `AgentLoopMiddleware` becomes a wrapper agent here, as in .NET,
  because Rust agent middleware cannot re-run its terminal.
- **Todo**: `TodoProvider` (context provider plus the `todos_*` tools) with a
  pluggable `TodoStore`; `TodoSessionStore` keeps the list in session state.
- **Tool approval**: `ToolApprovalAgent` adds standing "always approve"
  rules (per tool, or per tool and arguments), auto-approval rules, and
  presents several pending approvals one at a time. Auto-approval re-runs
  are capped per run (`with_max_auto_approval_iterations`, default 40, as
  .NET's `DefaultMaxAutoApprovalIterations`).
- **Agent mode**: `AgentModeProvider` with `mode_set` / `mode_get` tools,
  per-mode instructions and a one-time mode-change notification.

Not ported: file access, file memory, memory, background agents (so the
loop has no `background_tasks_running` condition), `TodoFileStore` and
`create_harness_agent`.

`SessionContext` gains a public `session: Option<AgentSession>` field, set by
`Agent` runs, so context providers can read and write per-session state, as
upstream's `before_run(session=...)` allows. Code that builds a
`SessionContext` with a struct literal must set it.

### Local shell tool (new crate `agent-framework-tools`)

Ports upstream's `agent_framework_tools.shell` (.NET
`Microsoft.Agents.AI.Tools.Shell`). Enable it through the umbrella crate's
`tools` feature, which re-exports it as `agent_framework::builtin_tools`.

- `LocalShellTool` runs model-written commands on the host, in a persistent
  shell (the default; `cd` and exports carry over) or one process per
  command. Timeouts kill the command's whole process group, output is held
  to a head/tail byte limit, and the working directory is re-anchored before
  each persistent command. `as_function()` returns the `run_shell` function
  for an agent.
- **Approval is required by default.** Building with
  `ApprovalMode::NeverRequire` fails unless `acknowledge_unsafe(true)` is
  also set.
- `ShellPolicy` is an allow/deny/custom pre-filter with no default
  patterns. It is documented as a UX filter, not a security boundary.
- `DockerShellTool` runs commands in a container through the `docker` (or
  `podman`) CLI, with no network, a read-only root, dropped capabilities, an
  unprivileged user and memory/process caps by default. `extra_run_args` that
  would undo those are rejected.
- `ShellEnvironmentProvider` is a context provider that tells the model which
  shell, OS, working directory and CLIs it has.
- `ShellResult` converts to the hosted-shell content types, which
  `agent_framework_core::types` now re-exports (`ShellToolCallContent`,
  `ShellToolResultContent`, `ShellCommandOutputContent`).

### AG-UI client

`agent_framework_hosting::agui::AgUiChatClient`, behind the new
`agui-client` feature (in `agent-framework-hosting`, and in the umbrella
crate, where it implies `hosting` and is part of `full`), is a `ChatClient`
for any AG-UI server, ported from upstream's `AGUIChatClient`. It posts a
`RunAgentInput` and maps the server's SSE events (text, tool calls and
results, reasoning, state and message snapshots, run finished/error, custom
and annotation events) to chat updates, streaming or not. The function tools
in a request are declared to the server; when its model calls one, an
`Agent` wrapping the client runs it locally and sends the result back, while
tools the server ran itself are left alone. `state_carrier` sends shared
state, and interrupt/resume options are forwarded in canonical form.
`AgUiEventConverter` and `messages_to_agui` are public for custom transports.

The feature is off by default, so serving AG-UI does not pull in `reqwest`.
The router and client share the wire model: `RunAgentInput` now also
serializes and gains `available_interrupts` and `resume` fields (ignored by
the router), and the event names are in `agui::event_type`.

### Multimodal embedding inputs (breaking)

`EmbeddingClient::get_embeddings` takes `Vec<EmbeddingInput>` instead of
`Vec<String>`, matching upstream's generic `EmbeddingInputT`. An
`EmbeddingInput` is one or more `Content` items embedded into a single
vector. Strings convert with `.into()`, so most callers change
`vec![text.to_string()]` to `vec![text.into()]`, and a custom
`EmbeddingClient` can call `EmbeddingInput::into_texts` to keep handling text
only.

- **Gemini** embeds images and other media (`Content::Data` inline,
  `Content::Uri` as `fileData`), alone or with text parts, as upstream does.
  A task is needed only when the batch has a text input.
- **Foundry** embeds images through the Models inference endpoint's
  `/images/embeddings` route, with an optional caption. A mixed batch is split
  across both routes and reassembled in input order. Set the image model with
  `with_image_model`, `FOUNDRY_IMAGE_EMBEDDING_MODEL` or the `image_model`
  option; it falls back to the text model.
- OpenAI, Azure OpenAI, Bedrock, Mistral and Ollama embed text only, as
  upstream does, and return an error naming the first non-text input.

### Feature stages (breaking)

APIs that upstream still marks experimental now sit behind `experimental-*`
cargo features and carry no semver promise, mirroring Python's
`@experimental` and .NET's `[Experimental]`. The rules are in
`docs/feature-stages.md`, and `cargo xtask features check` enforces them in
CI. To keep using a gated API, enable its feature:

| Feature | Gates | Crates |
|---|---|---|
| `experimental-vector-stores` | `agent_framework_core::vectors`, `CosmosVectorStore`, `AzureAISearchStore` | core, cosmos, azure-ai-search |
| `experimental-file-history` | `FileHistoryProvider` | core |
| `experimental-progressive-tools` | `LiveToolList`, and `FunctionInvocationContext::{tools, with_tools, add_tools, remove_tools}` | core |
| `experimental-declarative-agents` | the whole `agent-framework-declarative` crate | declarative |
| `experimental-harness` | `agent_framework_core::harness` (`LoopAgent`, `TodoProvider`, `ToolApprovalAgent`, `AgentModeProvider`) | core |
| `experimental-agent-hooks` | `agent_framework_core::agent_hooks`, `AgentBuilder::build_with_agent_hooks`, `Error::InterceptionBlocked` | core |

The umbrella crate re-exposes each feature under the same name and adds
`experimental` to turn them all on. **Its `declarative` feature is now
`experimental-declarative-agents`**, and `full` no longer includes it.
`agent_framework_cosmos::DEFAULT_VECTOR_API_VERSION` moves behind
`experimental-vector-stores` with the store it configures.

### Fixed

- **An approved tool re-ran on every later turn of a session.** The approval
  response stayed in session history and the function-invocation loop
  collected it again on each run, executing the tool every turn. As upstream's
  `_collect_approval_responses` does, a response is now settled once a later
  function result answers its `call_id` (or a later request reuses its id), and
  settled responses neither execute nor reach the model. The results resolving
  an approval are also returned ahead of the model's answer, so history records
  the approval as settled. Streaming and non-streaming runs share the fix.
  Settlement is decided before history compaction (a window could keep the
  response but drop the result that settled it) and before the no-tools fast
  path, so a settled approval never reaches the model on either route.
- **OpenAI Responses dropped completed hosted MCP calls.** An `mcp_call`
  output item (streamed: its completed `output_item.done` item, which carries
  the final arguments) is now parsed into
  `McpServerToolCall`/`McpServerToolResult` contents and replayed as one
  `mcp_call` input item carrying its output, as upstream does. A call that
  failed keeps its `error` (new `McpServerToolResultContent::error`) and
  replays it, so it does not look unfinished; upstream drops both the error
  and, when streamed, the arguments. A hosted
  approval replayed from local history is therefore followed by the call it
  approved, rather than standing alone as an approval still to act on.
  `McpServerToolCallContent` and `McpServerToolResultContent` are now exported
  from `agent_framework_core::types`.

## [0.10.0] — 2026-10-05

Upstream moved 119 non-merge commits in the week to `301a43c` (2026-10-05).
**Nine land on this port**, plus one Azure capability added on the back of a
commit that does not. Five are Azure-surface work and one of those is an
enforcement hole: a Purview-guarded *streamed* run was never policy-checked.
Two are fields a cloud API rejects outright, so the request failed rather than
degraded. The rest are a filter clause that dropped the rows it was meant to
return, a fan-out that could run one executor twice on one message, a replay
alignment that duplicated history, and an orchestrator decision that was
unreadable whenever the provider ignored `response_format`.

**Breaking, in three places.** `agent_framework_bedrock::convert::build_request`
takes the model id as a third argument — the request cannot be built correctly
without it, since which fields Converse accepts depends on the model (see
below). And `MagenticManager::plan`/`replan` take `&mut MagenticContext`,
because the task ledger they produce is the run's state rather than the
manager's; a custom manager updates its signatures and, if it decomposes a
plan, writes `context.task_ledger` instead of its own field.
`StandardMagenticManager::task_ledger()` is gone with the cache it read, and
so is `MagenticManager::current_task_ledger()` — the ledger is on the
context, and a trait method reading it off the manager could only ever
return a ledger belonging to no particular run. And `workflow::Condition` and
`workflow::Selection` now carry a `Result`, so code that builds one by
constructing the `Arc` directly returns `Ok(..)` or uses the new
`wrap_sync_condition` / `wrap_async_condition` / `wrap_selection` /
`wrap_async_selection` helpers; every builder and `Case` constructor that
takes a *closure* is unchanged, since both `bool` and `Result<bool>` closures
now satisfy the same bound.

### Added

- **Per-operation embedding options on the vector-collection provider**
  (`search_embedding_options`, `upsert_embedding_options`; upstream #8798).
  Searching and indexing are different operations, and for some providers the
  same text must be embedded differently for each — which is why the Gemini
  client above could not serve a vector store at all: its task is required
  and the two operations need different ones. `dimensions` is pinned from the
  vector field's declaration, and a conflicting value is refused at `build`
  rather than becoming a mis-shaped vector later. Leaving both unset is
  exactly the previous behaviour.
- **A Gemini embeddings client** (`GeminiEmbeddingClient`, `gemini` crate),
  speaking `batchEmbedContents` directly like this crate's chat client
  (upstream #8798). Gemini Embedding 2 conditions a vector on what the text is
  *for*, and takes that from a prefix on the text itself rather than a request
  field — `title: … | text: …` when indexing, `task: … | query: …` when
  searching — so the client applies it and sends no `taskType`, as upstream
  does. A task is therefore **required** per call
  (`EmbeddingGenerationOptions::with_task`, via the `GeminiEmbeddingOptions`
  trait): there is no safe default, since `RetrievalDocument` on a search
  query indexes the query as a document and no prefix at all silently opts
  out of the conditioning, and both failures are invisible in the response.
  `with_title` applies only when indexing, and is refused otherwise rather
  than dropped. The model is an allowlist (`gemini-embedding-2`,
  `gemini-embedding-2-preview`) because the prefix convention is Embedding
  2's: applying it to `gemini-embedding-001` would embed the prefix as
  literal text. Text only — upstream's multimodal inputs need a wider
  `EmbeddingClient` trait than this port has (recorded as a gap).
- **Token log-probabilities are surfaced** (upstream #8997). The OpenAI
  clients read `choices[].logprobs` on both paths and pass it through
  unchanged under `ChatResponse::additional_properties["logprobs"]`
  (`agent_framework_openai::convert::LOGPROBS_PROPERTY`). Across a stream the
  per-token entries accumulate, so the aggregated response carries every
  token's rather than the final chunk's — a deliberate divergence from
  upstream, which narrows to the last chunk and would have left this client's
  streaming and non-streaming paths answering the same question differently.
  Previously the field was dropped outright: a caller could ask for logprobs — the request half
  already worked, since `additional_properties` carries request-wide fields
  verbatim — and never see them. Requesting them still needs no typed option.
  The payload is passed through rather than modelled, because it is a scoring
  artifact to read or log and OpenAI has extended its shape before.
  Upstream's fix in this window was the streaming merge rule (a chunk with
  `"logprobs": null` must not clear the metadata gathered so far); here an
  explicit null is read as *no value*, so such a chunk contributes no key and
  `absorb_update` has nothing to clear. Azure OpenAI, Ollama, GitHub Copilot
  and Foundry Local inherit this, delegating both parsers wholesale.
- **Per-call `x-client-*` headers on the Foundry path** (upstream #8715,
  #8847). The Foundry Agent Endpoint forwards `x-client-`-prefixed headers
  transparently into the agent container, which makes them the channel for
  attesting *this run's* end user (`x-client-end-user-id`) — the one thing a
  multi-tenant caller cannot pin on the client, since one client serves every
  tenant. The port had no carrier for them at all, so upstream's validation
  fix landed on a capability that did not exist here.
  `agent_framework_foundry::FoundryClientHeaders` adds
  `with_client_header`/`with_client_headers` on `ChatOptions`, validating the
  `x-client-` prefix (case-insensitive), non-empty names and values, and the
  NUL/CR/LF refusals upstream added. Delivery is by construction rather than
  conditional: `ChatOptions` already reaches the transport, so none of
  upstream's agent decorator, policy registration or documented silent no-op
  is needed. The transport side is
  `agent_framework_azure::responses::CLIENT_HEADERS_PROPERTY`, a reserved
  `additional_properties` key lifted out of the request body rather than
  merged into it, and validated again where the request is built — the carrier
  is a public map here, unlike .NET's `internal` key, so the transport
  enforces the `x-client-` namespace itself. That covers the credentials and
  also the headers the request builder owns — `content-type` from
  `.json(body)`, `host` and `content-length` from the transport — any of
  which `reqwest` would have *appended* rather than replaced, putting two
  conflicting values on one request.
- **`AzureAISearchProvider::with_filter`** — an OData `$filter` applied by the
  service before ranking, so retrieval can be scoped to a tenant id, a
  document class or a security-trimming field. On a shared index this is an
  isolation control rather than a relevance knob: without it every run
  retrieves over the whole index, so one tenant's documents can land in
  another tenant's context. The semantic-mode equivalent of upstream's
  agentic `SearchIndexKnowledgeSourceParams(filter_add_on=…)` (#8673).
- **`WorkflowBuilder::add_multi_selection`** — a fan-out whose selection
  function picks a *subset* of the group's targets per message, the Rust
  analogue of upstream's `add_multi_selection_edge_group`. `add_switch`
  already covered picking exactly one branch, and `EdgeGroup::FanOut` already
  carried a `Selection` at runtime, but no builder method installed one, so
  the multi-target form was unreachable from the public API.
- **A Markdown-fence JSON scanner shared by group chat and Magentic.** A
  provider that ignores `response_format` wraps the orchestrator's decision in
  a code fence; the body of the fence is now a parse candidate. An opening fence must
  begin its line (CommonMark's rule, with up to three spaces of indent), so
  an inline backtick run in the surrounding prose cannot be mistaken for the
  start of the block and swallow the real one. A closing
  fence must be at least as long as its opening one and end its line, so
  backticks inside a JSON string value do not end the block and an outer
  fence may use four or more. Ports upstream's
  `extract_markdown_fence_bodies` (#8850).

### Fixed

- **The in-memory vector search cloned the whole collection per query**
  (upstream #8544). Every record was cloned out of the lock before any was
  looked at, so a `top: 5` search over 50k records copied all 50k and kept a
  second copy of each match while scoring. Scoring now happens in place and
  only the returned page is cloned. The lock is held for the scoring pass
  instead, which for a store meant for tests and development is the better
  trade. No behaviour change — this is the one entry here that is purely a
  performance one.
- **A failing workflow predicate was indistinguishable from one that said no**
  (upstream #8490). `Condition` returned a bare `bool`, so a predicate that
  could not reach a verdict — one that deserializes the payload, reads a
  field, parses a date — had nowhere to put the error and had to return
  `false`. In a switch/case group that means falling through to the **default
  branch**: a broken predicate looks like a working one and the message is
  delivered somewhere nobody chose. Upstream fixed the same fault from the
  other side, by no longer swallowing predicate exceptions. `Condition` and
  `Selection` now carry a `Result`, which the runner propagates — so an `Err`
  aborts the run instead of quietly routing. Writing an infallible predicate
  is unchanged: the new `IntoConditionResult` / `IntoSelectionResult` traits
  mean a `bool` closure and a `Result<bool>` one are both accepted wherever a
  predicate is taken, so existing `|m| m["x"] == 1` call sites compile as they
  were. The `wrap_*` helpers that adapt a closure are now public — the
  `Condition` docs had recommended them while they were crate-private.
- **A streamed response lost its continuation token.** `ChatResponse` and
  `AgentResponse` carry the handle for resuming a long-running operation, but
  the *update* types did not, so `response_to_updates` dropped it and no
  streamed run could be resumed. It now rides the final update and is taken
  back by aggregation, like `finish_reason`. Surfaced by review of the
  Purview fix below, which sets that token precisely so a caller can pick a
  blocked run back up.
- **A shared Magentic manager could replan one run from another run's facts**
  (upstream #8581). `StandardMagenticManager` cached the decomposed task
  ledger on itself, and two runs can share one manager — the same `Arc` handed
  to two builders, or `Workflow::run` taking `&self`. The cache was read at
  three places: `replan`, which builds its "update these facts" prompt from
  the previous ledger, and the plan-review and stall-intervention requests,
  where a human is shown facts and a plan to approve. So a run could be told
  to update the other run's facts — corrupting its own reasoning, not just a
  display, which the previous note on this recorded incorrectly — and a
  reviewer could be asked to sign off the other run's plan. Nothing raised an
  error. The ledger now lives on `MagenticContext::task_ledger`, the run's own
  state, which also puts it inside the run's checkpoint: previously the ledger
  died with the process, so the first stall after a resume failed with
  `replan() called before plan()`. `MagenticContext::reset` deliberately keeps
  it, since the replan that follows a reset exists to update it.
- **A Purview-guarded streamed run was not policy-checked** (upstream #8702).
  `PurviewAgentMiddleware` skipped its response-phase check whenever
  `ctx.is_streaming`, carried over from Python's "streaming responses are not
  supported for post-checks". That reason does not hold here:
  `Agent::run_stream` routes a middleware-guarded run through `run_core` and
  replays the *result* as updates, so the hook is handed the complete response
  and a replacement reaches the caller — the buffering upstream had to add to
  `ResponseStream` is already how this port streams such a run. The check now
  runs whenever a response exists. A blocked response carries an explicit
  allowlist of the evaluated response's control fields (`response_id`,
  `conversation_id`, `created_at`, `finish_reason`, `usage_details`,
  `continuation_token`) so the caller can still identify and resume the
  operation. Everything else is withheld, `value` and
  `additional_properties` alike: both are content channels, and `logprobs`
  (new in this release) puts the generated token *strings* in the latter, so
  returning that map would let a caller reconstruct the text the check just
  blocked. The list is an allowlist rather than "all but `value`" so that a
  field added to `AgentResponse` later is withheld until someone decides it
  is safe. `PurviewChatMiddleware` is gated on a response being present
  rather than on `is_streaming`, which is what is actually true of it: the
  streaming chat path honours only pre-call mutation, so a token stream never
  reaches that hook — attach the agent middleware to enforce one.
- **A negated `eq` filter dropped the Cosmos documents it should return**
  (upstream #8771). `eq` compiled to `(IS_DEFINED(x) AND x = @p)`, and Cosmos
  SQL evaluates `x = @p` on a stored null to UNDEFINED, which `NOT` leaves
  UNDEFINED — so `NOT (field eq value)` excluded exactly the null-valued
  documents that match it. `eq` now carries the `NOT IS_NULL` guard that
  `in`/`not_in` and the ordered operators already had.
- **Two Bedrock fields Converse rejects outright.** This client reaches
  Bedrock through `Converse` only (its streaming path aggregates one
  `Converse` call), so `guardrailConfig.streamProcessingMode` — valid only on
  `ConverseStream` — failed every guarded request; it is now dropped and the
  rest of the guardrail config still sent. And a `modelId` naming a Prompt
  Management prompt (`:prompt/`) takes its instructions, tools and inference
  settings from the prompt, so Converse rejects a request that also sets
  `inferenceConfig`, `system`, `toolConfig` or `additionalModelRequestFields`;
  those are dropped with one warning naming them. Prompt-*router* ARNs are
  ordinary inference targets and are untouched (upstream #8680).
- **`audio/mpegurl` and `audio/mpeg4-generic` were sent to OpenAI as MP3.**
  `audio_format` matched `"mpeg"` as a substring, but one of those is an M3U
  playlist and the other an MPEG-4 RTP payload. The media type is now
  normalized (parameters stripped, case-folded) and compared against the
  registered MP3 aliases — and WAV is matched against its own allowlist for
  the same reason, since `audio/x-wavpack` is a different codec that a
  substring test labelled as WAV. A non-matching sibling takes the
  unsupported-content
  path instead of failing at the provider after the upload (upstream #8787).
- **A fan-out selection could run one executor twice on one message.** A
  selection naming the same target more than once — two tags routing to the
  same executor — delivered once per mention, which also double-counted
  toward a downstream fan-in barrier. Each selected target is now delivered to
  once (upstream #8739). A selection naming a target outside its own group is
  now a routing error rather than a silent drop, matching the contract an
  explicit `target_id` already had.
- **A replay of a trimmed transcript duplicated stored history.** Alignment
  only looked for *all* of stored history inside a run's input, so a caller
  that keeps its own window — a UI holding the last N turns — matched nothing
  and had every replayed turn appended again on each run. The stored tail is
  now matched against the incoming head (upstream #8800); the longest overlap
  wins, and an overlap of a single id-less user message, or one consuming all
  of the input, is still read as real input rather than a replay. History
  *injection* follows the same alignment and now prepends exactly the part a
  trimmed replay is missing, instead of all of it (duplicating the window) or
  none of it (losing the turns before it).

### Changed

- **`async-trait` moved to 0.1.92** (lockfile only; the requirement stays
  `"0.1"`). Through 0.1.89 the macro pushed `#[must_use]` onto every
  generated method, which Rust 1.99's clippy reports as `double_must_use`
  on a method already returning a `#[must_use]` boxed future — 33 instances
  across `agent-framework-core`, all in trait definitions, none of them
  written by hand. 0.1.92 stops emitting the attribute. This bumps
  `async-trait` onto `syn 3`, so `syn` 2 and 3 now coexist in the lockfile;
  both declare `rust-version` 1.71, well under this workspace's 1.88 floor.
- **The one `Atomic::fetch_update` call site allows its deprecation.** Rust
  1.99 renamed the method to `try_update`, which does not exist on the
  declared `rust-version = "1.88"` — so the rename cannot be taken until the
  floor rises, and the attribute carries that note. Behaviour is unchanged;
  the method itself was only renamed.

## [0.9.0] — 2026-09-30

Four values the code already had and never read, a Foundry surface it could
not reach, and tool calls that neither hosting surface put on the wire.
Nearly all of it failed in the permissive direction: a cut-off turn reported
as a completed one, a blocked prompt reported as an answer, a hosted MCP
allowlist of none reported as all.

**Breaking, in three places.** `AgentResponse` and `AgentResponseUpdate` each
gain a `finish_reason` field, so a struct literal for either without
`..Default::default()` needs it. `agent_framework_hosting::ResponseObject`
gains `incomplete_details` on the same terms, and its `output` changes from
`Vec<OutputMessage>` to `Vec<OutputItem>` now that a response can carry
function calls beside its message — `OutputItem::as_message()` recovers the
old view.

**One behaviour change worth calling out.** The in-memory vector store's
`hamming` score is now the *fraction* of differing coordinates rather than the
raw count, matching upstream. Ranking is unchanged; a `score_threshold` tuned
against the old raw count needs dividing by the vector width.

### Added

- **Microsoft Foundry managed memory** (`FoundryMemoryProvider`, `foundry`
  crate). A `ContextProvider` that searches a Foundry memory store for
  relevant memories before a run and writes the turn back after it, spoken
  directly against the project data plane
  (`{endpoint}/memory_stores/{name}:search_memories` and `:update_memories`,
  `api-version=v1`, bearer-scoped to `FOUNDRY_SCOPE`). Memories are isolated
  by scope — `with_scope` pins one, otherwise the session id is used, and
  nothing is read or written when neither is available. Search and update
  cursors chain incrementally, the user-profile fetch happens once per scope,
  and every service failure is logged rather than raised, since `after_run`
  also runs on the agent's own failure path. Provider state is keyed by scope,
  so a provider shared by `Arc` across runs cannot let one session read
  another's profile or resume its cursor; because `after_run` is handed no
  session, a provider that has served more than one session declines to write
  rather than guess a scope — set `with_scope` when sharing one.
- **Foundry embeddings from a project endpoint.** `FoundryEmbeddingClient`
  previously spoke only the Foundry *Models* inference endpoint, so a Foundry
  project holding an embedding deployment could not be embedded against
  without provisioning a second surface. `with_project_endpoint` derives the
  resource-scoped `{resource}/openai/v1/embeddings` route from a project
  endpoint (`openai_model_base_url`), path-versioned and Entra-only. `from_env`
  reads `FOUNDRY_PROJECT_ENDPOINT` (and `FOUNDRY_ENDPOINT`, the name
  `FoundryChatClient` takes first) and still prefers `FOUNDRY_MODELS_ENDPOINT`
  when both are set (upstream #8454).
- **Tool calls on both hosting surfaces.** Core leaves a
  `FunctionCallContent` intact for the caller to execute, but neither host
  put it on the wire: `/v1/chat/completions` and `/v1/responses` serialized
  text only, so a client was told a call had been requested and given no id,
  name or arguments to act on. `/v1/chat/completions` now emits
  `message.tool_calls` (with `content: null` for a call-only turn, as OpenAI
  sends) and, when streaming, `delta.tool_calls` fragments keyed by a stable
  per-call `index`. `/v1/responses` now emits `function_call` output items
  as siblings of the assistant message, announced once over the stream as
  `response.output_item.added` and then fed
  `response.function_call_arguments.delta`. A call streamed across several
  updates is announced once, with its fragments reassembled, and the
  terminal payload reuses the ids the client already saw. Arguments go out
  as a JSON *string* on both surfaces, as the wire format has it. Consequently a `tool_calls` / `function_call` finish reason
  now passes through whenever the response actually carries a call, instead
  of degrading to `stop`; it still degrades when the turn declares none.

  Only calls that are still **unanswered** are serialized. Core keeps a
  call and its `FunctionResultContent` together in the response — after a
  local tool ran, and when a provider executed a hosted tool itself — and
  advertising those would ask the client to re-run work already done. The
  streaming surfaces hold calls until the stream ends to make that
  decision, because with local tools the result arrives in a later update
  than the call; a call's arguments therefore arrive in one delta rather
  than forming incrementally.

  The inbound half works too: `/v1/chat/completions` reads `tool_calls` and
  `tool_call_id` off the follow-up request, and `/v1/responses` reads
  `function_call` and `function_call_output` input items, so an executed
  call reaches the agent as a `FunctionCallContent` / `FunctionResultContent`
  pair rather than as empty text.
- `agent_framework_hosting::{OutputItem, OutputFunctionCall}`.
- `AgentResponse::finish_reason` / `AgentResponseUpdate::finish_reason`, and
  `incomplete_details` on the hosting `ResponseObject`.

### Fixed

- **A truncated Responses stream lost its terminal metadata** (`openai`
  crate). `parse_responses_event` handled `response.completed` but not
  `response.incomplete`, so a stream cut off by the content filter or the
  token budget delivered no finish reason, usage or response id — exactly
  the information the distinct event name exists to report. Both are now
  handled on the same arm.

- **An empty hosted-MCP allowlist enabled every tool on the server**
  (Anthropic). `allowed_tools: Some(vec![])` means "expose none", and was
  treated as "no allowlist" — leaving the API default, which enables all of
  them. Now encoded as `tool_configuration: {"enabled": false}`. The OpenAI,
  Azure and Foundry paths were already correct (upstream #8576).
- **A cut-off turn was reported as a completed one.** `finish_reason` never
  reached the agent types, so an Azure OpenAI content-filter block or a
  token-budget truncation was indistinguishable from a finished answer. The
  OpenAI-compatible surface no longer hardcodes `finish_reason: "stop"`, and
  the Responses surface emits `status: "incomplete"` with
  `incomplete_details.reason` (`content_filter` / `max_output_tokens`) and a
  `response.incomplete` terminal event (upstream #8478).
- **Hamming distance could not distinguish integers above 2^24.** Stored
  vector coordinates were narrowed to `f32` before comparison, so distinct
  large integers — ids, timestamps, hashes — compared equal and a
  non-matching record scored as an exact match. Coordinates are now compared
  at `f64` (upstream #8637).
- **Hamming could not distinguish 64-bit integers.** Widening stored vector
  coordinates to `f64` moved the collision boundary from 2^24 to 2^53; it did
  not remove it, since `f64` cannot represent `2^53 + 1` either. Hamming asks
  whether two coordinates are the *same*, and a store keyed on ids, timestamps
  or hashes lives exactly there — so stored coordinates now keep the integer
  JSON carried and Hamming compares integers as integers, while the
  arithmetic metrics keep using `f64`.
- **`finish_reason` was dropped by `response_to_updates`.** The buffered
  `SupportsAgentRun::run_stream` and `Agent` streaming through agent
  middleware rebuild updates from an `AgentResponse`, and that helper did not
  carry the reason across — so a content-filtered turn still reached hosting
  as `stop`/`completed` on those paths. It now rides the final update, and a
  reason with no messages emits one, matching what `client.rs` does for
  `ChatResponse`.
- **An incomplete response carried a completed output item.**
  `OutputMessage::assistant_text` hardcodes `status: "completed"`, so a
  truncated or filtered response contained a message item claiming it
  finished — on the buffered and DevUI streaming paths alike. The item's
  status now follows its response's.
- **A run with no searchable input lost its memories.** An instruction-only
  run, or one carrying only system turns, returned from
  `FoundryMemoryProvider::before_run` before injecting the profile already
  fetched for its scope. The contextual search is still skipped (there is
  nothing to search with); the injection is not.
- **The memory provider held its state lock across HTTP calls**, so one slow
  scope stalled every other session sharing the provider. It now snapshots
  under the lock, releases it for the request, and re-acquires to record the
  result.
- **An unfamiliar finish reason was reported as a completed turn.** The
  Responses surface mapped only `content_filter` and `length`, defaulting
  everything else to `completed`. `FinishReason` is an open string and
  providers use it — the Anthropic converter deliberately preserves
  `model_context_window_exceeded` — so the abnormal endings that matter most
  were the ones swallowed. Completion is now an allowlist (absent, `stop`,
  `tool_calls`); every other reason is reported incomplete and passes through
  verbatim.
- **The scope cache never came back down after a burst.** Keeping in-flight
  scopes alive (below) lets the cache exceed its bound, but eviction ran
  only when inserting a *new* scope and dropped at most one slot — so a
  burst of concurrent sessions stayed resident permanently. Trimming now
  runs on every touch and loops until the bound is met, making the overrun
  temporary, which is the only thing that made it acceptable.
- **Evicting a scope with a request in flight defeated per-scope locking.**
  The LRU dropped the least-recently-used slot unconditionally, so a slow
  run could have its slot evicted, the next run for that scope would build
  a *second* mutex, and the two would race and fork the cursor — the exact
  failure per-scope locking exists to prevent. Only a slot nobody holds is
  now evictable; when every candidate is busy the cache runs briefly over
  capacity instead.
- **A tool finish reason was advertised without the tool calls.** Reporting
  `tool_calls` / `function_call` tells a client to execute the call in
  `message.tool_calls`, and the OpenAI-compatible host serializes text only
  — so the client was handed an instruction with no id, name or arguments.
  Those two reasons now degrade to `stop` with the real one in
  `x_finish_reason`, until the surface can serialize the calls it is
  promising. (Recorded as a capability gap, not a permanent answer.)
- **Concurrent turns on one memory scope forked the update chain.** The
  provider snapshotted `previous_update_id`, released its lock for the
  request and wrote back, so two turns for the same scope both resumed from
  the same cursor and one branch fell out of the service's incremental
  chain. The provider now holds **one lock per scope** rather than one per
  provider: the snapshot → request → commit sequence is atomic within a
  scope, and unrelated scopes still never wait on each other. The same
  change removes the duplicate profile fetch the previous release note
  described as merely idempotent.
- **A `function_call` turn was reported as a failure.** OpenAI's finish
  vocabulary has five values, not four: `function_call` is the deprecated
  spelling of `tool_calls` and, like it, marks a turn that *succeeded*. It
  was missing from core's `FinishReason` constants, so the Responses surface
  called it incomplete and the Chat Completions surface rewrote it to
  `length` — telling clients a working tool call had been truncated.
  `FinishReason::FUNCTION_CALL` now exists and both surfaces treat it as the
  completion it is.
- **The DevUI stream closed an incomplete response with
  `response.completed`.** The terminal event was chosen from
  `incomplete_details`, which the fix below makes absent for an unfamiliar
  provider reason — so precisely the cut-off turns this path exists to flag
  were announced as successes. It now follows the response's `status`.
- **A provider-only reason could reach the Chat Completions `finish_reason`
  enum.** That wire field is a closed set, and a strict generated client can
  reject a response over a value outside it. An unfamiliar reason is now
  reported as `length` — the only legal value that says "not a complete
  answer" without asserting a cause — and carried verbatim in a
  non-standard `x_finish_reason` beside it, on both the buffered and
  streaming paths.
- **A failed concurrent profile fetch could erase a scope's memories
  permanently.** Releasing the state lock for the request (below) lets two
  first runs for one scope overlap; the failing one committed an empty
  profile over the successful one's and set `initialized`, so nothing ever
  re-fetched it. A failure now leaves the cached profile untouched.
- **The scope cache grew without bound.** With the session fallback a scope
  is a session, so a long-lived agent accumulated one `ScopeState` per
  session forever. It is now an LRU bounded by `DEFAULT_MAX_CACHED_SCOPES`
  (512), overridable with `with_max_cached_scopes`; eviction costs a scope
  only a re-fetched profile and a restarted cursor.
- **`incomplete_details.reason` could carry a value the schema forbids.**
  The Responses schema names exactly two reasons, and a strict generated
  client can reject a whole response over a third — so a provider string
  like `model_context_window_exceeded` no longer goes in that field. Such a
  run is still reported `incomplete`, and the raw reason moves to the new
  `x_finish_reason` extension, which a strict client ignores rather than
  has to parse.
- **The project embeddings route dropped the OpenAI `user` option.** The
  per-route allowlist added below pruned Inference-only fields but also
  pruned `user`, which the derived `/openai/v1/embeddings` endpoint *does*
  accept and which `agent-framework-openai` and `agent-framework-azure` both
  forward on the same surface. The two routes now differ in both directions:
  `input_type` is Models-only, `user` is project-only.
- **The project embeddings route forwarded Azure AI Inference-only fields.**
  Selecting the project route changed the URL but not the payload, so
  `input_type` — an Inference field with no OpenAI equivalent — was still
  sent to the derived `/openai/v1/embeddings` endpoint, which rejects it.
  Request properties and the `extra-parameters` pass-through header are now
  chosen per route.
- **A failed contextual search discarded the cached profile.** In
  `FoundryMemoryProvider::before_run`, a transient failure of the second
  search returned early and dropped the static memories the first had
  already fetched. It is now treated as an empty contextual result, so the
  known-good profile is still injected.
- **A blank `FOUNDRY_PROJECT_ENDPOINT` shadowed the `FOUNDRY_ENDPOINT`
  alias.** The fallback was selected before the blank check, so an env
  template declaring the optional variable as `""` suppressed a perfectly
  good alias. Each candidate is now filtered before the fallback.
- **A whitespace-only instruction became a contentless system message.**
  `prepare_messages` skipped `""` but not `" "` or `"\n"`. A real instruction
  still prepends verbatim (upstream #8524).

### Documentation

- `StandardMagenticManager` now documents that its cached task ledger is
  shared by every run holding the same manager, and that the two readers of
  that cache are the plan-review and stall-intervention requests — so
  concurrent runs can show a human reviewer the other run's plan. Prefer
  `standard_manager`, which takes the manager by value (upstream #8581).


## [0.8.0] — 2026-09-21

An Azure Cosmos DB vector store and a provider that hands any vector
collection to an agent as tools, plus four faults in the Purview middleware
that each let content reach the model without being evaluated — or, in one
case, evaluated as its own escaping.

**Breaking, in four places.** `Error::ServiceContentFilter` gains a `detail`
field, so a struct-variant destructuring without `..` stops compiling.
`FunctionInvocationConfig` gains `allow_concurrent_invocation`, so a struct
literal without `..Default::default()` needs it. `DlpAction` gains
`RestrictAccess`, so an exhaustive match over it needs an arm. And
`ProcessConversationMetadata::content` is now a `PurviewContent` rather than
a `PurviewTextContent`, because an attachment is submitted as binary.

**Two behaviour changes in the Purview middleware.** It now **fails closed**
when no Entra user id can be resolved: previously it reported the content as
allowed, which is an unevaluated message reported as a cleared one.
`ignore_exceptions` still trades that for availability, deliberately and for
every error rather than only this one. And it submits **every content item**
rather than each message's text, so a deployment will see more
`processContent` calls and, correctly, more verdicts.

**One behaviour change in core serialization**, unrelated to Purview:
`FunctionResultContent::exception` now serializes to a fixed marker, so a
conversation persisted from here on keeps the *fact* of a tool failure but not
its diagnostic text. Conversations already stored are unaffected, the model
still sees the real text, and Purview still evaluates the real text — the
evaluation path reads the live value rather than the serialized one.

### Added

- **An Azure Cosmos DB for NoSQL vector store**: `CosmosVectorStore` and
  `CosmosVectorCollection` in `agent-framework-cosmos`. A container becomes a
  collection — the `vectorEmbeddingPolicy` and `indexingPolicy` at creation,
  point reads and upserts in the item's own partition, and a `VectorDistance`
  query with every filter literal bound as a parameter.

  Four details are decisions rather than defaults. The vector surface needs
  its own api-version (`DEFAULT_VECTOR_API_VERSION`, `2020-07-15`): the
  `2018-12-31` the history and checkpoint stores speak predates vector search,
  so a container created under it comes back without the policy it asked for.
  A distance function Cosmos cannot compute is **refused** rather than
  substituted, since Cosmos computes cosine as a *similarity* and a silent
  substitution ranks by a metric the caller's `higher_is_closer` then reads
  backwards. `skip` rides in the limit, because Cosmos will not take `OFFSET`
  beside an `ORDER BY VectorDistance`. And an ordered filter on an untyped
  field carries a SQL type guard, because Cosmos SQL orders *across* types.

- **`VectorCollectionContextProvider`** (`agent_framework_core::vectors`):
  generates search / get / upsert / delete tools over any `VectorCollection`,
  so the in-memory, Azure AI Search and Cosmos DB stores become things an
  agent can use rather than things a caller drives.

  Writes require approval by default and reads do not. The `scope_filter` is
  applied everywhere — conjoined into search, checked on the records a read,
  write or delete touches, and checked against the *existing* record on an
  upsert so a scoped agent cannot overwrite another group's record by naming
  its key. An out-of-scope record reads as absent rather than as a refusal,
  since "you may not read that" confirms it exists. It remains grouping rather
  than an authorization boundary, and the module docs say so in those words.

  The model never authors an embedding: `embed_from_field` names the field the
  vector is derived from, and the vector field is not in the tool schema at
  all. Without it there is no upsert tool rather than one that cannot work —
  the same reason a multi-vector collection gets no upsert tool, since one
  embedding source cannot maintain several vectors and a whole-document write
  would drop the rest.

- **Azure content-filter detail** on `Error::ServiceContentFilter`:
  `ContentFilterDetail` carries the policy code, the `param`, and the
  per-category verdicts from Azure's nested `innererror`, reaching Azure
  OpenAI chat, Responses and embeddings. Every value is an open string —
  upstream parses the code and severity into enums, which raise on a value
  Azure has not shipped yet.

- **`reasoning_details` on Chat Completions**, in both directions. A
  reasoning-capable provider on that surface (DeepSeek in thinking mode,
  OpenRouter, vLLM) requires the payload back on the next request of the same
  turn, so a conversation that used reasoning could not be continued. It rides
  in `TextReasoningContent::protected_data`, and fragments split across a
  stream concatenate in order.

- **`FunctionInvocationConfig::allow_concurrent_invocation`**: runs one model
  response's tool calls in model order when `false`. The point is the side
  effects rather than the result order — a tool holding a non-reentrant handle
  behaves differently when two invocations overlap.

- **`FUNCTION_INVOCATION_ERROR_MARKER`** and
  `FunctionResultContent::is_error`.

### Fixed

- **Purview evaluated a fraction of each message.** It submitted
  `Message::text()` and nothing else, so a function result — where exfiltrated
  data actually appears — an attachment, a reasoning block and a tool call all
  reached the model unevaluated. Every content item is mapped now, one request
  per entry: an attachment's bytes go as Graph `binaryContent` (handing a
  classifier the base64 *string* reads as gibberish and passes every policy),
  and everything else is serialized whole. Only `usage` is skipped. This port
  had a hole on top of upstream's: `text()` returns `""` whenever a refusal is
  present, so a partly-declined turn submitted nothing at all.

- **Purview could not parse a verdict carrying `restrictAccess`**, losing it
  entirely. The action is named now; it is deliberately not a block on its
  own, since its `restrictionAction` selects the mode. The enums stay strict:
  an unrecognized value fails the parse, which the default
  `ignore_exceptions = false` turns into a stopped request, rather than
  landing in a catch-all that `should_block` would answer `false` for.

- **Purview reported an unevaluated message as allowed** when no user id
  resolved. See the behaviour note above.

- **A failing tool's diagnostics were persisted verbatim** into every history
  store and workflow checkpoint. `FunctionResultContent::exception` is
  host-internal text this framework does not control — a connection string or
  a SQL error quoting the row it failed on are ordinary things to find in one.
  It serializes as a fixed marker now; failure state survives, deserialization
  is untouched, and the model is unaffected because every provider converter
  reads the field directly.

- **`gen_ai.client.operation.duration` omitted failed calls**, which leaves no
  error rate at all and skews the latency distribution. All three failure
  paths record with `error.type` now, including the stream that fails midway —
  which ends before its completion arm and so was missing entirely.

- **A streamed `function_call` fragment could land on the wrong call.** An
  untagged continuation delta went to the *first* in-flight call rather than
  the one still streaming; an untagged chunk could be absorbed by a call
  already carrying an occurrence id; two calls sharing a provider `call_id`
  but carrying *different* occurrence ids merged, appending one call's
  arguments onto the other's; and a failed merge dropped the fragment silently.

- **A handoff dropped the user's attachments.** An image or file is the
  request as much as the words beside it. Multimodal content is kept on user
  messages and still dropped elsewhere, since providers reject input-only
  parts replayed on an assistant turn. The same pass stops an approval
  *response* riding on a user message through to the next agent.

- **Concurrent checkpoint saves collided on one temp file.** Two saves of the
  same id wrote into the same file, so the survivor could be a blend of both,
  and the loser's rename failed with `NotFound`.

- **A Cosmos id containing `%` stored but could not be read back.** The id is
  percent-encoded as a path segment now while the signature stays over the raw
  link, as Cosmos's auth scheme requires. Previously a point read 404'd and a
  delete reported success while leaving the record.

- **A percent-escaped data URI was evaluated as its own escaping.** RFC 2397's
  data segment is URL characters, so the payload may arrive escaped; decoding
  that as base64 fails and the item went to Purview as text. Payloads are
  percent-decoded and unwrapped before the base64 decode.

### Changed

- The Cosmos chat store documents that a thread id **selects** history rather
  than protecting it: distinct ids prevent accidental overlap without
  restricting a client whose credentials already authorize the container.

## [0.7.0] — 2026-09-15

Portable vector filters and an Azure AI Search vector store, bounds on the
tool loop, and four fixes to isolation and replay.

**Breaking, in five places.** `VectorSearchOptions::filter` is now a
`FilterExpression` rather than a provider-dialect `String`; the string form
moved to `provider_filter` / `with_provider_filter`.
`McpStreamableHttpTransport::new` returns a `Result` (it validates the URL).
`SecretString` no longer implements `Serialize`, so a `#[derive(Serialize)]`
on a struct holding one stops compiling — deliberately; see below. Every
workflow graph signature changes, so a checkpoint written by 0.6.x is refused
on resume with a message naming the scheme change (and
`run_from_checkpoint_unchecked` still resumes it). And `VectorSearchResult`
gains a `score_kind` field, so code constructing one with struct-literal
syntax needs updating (it does not derive `Default`). Reading one is
unaffected.

Two behaviour changes to look for. Redis keys are now derived through an
injective encoding: a literal-safe prefix and session id — the default prefix
and any UUID — produce exactly the keys they always did, so existing data is
unaffected, but an identifier containing `:`, uppercase, or a glob character
now addresses a different key than before. And an Anthropic reasoning content
with no signature is no longer sent as a thinking block; with extended
thinking enabled the API rejects one, so this turns a 400 into a request that
works.

### Added

- **Portable vector filters** (`agent_framework_core::vectors::filters`):
  `Filter` leaves and `FilterGroup` nodes across 18 operators, validated at
  construction (operand shape per operator, depth ≤ 8, ≤ 64 nodes), with a
  namespaced escape hatch (`azure_ai_search.match`) for anything a connector
  defines itself. `InMemoryVectorStore` evaluates them instead of refusing
  every filter.

  The semantics are upstream's: a missing field is a non-match for every
  operator except `exists` (so `ne` and `not(eq)` differ), a boolean never
  equals a number, and numbers compare by **value** — a record that
  round-tripped through JSON as `1` matches `eq: 1.0` — with integers compared
  as integers, since routing them through `f64` to get that rule rounds past
  2^53 and makes two adjacent 64-bit ids equal. An expression that never went
  through a constructor (`Deserialize` bypasses them) is reported as a
  configuration error by `matches` rather than panicking or silently becoming
  match-all.

- **An Azure AI Search vector store**: `AzureAISearchStore` and
  `AzureAISearchCollection` in `agent-framework-azure-ai-search`. Index
  create/exists/delete from a `VectorStoreCollectionDefinition` (EDM types,
  vector profiles, hnsw vs exhaustiveKnn, metric mapping), document
  upsert/get/delete, vector search with the portable filter translated to
  OData, keyword-hybrid search, and index aliases. `build_index()` and
  `prepare_filter()` are public so a caller driving the REST API themselves
  can reuse either.

  Details that decide whether results are right: filtering is `preFilter` and
  `k` covers `skip + top`, so a selective filter does not return fewer than
  `top`; `upsert` sends `upload`, honouring the trait's insert-or-*replace*
  contract rather than merging; an indexing batch is bounded by serialized
  payload size as well as the 1,000-action limit, because a 1536-dimension
  embedding is ~18 KB and a full batch would be ~18 MB; a **207** partial
  success is raised rather than read as success; and an operator Azure cannot
  express is refused with the alternative named, never silently dropped.

- **Bounds on the function-invocation loop**:
  `FunctionInvocationConfig::max_function_calls` caps total tool executions
  per request and `max_duration_seconds` caps elapsed wall time.
  `AgentBuilder::function_invocation_config` reaches the whole config from an
  agent for the first time, `max_iterations` included.

  Both bounds are cumulative across approval round trips — parked in the
  session, resumed only by a request that actually carries the approval
  responses, and discarded when a run ends or fails — and both are graceful:
  tools are disabled and the model answers with what it has. The call budget
  charges only calls that reached a tool, so a hallucinated tool name costs
  nothing. The clock is re-checked after each model call, so a slow provider
  response cannot buy a free tool batch. And a spent budget stops *executing*
  rather than only setting `tool_choice`, which a provider may ignore; when it
  expires mid-response, work the provider already did (a hosted tool call and
  its result) is kept and handed to the final model call.

- `agent_framework_core::storage_keys::storage_key_segment`, the derivation
  behind the Redis key fix below.

- `agent_framework_azure::DEFAULT_CHAT_API_VERSION` /
  `DEFAULT_EMBEDDING_API_VERSION`, both public.

### Fixed

- **MCP HTTP headers leaked across a redirect.** Headers attached to an
  `McpStreamableHttpTool` went onto a client that follows redirects and strips
  only `Authorization`, `Cookie` and `Proxy-Authorization` on a cross-*host*
  hop — so an `X-Api-Key` reached whatever host the server redirected to, and
  even `Authorization` survived a redirect to a different port or scheme. The
  `Mcp-Session-Id` was both sent to and adopted from a redirect target. The
  transport now follows redirects itself, scoped to scheme/host/port, and
  preserves the POST method and body across every hop. Session teardown goes
  the same way.

- **Anthropic extended-thinking signatures were dropped**, on both the
  buffered and streaming paths, and `redacted_thinking` blocks vanished
  entirely. With extended thinking enabled the Messages API requires a
  replayed thinking block to carry its signature, so a conversation that used
  it could not be continued — the tool-result turn was rejected. Signatures
  now ride in `protected_data`, a decoded block is replayed verbatim (the
  signature covers the exact text), and two thinking blocks in one streamed
  message stay separate instead of coalescing into one that neither signature
  covers.

- **Redis keys were ambiguous.** `{key_prefix}:{session_id}` addressed one
  list for `("chat", "a:b")` and `("chat:a", "b")`, so two conversations that
  should be isolated shared a history — one tenant reading another's, where
  the prefix is the tenant boundary. The context provider had a second
  variant: a prefix containing `:entry:` put one provider's entries inside
  another's `SCAN MATCH`, which `clear()` would then delete.

- **The workflow graph signature was not injective.** Executor ids were joined
  with `,` and `->`, so a fan-out to `["x", "y"]` and one to `"x,y"` signed
  identically and a checkpoint from either was accepted for the other. Now
  JSON-encoded; the scheme tag is `v2`, and a `v1` checkpoint is reported as a
  scheme mismatch rather than as a graph change.

- **`SecretString` serialized its secret in cleartext.** The derive is
  removed rather than masked, because a masked `Serialize` round-trips the
  mask back as the value.

- **The Azure chat api-version was pinned to GA `2024-10-21`**, which rejects
  request fields this client sends (`store` first among them) with
  "Unrecognized request argument supplied". Now `2024-12-01-preview` for chat,
  matching upstream, with embeddings left on GA as upstream also has it.

### Changed

- `VectorSearchResult` gains `score_kind` and `higher_is_closer(..)`. A
  collection's distance function is a *request*, and on several services it
  does not describe what comes back: Azure AI Search returns a
  higher-is-better relevance score whatever metric the profile declares, and a
  reciprocal-rank-fusion score for a hybrid query. A caller reading direction
  from `DistanceFunction::higher_is_closer` — false for the distance metrics,
  which are the usual declaration — would have ranked every result backwards.
  `InMemoryVectorStore` leaves it `None`, meaning the declared function does
  describe the score. Mirrors upstream's `score_kind` result metadata.

- `VectorStoreCollectionDefinition` gains `try_get_field` and
  `storage_name_for`.

- `AzureAISearchStore::collection` returns the concrete collection type, for
  the surfaces (`search_hybrid`, `build_index`) the object-safe
  `VectorCollection` trait cannot declare.

## [0.6.0] — 2026-09-09

Provider refusals handled properly end to end, Entra ID authentication for
Cosmos DB, and the first vector-store abstractions.

**This one can break a compiling caller**, unlike 0.4 and 0.5. Two public
structs gained a field: `TextContent::refusal` and `FunctionCallContent::id`.
Neither is `#[non_exhaustive]`, so any code constructing them with
struct-literal syntax needs updating. `TextContent` derives `Default`, so a
`..Default::default()` literal keeps working; `FunctionCallContent` does not,
so its literals must add `id` (or move to `FunctionCallContent::new`, which
takes the same three arguments as before). Nothing was removed, and no
function signature changed.

The behavior change to check for is refusal handling. A provider decline used
to arrive as ordinary text, so `Message::text()`, `ChatResponse::text()` and
`AgentResponse::text()` returned the refusal prose as though it were the
answer — and `parse_json` tried to parse it. They now return `""` for a
refused turn, and the decline is read deliberately through `refusal_text()`.
Code that displayed whatever `text()` returned will show an empty string
where it used to show "I can't help with that"; branch on `has_refusal()` to
render a decline. This is the fix, not a regression, but it is visible.

### Added

- **Microsoft Entra ID authentication for `agent-framework-cosmos`.**
  `CosmosChatMessageStore::with_token_credential` and
  `CosmosCheckpointStorage::with_token_credential` take any
  `agent_framework_azure::TokenCredential`, so a Cosmos store authenticates
  with a managed identity, a workload identity, or the Azure CLI instead of a
  master key — and a Cosmos account with `disableLocalAuth` set, which has no
  key to give, becomes usable at all.
  `CosmosChatMessageStore::from_state_with_token_credential` restores a
  serialized store, since a credential cannot round-trip through a state blob
  the way a key does. The token scope defaults to
  `https://cosmos.azure.com/.default` — Cosmos DB's data-plane audience is
  service-wide, not per account — and `AZURE_COSMOS_AAD_SCOPE_OVERRIDE`
  overrides it, matching the official `azure-cosmos` SDK.

  Note that Cosmos DB's Entra RBAC grants data-plane actions only:
  `ensure_created` cannot succeed with a token whatever role the principal
  holds, so the database and container must be provisioned out of band. The
  call now says so rather than surfacing a bare `403`.

- **`InMemoryVectorStore` validates vectors before scoring**: non-numeric
  elements, wrong-width stored vectors, and non-finite values are skipped
  rather than reshaped or ranked, and a non-finite query vector is rejected
  outright (it would score every record identically and produce a score that
  cannot serialize as JSON).

- **Vector-store abstractions** (`agent_framework_core::vectors`):
  `VectorStoreField`, `VectorStoreCollectionDefinition`, `IndexKind`,
  `DistanceFunction`, `VectorSearchOptions`, `VectorSearchResult`, the
  object-safe `VectorCollection` and `VectorStore` traits, and an
  `InMemoryVectorStore`. Records are `serde_json::Value` objects keyed by
  field name; a definition maps logical names to storage names. No provider
  crate implements the traits yet.

- **`preserve_first_user()`** on the `Truncation`, `SlidingWindow` and
  `TokenBudget` compaction strategies: retain the earliest user message
  whatever the budget, so a long conversation cannot lose the request it is
  about. Off by default; a preserved message is kept regardless of budget, so
  the result may exceed the configured limit by one.

- **`FunctionCallContent::id`**, a framework-generated occurrence id
  (`af-call-<uuid>`) minted when a call is deferred for approval, plus
  `ensure_occurrence_id()` and `same_invocation()`. Approvals now bind to a
  specific occurrence rather than to the provider `call_id`, which providers
  reuse.

- **`TextContent::refusal`** and `TextContent::refusal(..)`, plus
  `Message::has_refusal()` / `Message::refusal_text()`.

### Fixed

- **Provider refusals no longer read as answers.** An OpenAI refusal was
  parsed into plain text, so `Message::text()` returned it as though the model
  had answered and `parse_json` would try to parse it as the requested output.
  Refusal text is now marked, `Message::text()` returns `""` when one is
  present, and streamed coalescing keeps refusal and ordinary text in separate
  content items. The Chat Completions parser also used to emit a refusal only
  when there was no content, hiding a model that answered part of a request
  and declined part; both are now kept. Streamed refusals are parsed too
  (`delta.refusal` on Chat Completions, `response.refusal.delta` on
  Responses) — they were dropped outright, so a streamed decline reached the
  caller as an empty response. And `structured_output_text` withholds a
  refused turn, so `parse_json` / `value` cannot be populated from a refusal
  (nor fall back to an older turn's JSON and serve it as this run's answer).
  The guard is applied at the response level too — `ChatResponse::text`,
  `AgentResponse::text` and both streaming updates — because a tool loop
  accumulates earlier assistant turns, so blanking only the refusing message
  still handed back an intermediate aside as the final answer. On the two
  response types that guard is anchored to the run's **final** assistant
  turn rather than to any accumulated one: a provider may decline part of a
  request while still calling a tool for the rest (OpenAI puts `refusal` and
  `tool_calls` on the same message, and the parser keeps both), and an
  `any`-style check let that carried turn blank the successful answer that
  followed it. A run that ends on a refusal is still withheld whole.
  `has_refusal()` / `refusal_text()` are available on each. The **outbound**
  path preserves the marker too: replaying a refusal as history now uses the
  assistant message's own `refusal` field (Chat Completions) and a
  `{"type": "refusal"}` output content part (Responses), instead of folding
  it into ordinary content and telling the provider the assistant answered.

- **An empty vector-store storage-name override is rejected.** A
  `VectorStoreField` whose `name` was empty was refused, but one carrying
  `storage_name: Some("")` was accepted and produced the same broken result:
  the field is written under an empty JSON key. `InMemoryVectorStore`
  tolerates that, so it passed locally and would have failed at a real
  provider's collection creation or first write. An absent override is
  unaffected — it still falls back to the field name.

- **Two approvals pending under one provider `call_id` are no longer
  conflated.** They were matched structurally, so a second pending approval
  for the same call looked like a replay and was dropped, and one result
  answered both. Occurrence ids now decide identity when present, with the
  structural rule as a fallback so approvals stored before this keep
  resolving.

- **Purview asks for inline evaluation.** Every `processContent` request now
  carries `Prefer: evaluateInline`. Without it the service may evaluate
  content offline and return no actionable verdict, which would leave the
  blocking middleware with nothing to decide on — enforcement would quietly
  become a no-op rather than fail.

- **A token count reported as zero is no longer dropped from the OpenAI usage
  breakdown.** `parse_usage` skipped any `completion_tokens_details.*` /
  `prompt_tokens_details.*` value equal to `0`, mirroring a truthiness bug in
  upstream's Python (since fixed there), so a provider reporting zero audio,
  accepted-prediction, rejected-prediction or cached tokens produced no entry
  at all — indistinguishable from a provider that does not report that count.
  Both now appear as `Some(0)` and `None` respectively in
  `UsageDetails::additional_counts`, which is what the GenAI metrics layer
  reads. Non-integer values are still ignored. The Responses path and the
  typed fields (`reasoning_output_token_count`, `cache_read_input_token_count`)
  were already correct, so this also removes a silent disagreement between the
  two paths about the same response.

### Changed

- `CosmosChatMessageStore::serialize` omits `key` and emits
  `"auth": "token_credential"` for a credential-authenticated store (a
  master-key store's state is unchanged). `from_state` on such a blob now
  returns an error naming `from_state_with_token_credential` rather than
  reporting a missing field.

## [0.5.0] — 2026-08-31

Embeddings for two more providers, and a Gemini finish-reason fix.

Nothing was removed and no existing public item changed shape, so a compiling
caller should keep compiling. The bump is a minor one for the new surface —
two embedding clients, both re-exported from the prelude — rather than for an
incompatibility. Pre-1.0 that is still a compatibility break to cargo, so
dependants pinned to `0.4` need to move to `0.5` to pick this up.

The one behavior change is in the Gemini client: finish reasons that used to
arrive as raw lowercased strings now arrive as the canonical values, and
`FINISH_REASON_UNSPECIFIED` now reads as absent. Code matching on the old
strings needs updating — see **Fixed** below.

### Added

- **`BedrockEmbeddingClient`** (`agent-framework-bedrock`) — Amazon Titan Text
  Embeddings through Bedrock Runtime `InvokeModel`
  (`POST /model/{modelId}/invoke`), SigV4-signed by the same `sigv4` module
  the Converse chat client uses.
  - Titan's body carries a single `inputText` rather than a batch, so a batch
    of *n* values costs *n* signed requests, issued concurrently and
    reassembled in input order, with `inputTextTokenCount` summed across the
    batch into `UsageDetails::input_token_count`.
  - `with_endpoint` overrides the base URL — a PrivateLink interface
    endpoint, a region the host derivation does not cover, or a local test
    server. Both the `Host` *and* any path prefix on that URL are folded into
    the canonical request before signing, so the signature always covers the
    path actually requested; an endpoint or proxy mounted under a sub-path
    works rather than failing AWS authentication.
  - Credentials are the static or `AWS_*`-environment ones the chat client
    already takes. Upstream reaches Bedrock through boto3 and inherits its
    whole credential chain; this crate signs directly and does not.
  - Titan's `normalize` knob is forwarded through
    `EmbeddingGenerationOptions::additional_properties`.
- **`FoundryEmbeddingClient`** (`agent-framework-foundry`) — the Azure AI
  Foundry **Models** inference endpoint (`POST {models_endpoint}/embeddings`),
  with `api-key` or Entra `TokenCredential` auth. Distinct from
  `AzureOpenAIEmbeddingClient`, which is deployment-scoped; the body and
  response are OpenAI-shaped, so parsing and error classification are shared
  with `agent-framework-openai` rather than duplicated.
  - **Text inputs only.** Upstream also accepts image `Content` and splits a
    batch across `ImageEmbeddingsClient` (`/images/embeddings`). The core
    `EmbeddingClient` trait takes `Vec<String>`, so an image input cannot be
    expressed at that boundary at all — closing this needs a trait widening
    across every provider, not a change to this client.
    `FOUNDRY_IMAGE_EMBEDDING_MODEL` is correspondingly not read.
  - `DEFAULT_API_VERSION` reproduces the default in `azure-ai-inference`
    1.0.0b9 (the version upstream's `foundry` package pins), since upstream
    sends no explicit `api-version` of its own. It is the one value here taken
    from an SDK default rather than a verified service contract, so
    `with_api_version` overrides it.
  - Entra tokens are requested for `DEFAULT_SCOPE`
    (`https://cognitiveservices.azure.com/.default`), the Azure AI Services
    **data plane** audience — *not* `FOUNDRY_SCOPE`
    (`https://ai.azure.com/.default`), which is the Foundry **project**
    audience `FoundryChatClient` needs for the Responses API. The Models
    inference endpoint rejects a token minted for the project scope.
    `with_scope` overrides it.
  - `extra_parameters` is **expanded** into the request body under each
    entry's own name, matching what upstream's `model_extras` mapping does,
    rather than sent as a literal `extra_parameters` field the model would
    ignore. Expanding also sends `extra-parameters: pass-through`, without
    which Azure AI Inference rejects body fields outside its schema.
    `encoding_format` and `input_type` continue to forward verbatim.
  - `additional_properties` is an **allowlist**, matching the kwargs upstream
    builds and the sibling OpenAI/Azure clients: only `encoding_format`,
    `input_type` and `extra_parameters` reach the wire. Copying arbitrary
    entries through would put fields outside the inference schema on the
    request, which the service rejects — so an options struct reused across
    providers (carrying, say, OpenAI's `user`) would 4xx on an option that is
    merely irrelevant here. `extra_parameters` remains the escape hatch for
    anything genuinely model-specific.
  - `from_env` requires only the endpoint and model, matching upstream's
    `required_fields`. With no `FOUNDRY_MODELS_API_KEY` it falls back to
    `DefaultAzureCredential` at the scope above, so a managed-identity or
    `az login` environment works keyless — the same fallback
    `FoundryChatClient::from_env` already had.
- Both clients are re-exported from `agent_framework::prelude` under their
  existing `bedrock` and `foundry` features. No new default dependencies;
  `agent-framework-foundry` gains `reqwest` and `agent-framework-openai`.

### Fixed

- **Base64 embedding responses were rejected.** Every OpenAI-shaped embedding
  client here (`OpenAIEmbeddingClient`, `AzureOpenAIEmbeddingClient`,
  `MistralEmbeddingClient`, `OllamaEmbeddingClient`, and the new
  `FoundryEmbeddingClient`) forwards `encoding_format` verbatim, but the
  shared response parser accepted only a numeric `embedding` array. Asking for
  `encoding_format: "base64"` — a documented, forwarded option — therefore
  failed a perfectly successful response. The parser now decodes the base64
  form (packed little-endian `f32`s) as well, and reports undecodable or
  misaligned payloads rather than silently truncating. Pre-existing since
  embeddings were introduced, not new in 0.5.0.
- **Gemini finish reasons.** Five of the names in the Gemini API's
  `FinishReason` enum were not in `agent-framework-gemini`'s mapping table and
  fell through its passthrough arm as lowercased raw strings, so a caller
  matching on the canonical value saw `"language"`, `"image_recitation"`,
  `"image_prohibited_content"`, `"malformed_function_call"` or
  `"unexpected_tool_call"` where `content_filter` or `tool_calls` was meant.
  All five now map, matching upstream's table (#7837).
- **Gemini `FINISH_REASON_UNSPECIFIED`.** Proto3's "field never set" was being
  reported as a finish reason named `finish_reason_unspecified`. It now reads
  as *absent*, like a response carrying no `finishReason` at all — so a turn
  that ends in a function call is upgraded to `tool_calls` as it already was
  when the field was omitted entirely.

## [0.4.0] — 2026-08-26

Two optional integrations: Entra ID credentials from the official Azure SDK
for Rust (GA'd in May 2026), and a ready-made OTLP export pipeline for the
spans and GenAI metrics this framework already emits.

Nothing in the existing API was removed or changed, and the default build
gains no dependencies, so nothing here should break a compiling caller. The
minor bump reflects the size of the new surface and the build requirement
below rather than an incompatibility — but pre-1.0 it is still a
compatibility break to cargo, so dependants pinned to `0.3` need to move to
`0.4` to pick this up.

> **Build requirement for the new features.** Enabling `entra-sdk` or
> `otel-export` requires a **C toolchain (`cmake`)**. Both pull `aws-lc-rs`,
> which is the TLS provider for the reqwest 0.13 that `azure_core` and
> `opentelemetry-otlp` depend on — a separate major version from the reqwest
> 0.12 used elsewhere here, whose features cargo cannot unify with it. Builds
> that do not enable either feature are unaffected.

### Added

- **Entra ID credentials from the official Azure SDK for Rust**, behind
  `agent-framework-azure`'s new optional `entra-sdk` feature.
  `SdkTokenCredential` adapts any [`azure_identity`] credential onto this
  crate's `TokenCredential`, so SDK and hand-rolled credentials are
  interchangeable wherever a credential is accepted. This is additive: the
  hand-rolled chain remains the default and is unchanged when the feature is
  off. It exists to reach credential types this crate does not implement —
  `ClientCertificateCredential`, `ClientAssertionCredential`,
  `AzurePipelinesCredential`, `AzureDeveloperCliCredential` — and to let
  Microsoft own IMDS quirks, sovereign-cloud endpoints and token lifetimes
  under a semver guarantee.
  - `azure_core` depends on **reqwest 0.13**, a different major from the 0.12
    the workspace uses. Cargo does not unify features across
    semver-incompatible versions, so the workspace's `rustls-tls` does not
    apply to the SDK's client and `azure_core` carries its own
    `reqwest_rustls`. Without it reqwest 0.13 resolves with no TLS backend:
    everything compiles and every unit test passes, but every Entra token
    request fails at run time. Enabling it means `aws-lc-rs` (a C library
    needing `cmake`) is the TLS provider on that side, so builds with this
    feature need a C toolchain. Verified both ways against the live Entra
    endpoint — see the ignored `tls_probe` test in `agent-framework-azure`.

- **A ready-made OTLP export pipeline**, behind `agent-framework-core`'s new
  optional `otel-export` feature (re-exported by the umbrella crate).
  `observability::export::OtelExport` builds an OTLP exporter, tracer
  provider, meter provider and the `tracing`↔OpenTelemetry bridge in one call,
  wired to the GenAI conventions this crate already emits. Enabling it implies
  `otel-metrics`, since the pipeline installs the `MeterProvider` those
  histograms record through. This is the only feature that pulls an OTel
  *SDK* — the default build stays API-only, so consumers who don't ask for
  export never inherit the SDK or its version churn.
  - `OtelPipeline::install` claims the global subscriber for applications that
    have none; `OtelPipeline::tracing_layer` returns the bridge layer for
    applications composing their own. `install` is a no-op when traces are
    disabled, so a metrics-only pipeline leaves the global subscriber free for
    the application's own logging rather than taking it irreversibly.
  - `OtelPipeline::shutdown` returns its outcome instead of logging it, and
    attempts both providers before reporting. Logging would go nowhere in the
    setup this module recommends: `install` builds a subscriber of an
    `EnvFilter` and the OpenTelemetry layer alone, so there is no formatting
    layer for the event to reach, and the one layer present feeds the tracer
    provider being shut down. A failed flush means telemetry was dropped —
    which happens whenever the collector is unreachable — and the caller can
    now see it.
  - Transport is OTLP-over-HTTP/protobuf on reqwest's *blocking* client, not
    gRPC and not the async client. gRPC would add the tonic/hyper stack; the
    async client panics with "there is no reactor running" when the batch span
    processor flushes, because that processor and the periodic metric reader
    each export from a background thread with no tokio runtime on it.
  - `opentelemetry-otlp` brings its own reqwest 0.13 and so needs its own TLS
    feature (`reqwest-rustls`) for the same reason as `azure_core` above;
    without it the exporter reaches plain-HTTP collectors only and fails
    against any HTTPS endpoint.
  - `examples/observability/otel_export.rs` demonstrates both routes and is
    compiled by CI, so the wiring cannot drift — replacing the previous
    ` ```ignore ` snippet in the module docs, which was never compiled.

### Fixed

- **`observability`: corrected stale documentation on the third GenAI
  histogram.** The module docs and
  `metrics::record_function_invocation_duration` both stated that
  `agent_framework.function.invocation.duration` was defined but "not yet
  called anywhere in this crate", and listed switching to `tool_span_ex` and
  adding `record_tool_arguments`/`record_tool_result` as outstanding
  follow-ups. All of that had in fact landed —
  `FunctionInvokingChatClient` times each tool invocation and records the
  histogram, and uses all three span helpers. A reader following the old docs
  would have concluded tool-call timing was unavailable, or wired up a second
  recording of it. Documentation only; no behavior change.

## [0.3.0] — 2026-08-24

Upstream-alignment passes against `microsoft/agent-framework`, moving the
baseline from `4b1afd90` (2026-08-07) to `a63d462` (2026-08-24). Six
upstream changes are ported and 100+ intervening commits are triaged in
[`ALIGNMENT_PROGRESS.md`](./ALIGNMENT_PROGRESS.md), which records why each
one that does not apply does not.

This release **breaks API** (pre-1.0, so a minor bump): the observability
recording functions take an `&ObservabilityConfig` in place of a
`capture_content: bool`, `chat_span` takes the semantic-convention flag, and
`gen_ai.system` is no longer emitted alongside `gen_ai.provider.name`. See
**Changed** below.

### Changed

- **The GenAI semantic-convention version is now selectable, and the provider
  tag follows it** (upstream #7673, [BREAKING] there). `gen_ai.system` was
  renamed to `gen_ai.provider.name` above the OTel v1.36.0 baseline, and this
  port emitted *both* names on every chat span, so a consumer pinned to the
  baseline saw an attribute its version does not define. `ObservabilityConfig`
  now reads `OTEL_SEMCONV_STABILITY_OPT_IN` and exposes
  `use_latest_experimental_gen_ai_semconv()`: unset means the latest
  conventions (upstream's default too), and a list omitting
  `gen_ai_latest_experimental` selects the baseline. Exactly one provider
  attribute is emitted — on spans and on the metrics attributes — and the four
  above-baseline attributes (`gen_ai.usage.cache_creation.input_tokens`,
  `gen_ai.usage.cache_read.input_tokens`,
  `gen_ai.usage.reasoning.output_tokens`, `gen_ai.tool.definitions`) plus
  `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` are withheld at the
  baseline. Under the default nothing changes except that `gen_ai.system` is
  no longer emitted alongside `gen_ai.provider.name`.
  - API: `record_request`, `record_response`, `record_tool_arguments` and
    `record_tool_result` take `&ObservabilityConfig` in place of a
    `capture_content: bool`; `chat_span` takes the semconv flag;
    `ObservableChatClient` gained `with_observability_config`
    (`with_content_capture` still works and now sets the flag on the config).

### Added

- **`Error::MiddlewareFailure`, a fail-closed signal for function middleware**
  (upstream #7562). The function-invocation loop absorbs every error a tool or
  its middleware produces into a tool-error result, hands it to the model and
  keeps looping — the right default for a tool failure the model can route
  around, but it left an enforcement layer (a guardrail, a policy or
  authorization gate) no way to stop a run: refusing a call just produced an
  error string the model could try again. Middleware returning
  `Error::middleware_failure(..)` is now propagated instead of absorbed, and
  because the parallel batch is driven by `try_join_all`, propagating it also
  drops the sibling calls still in flight. Every other error keeps the
  absorb-and-continue contract unchanged.

### Fixed

- **Replaying a conversation duplicated stored history** (upstream #7242). A
  history provider is handed a run's input plus its response, so a caller that
  keeps its own transcript and replays all of it each turn handed back
  everything already stored — and each provider appended it unconditionally.
  History grew superlinearly, and since `before_run` prepends it to the
  request, the duplicated turns were resent to the model on every later run.
  `agent_framework_core::history::filter_new_messages` now aligns the stored
  run inside the incoming one (by `message_id`, or by role and contents when
  there is none) and stores only what follows it; `InMemoryHistoryProvider`,
  `FileHistoryProvider`, `RedisChatMessageStore` and `CosmosChatMessageStore`
  all use it, the last two reading their stored history first (and skipping
  that read entirely when there is nothing to store, or when a Redis store is
  configured to retain nothing). A run that cannot be aligned is appended
  exactly as before — unlike upstream, no set-based fallback drops a turn that
  merely repeats an earlier one. Alignment sees a run's **input** only:
  response messages were just generated and can never be a replay, so they are
  always stored, even when one happens to reproduce the stored tail.
- **A replayed transcript was also sent to the model twice.** Storing only the
  new suffix fixed history growth, but the request is assembled the other way
  round — injected context first, then the caller's input — so a history
  provider that unconditionally injected what it held sent `q1, a1, q1, a1,
  q2` for a caller replaying `q1, a1, q2`. All four history providers now
  inject nothing when the run's input already carries the stored run
  (`inject_stored_history`).
  `StoredHistory::{Complete, Window}` selects where a stored run is looked for:
  a complete history is matched at the start only — it begins at the
  conversation's first message, so a replay of it can only begin with it — while
  a window is searched for, preferring a match at the start (an at-cap list that
  has never actually been trimmed is still complete) and otherwise taking the
  last occurrence. The Redis store asks for `Window` only when its list is at
  its cap. Alignment also requires *evidence* that stored history could be a
  replay at all — a matching message id, or a non-user turn, since a replay is
  a transcript and carries the assistant's replies. Stored history that is
  nothing but id-less user messages is indistinguishable from new input that
  repeats it, and is left alone. An empty `message_id` counts as no id at all,
  matching the `!id.is_empty()` guard the crate already applies elsewhere.
- **Tool spans could report a different semconv version than chat spans.** The
  function-invocation loop rebuilt an `ObservabilityConfig` from the
  environment for every tool call, so a client configured explicitly for one
  convention version emitted tool spans under whatever the environment said.
  `FunctionInvokingChatClient` now carries the config, settable with
  `with_observability_config` and resolved from the environment once at
  construction. `AgentBuilder::observability_config` reaches that wrapper,
  which the builder constructs itself.
- **A Redis retention limit of zero retained everything** (upstream #7470).
  `RedisChatMessageStore::with_max_messages(0)` is a request to retain
  nothing — unlimited is expressed by not calling it at all — but trimming to
  `-(max)` emits `LTRIM key 0 -1` for a limit of zero, which is Redis's "keep
  the whole list", so the trim ran on every save and did nothing.
  `add_messages` now returns before serializing, so no payload reaches Redis
  (or an AOF or a replica) even briefly. Stored history is deliberately left
  alone rather than deleted: the key carries no per-provider discriminator, so
  two stores sharing a prefix and session id address the same list, and
  deleting it would drop a co-located store's history. Use `clear` to remove
  history. A negative limit — the other half of upstream's fix — is
  unrepresentable here, since `max_messages` is a `usize`.
- **Gemini 3 thought signatures were dropped across an approval round trip**
  (upstream #7546). Gemini 3 rejects a `functionCall` part that lacks the
  `thoughtSignature` it was issued with. Signatures were paired to calls by
  adjacency alone, and any intervening content cleared the held signature —
  so a `FunctionApprovalResponse` sitting between a reasoning carrier and its
  call dropped it, and a call replayed in a later message could never be
  signed at all. Both turns then failed with a 400. Content that emits no
  wire Part no longer clears the signature, and a `call_id -> signature` map
  accumulated as the conversation is emitted signs a later replay. Precedence
  is unchanged: the call's own `protected_data`, then an adjacent carrier,
  then the map. The map is written by the emit walk rather than a pre-pass, so
  the pairing rules have a single implementation — a pre-pass that restated
  them laxly would re-sign the very calls adjacency had refused.
- **Stateless Responses requests never asked for the encrypted reasoning
  item.** A reasoning item is only replayable on the next turn of a `store:
  false` tool loop if it carries `encrypted_content`, and the service only
  returns that when the request's `include` asks for it. This port set
  `include` nowhere, so the replay path in `messages_to_input` — which
  re-emits the item verbatim and drops one lacking `id`/`encrypted_content` —
  had nothing valid to re-send. Both Responses clients (OpenAI and Azure
  OpenAI) now add `reasoning.encrypted_content` when a request carries no
  service-side-storage indicator, matching upstream. A caller's own `include`
  entries are preserved and never duplicated.
- **Foundry opts out of the above**, matching upstream #7536: it does not want
  encrypted reasoning unless asked for by name. New
  `AzureOpenAIResponsesClient::without_implicit_encrypted_reasoning`, which
  `FoundryChatClient` sets on its transport. An explicitly requested
  `reasoning.encrypted_content` is still honored.
- **The tool loop reported only its last iteration's token usage.** Each model
  call in the function-invocation loop reports its own usage, and every exit
  path returned the final call's `ChatResponse` untouched — so a run that
  called tools five times reported roughly a fifth of the tokens it spent, and
  the `gen_ai.usage.*` OTel metrics (which read `usage_details`) under-reported
  with it. Usage is now summed across every iteration and applied to whichever
  response the loop returns, including the approval pause, the
  declaration-only hand-back, and the tools-disabled failsafe. A run where no
  iteration reported usage still reports none rather than a synthesized zero.
  (upstream #7539)

## [0.2.0] — 2026-08-08

Upstream-alignment pass against `microsoft/agent-framework` `4b1afd90`
(2026-08-07), re-baselining from `beb65b21` (2026-07-13). See
[`ALIGNMENT_PROGRESS.md`](./ALIGNMENT_PROGRESS.md) for the full triage of the
112 intervening upstream commits, including the items deliberately left open.

### Fixed

- **Structured output was parsed from the wrong text.** The JSON value was
  built from every message joined with separators and included reasoning
  content, so a tool result could be mistaken for the answer, chain-of-thought
  was folded into the payload, and a JSON document split across streaming text
  chunks had separators injected into it. Now taken from the last non-empty
  assistant message's `text` contents, concatenated with no separator.
  (upstream #6990)
- **Anthropic streaming double-counted tokens.** Anthropic streams cumulative
  usage snapshots and the port summed them as if they were increments, so a
  response reporting 25 input tokens aggregated to 50. (upstream #7162)
- **OpenAI Chat Completions rejected some author names.** A `author_name`
  containing `/`, `|`, `\`, `<` or `>` was sent verbatim and failed the whole
  request with a 400; one containing a space was silently dropped. Names are
  now sanitized to `[a-zA-Z0-9_]` and truncated to 64 characters, matching the
  Python and .NET clients. (upstream #7126)
- **Gemini 3 function-call replays lost `thought_signature`.** Gemini 3
  requires the signature echoed when a call is replayed. Both placements are
  handled: on the function-call part itself (the usual one) and on a preceding
  thought part (backfill only). Reasoning content is no longer sent back as a
  part, matching upstream. (upstream #7095)
- **Approval replacement is now a single ordered walk.** The outstanding-call
  set is derived as the walk decides each content, rather than maintained as
  separate bookkeeping around an order-blind pre-scan — which netted a call
  against a result arriving *after* a replayed request and expanded the
  request into a duplicate declaration.
- **Approval round-trips could duplicate a function call.** The restored call
  was deduped against only the message being scanned, but a hosting layer
  replays the stored call and its approval request as two separate messages, so
  a second copy was restored and left unanswered — which the Responses API
  rejects with "No tool output found for function call ...". (upstream #7271)
- **Compaction could emit conversations providers reject.** A function call and
  its result are now retained or dropped together — previously `TokenBudget`
  could keep a tool result whose call fell outside the budget (a tool message
  answering nothing), and `SelectiveToolResult` deleted stale results while
  their assistant `tool_calls` entries remained (an unanswered call). Either
  half alone is a 400 on the next request. (upstream #7406)
- **Compaction could reduce a conversation to system messages only.**
  `Truncation`/`SlidingWindow` with a budget at or below the system prefix left
  no turn for the model to answer; the most recent non-system message is now
  retained even when that exceeds the limit. (upstream #7219)

### Changed

- **BREAKING: mem0 retrieval scope no longer inherits the storage scope.**
  `Mem0Provider::before_run` searched with the storage `user_id`/`agent_id`, so
  a provider configured with a shared `agent_id` retrieved memories written by
  every user of that agent and injected them into the current user's
  conversation. Retrieval now uses only `with_search_user_id` /
  `with_search_agent_id` / `with_search_application_id`; with none set, nothing
  is retrieved and a warning is logged once. Code that retrieved via
  `with_user_id` alone must add `with_search_user_id`. (upstream #7531)
- **`SelectiveToolResult` now replaces a stale tool result's payload with
  `OMITTED_TOOL_RESULT` instead of deleting the result content.** Deleting it
  orphaned the matching function call; replacing the payload sheds the same
  bulk while keeping the exchange valid. Mirrors the intent of upstream's
  `ToolResultCompactionStrategy`, which replaces stale tool groups with a
  compact stand-in rather than removing them.

### Added

- **Bedrock Converse image blocks.** The Bedrock converter previously dropped
  all `Data`/`Uri` content; inline images in Converse's accepted formats
  (`png`/`jpeg`/`gif`/`webp`) are now emitted as `{"image": ...}` blocks,
  closing a parity gap with upstream's Bedrock client.
- **`Content::renders_on_every_provider`.** The wire-visibility contract that
  compaction's minimum-retention logic depends on now lives on `Content`,
  pinned by contract tests in each provider crate — a converter change that
  invalidates a row fails a test next to the converter instead of surfacing as
  a compaction bug. `Uri` content is excluded (Bedrock has no remote-URL image
  source), and `Data` images are bounded by Bedrock's format set.
- `DataContent::from_uri` / `DataContent::media_type_from_uri`: validating
  construction from a `data:` URI, rejecting a missing scheme, missing `,`, or
  non-base64 declaration instead of silently mis-slicing it. (upstream #6916)
- `TextReasoningContent::protected_data`, mirroring upstream's
  `Content.protected_data`, for provider-opaque reasoning replay tokens.
- OpenAI cache-**write** token counts on both the Chat Completions and
  Responses surfaces, populating `UsageDetails::cache_creation_input_token_count`
  and so the `gen_ai.usage.cache_creation.input_tokens` OTel attribute.
  (upstream #7369)

## [0.1.1] — 2026-07-13

First published release on crates.io — identical in content to 0.1.0.

- The v0.1.0 release-pipeline run failed at the publish step (the crates.io
  token secret was misnamed), after the `v0.1.0` tag had already been
  pushed, so 0.1.0 was never published and its version number is burned.
- Release workflow: publish to crates.io **before** tagging and creating
  the GitHub Release, so a failed publish no longer burns the version —
  the run can simply be retried after fixing the cause.

## [0.1.0] — 2026-07-13

First release: a Rust implementation of the Microsoft Agent Framework at
structural parity with upstream (`microsoft/agent-framework`) as of upstream
commit `beb65b21` (2026-07-13). 22 crates.

### Core (`agent-framework-core`)

- Data model: `Message`, the `Content` union (text, reasoning, data/URI,
  function call/result, hosted tool calls/results, usage, approvals),
  `ChatResponse`/`AgentResponse` (+ streaming updates and aggregation),
  `ChatOptions` with Python `&`-merge semantics, structured output
  (`ResponseFormat` + `parse_json`), typed `UsageDetails` with
  cache/reasoning counts, **embeddings** (`Embedding`, `GeneratedEmbeddings`,
  `EmbeddingGenerationOptions`, `EmbeddingClient`).
- `ChatClient` trait + `FunctionInvokingChatClient`: parallel tool
  execution, human-in-the-loop approval gating, declaration-only (frontend)
  tools, **progressive tool exposure** (live `FunctionInvocationContext::tools`
  with `add_tools`/`remove_tools`), retry layer (`RetryingChatClient`)
  honoring `Retry-After`.
- Agents: `Agent`/`AgentBuilder`, three-level middleware (agent/chat/
  function), per-run options, dynamic `ToolSource`s (MCP catalog changes),
  `as_tool` (with `propagate_session` child-session semantics,
  `stream_callback`, `approval_mode`).
- Sessions: `AgentSession` + shared-by-reference `SessionState`,
  `HistoryProvider` (in-memory/file), context providers
  (`before_run`/`after_run`), history compaction (four strategies +
  `CompactionProvider`), skills (progressive disclosure), settings
  (`SecretString`, `load_setting`).
- Workflow engine: Pregel-style supersteps, checkpointing (+ resume,
  graph-signature validation), human-in-the-loop pause/resume, output
  designation, async edge conditions, shared state, sub-workflows,
  Mermaid/DOT visualization.
- Orchestrations: Sequential, Concurrent, GroupChat, Handoff (enforced mesh
  topology), Magentic (plan review + stall intervention HITL),
  `WorkflowAgent`, post-agent approval (`AgentApprovalExecutor`).
- Observability: OTel GenAI-semconv spans and (feature-gated) metrics.

### Providers

- OpenAI (Chat Completions + Responses + **embeddings**), Azure OpenAI
  (api-key + Entra ID credential chain, Responses, **embeddings**),
  Anthropic (incl. Bedrock/Vertex/Foundry cloud transports), AWS Bedrock
  (Converse, dependency-free SigV4), Foundry (Responses + Prompt Agents),
  Foundry Local, Gemini, Mistral (chat + **embeddings**), Ollama (chat +
  **embeddings**), GitHub Copilot (token exchange), Copilot Studio.

### Integrations & hosting

- MCP (stdio/HTTP/websocket, sampling, prompts, roots), A2A client + serving,
  declarative agents/workflows (Rust-native `WorkflowSpec`), hosting crate
  (DevUI-style API, AG-UI protocol, OpenAI-compat endpoint, security
  middleware), Redis / Mem0 / Azure AI Search context providers, Cosmos DB
  message store + checkpoint storage, Purview compliance middleware.

### Known divergences from upstream

Documented in `ALIGNMENT_PROGRESS.md` / `PARITY.md`: streaming is expressed
as Rust method pairs (`run`/`run_stream`); the declarative *workflow* DSL is
Rust-native rather than Power Platform; `durabletask`, the `@experimental`
harness/security/evaluation modules, and the Claude Agent SDK wrapper are out
of scope; DevUI's bundled-frontend routes are partial.
