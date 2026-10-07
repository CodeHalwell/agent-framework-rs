//! # agent-framework-tools
//!
//! Built-in tools for agents, porting upstream's `agent-framework-tools`
//! package (Python) and `Microsoft.Agents.AI.Tools.Shell` (.NET).
//!
//! It currently holds the [`shell`] tool: [`LocalShellTool`] runs commands on
//! the host behind human approval, and [`DockerShellTool`] runs them in a
//! locked-down container. Read the [`shell`] module's security model before
//! enabling either.
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use agent_framework_core::prelude::*;
//! use agent_framework_tools::shell::{LocalShellTool, ShellEnvironmentProvider};
//!
//! # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
//! let shell = LocalShellTool::builder().workdir("/srv/project").build()?;
//! let agent = Agent::builder(client)
//!     .instructions("You are a careful operator.")
//!     .tool(shell.as_function())
//!     .context_provider(Arc::new(ShellEnvironmentProvider::new(Arc::new(shell.clone()), None)))
//!     .build();
//! let response = agent.run_once("How much disk space is free?").await?;
//! // Every shell call comes back as an approval request first.
//! for request in response.user_input_requests() {
//!     println!("approve? {:?}", request.function_call.arguments);
//! }
//! # Ok(())
//! # }
//! ```

pub mod shell;

pub use shell::{DockerShellTool, LocalShellTool, ShellPolicy, ShellResult};
