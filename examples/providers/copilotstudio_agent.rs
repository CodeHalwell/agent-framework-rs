//! Talk to a published Microsoft Copilot Studio agent via the Direct-to-
//! Engine (D2E) protocol. `CopilotStudioAgent` implements the `SupportsAgentRun` trait;
//! reusing one `AgentThread` keeps the Copilot Studio conversation id across
//! turns.
//!
//! Skips gracefully unless configured. Uses Python's env-var conventions:
//!   COPILOTSTUDIOAGENT__ENVIRONMENTID   the Power Platform environment id
//!   COPILOTSTUDIOAGENT__SCHEMANAME      the agent's schema name
//! plus COPILOTSTUDIO_TOKEN -- a pre-acquired OAuth bearer token for the
//! Power Platform API (acquire via MSAL / az CLI; token acquisition itself
//! is out of scope for the crate, exactly as in the Python package).
//!
//! ```bash
//! COPILOTSTUDIOAGENT__ENVIRONMENTID=... COPILOTSTUDIOAGENT__SCHEMANAME=... \
//! COPILOTSTUDIO_TOKEN=eyJ... \
//! cargo run -p agent-framework-examples --example copilotstudio_agent
//! ```

use agent_framework::copilotstudio::{
    CopilotStudioConnectionSettings, CopilotStudioSettings, StaticTokenProvider,
};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Reads the COPILOTSTUDIOAGENT__* variables (absent -> None).
    let settings = CopilotStudioSettings::from_env();
    let token = std::env::var("COPILOTSTUDIO_TOKEN").ok();

    let (Some(_), Some(_), Some(token)) = (&settings.environment_id, &settings.schema_name, token)
    else {
        println!(
            "set COPILOTSTUDIOAGENT__ENVIRONMENTID, COPILOTSTUDIOAGENT__SCHEMANAME, \
             and COPILOTSTUDIO_TOKEN to run this example"
        );
        return Ok(());
    };

    // Validates the required fields and derives the D2E endpoint from the
    // environment id + cloud (defaults to the public Power Platform cloud).
    let connection = CopilotStudioConnectionSettings::from_settings(&settings)?;
    let agent = CopilotStudioAgent::new(connection, StaticTokenProvider::new(token))
        .with_name("copilot-studio");

    // A thread keeps the server-side conversation going across turns.
    let mut thread = agent.get_new_thread();
    let response = agent
        .run(
            vec![Message::user("Hello! What can you help with?")],
            Some(&mut thread),
        )
        .await?;
    println!("{}", response.text());

    Ok(())
}
