//! Configuration and secret handling: `load_setting` resolves a single
//! setting through a fixed precedence chain, and `SecretString` keeps a
//! credential out of your logs.
//!
//! `load_setting(key, override_value, default)` tries, in order:
//!
//! 1. `override_value` — an explicit argument (a constructor parameter, a CLI
//!    flag), which always wins;
//! 2. a `./.env` file in the current working directory, looked up by `key`;
//! 3. the `key` process environment variable;
//! 4. `default`.
//!
//! `None` when nothing produced a value. This is the same precedence every
//! `from_env` constructor in the workspace follows, so reaching for it in
//! your own code keeps configuration behaving consistently.
//!
//! `SecretString` wraps a credential so that `Debug` and `Display` both print
//! `***`. That matters more than it looks: a `#[derive(Debug)]` on a config
//! struct, or a `tracing` span recording it, is the usual way an API key ends
//! up in a log file. Reading the real value requires the deliberately ugly
//! `expose_secret()`.
//!
//! Runs fully offline — it writes a throwaway `.env` in a temp directory to
//! demonstrate step 2, then cleans up.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example settings_and_secrets
//! ```

use agent_framework::prelude::*;

/// A config struct holding a credential. `Debug` is safe to log because the
/// secret masks itself — swap `SecretString` for `String` here and the key
/// lands in the output below.
#[derive(Debug)]
struct ProviderConfig {
    endpoint: String,
    api_key: SecretString,
    timeout_secs: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("== SecretString: masked by default ==\n");

    let config = ProviderConfig {
        endpoint: "https://api.example.com/v1".into(),
        api_key: SecretString::new("sk-live-abcdef0123456789"),
        timeout_secs: 30,
    };

    println!("  Debug:   {config:?}");
    println!("  Display: {}", config.api_key);
    println!("  real value (via expose_secret): {}", config.api_key.expose_secret());
    println!(
        "\n  Note the derived `Debug` on the whole struct is already safe -- \n  \
         that is the point: you cannot leak it by forgetting."
    );
    println!("  endpoint={} timeout={}s", config.endpoint, config.timeout_secs);

    // Equality compares the underlying value, so a config round-trip can
    // still be asserted on.
    assert_eq!(
        SecretString::new("sk-live-abcdef0123456789"),
        config.api_key
    );

    println!("\n== load_setting: the precedence chain ==\n");

    // Work in a temp directory so the `.env` written below cannot collide
    // with a real one in the repo.
    let dir = std::env::temp_dir().join(format!("af-settings-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| Error::Configuration(e.to_string()))?;
    let previous_dir = std::env::current_dir().map_err(|e| Error::Configuration(e.to_string()))?;
    std::env::set_current_dir(&dir).map_err(|e| Error::Configuration(e.to_string()))?;

    // 4. default only -- nothing else is set.
    show("nothing set", load_setting("DEMO_MODEL", None, Some("gpt-4o-mini".into())));
    show("nothing set, no default", load_setting("DEMO_MODEL", None, None));

    // 3. the process environment beats the default.
    //
    // SAFETY: single-threaded at this point in `main`; `set_var` is only
    // unsound when another thread may be reading the environment concurrently.
    unsafe { std::env::set_var("DEMO_MODEL", "from-the-environment") };
    show("env var set", load_setting("DEMO_MODEL", None, Some("gpt-4o-mini".into())));

    // 2. a ./.env file beats the process environment.
    std::fs::write(
        dir.join(".env"),
        "# a comment\nDEMO_MODEL=from-the-dotenv-file\nDEMO_REGION='eu-west-1'\n",
    )
    .map_err(|e| Error::Configuration(e.to_string()))?;
    show("./.env present", load_setting("DEMO_MODEL", None, Some("gpt-4o-mini".into())));
    show("./.env, quotes stripped", load_setting("DEMO_REGION", None, None));

    // 1. an explicit override beats everything.
    show(
        "explicit override",
        load_setting("DEMO_MODEL", Some("from-the-caller".into()), Some("gpt-4o-mini".into())),
    );

    // Clean up: restore the working directory and drop the temp files.
    std::env::set_current_dir(&previous_dir).map_err(|e| Error::Configuration(e.to_string()))?;
    unsafe { std::env::remove_var("DEMO_MODEL") };
    let _ = std::fs::remove_dir_all(&dir);

    println!(
        "\n  The chain is deliberately 'closest to the caller wins': an explicit\n  \
         argument overrides a checked-in .env, which overrides whatever the\n  \
         deployment environment happens to have set."
    );

    println!("\n== wiring it into your own client ==\n");
    println!(
        "  let key = load_setting(\"MY_PROVIDER_API_KEY\", override_key, None)\n      \
         .map(SecretString::new)\n      \
         .ok_or_else(|| Error::Configuration(\"MY_PROVIDER_API_KEY is not set\".into()))?;\n\n  \
         -- then pass `key.expose_secret()` to the one place that builds the\n  \
         Authorization header, and keep the `SecretString` everywhere else."
    );

    Ok(())
}

fn show(label: &str, value: Option<String>) {
    println!("  {label:<26} -> {value:?}");
}
