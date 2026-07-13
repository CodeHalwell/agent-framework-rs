//! Conversation-history compaction strategies, applied directly to a message
//! list via `agent_framework_core::compaction`: a `Tokenizer`
//! (`ApproxTokenizer`, a ~4-chars-per-token heuristic) plus four strategies --
//! `Truncation` (keep the most recent N messages), `SlidingWindow` (keep the
//! last N *non-system* messages), `TokenBudget` (keep what fits a token
//! budget, newest first), and `SelectiveToolResult` (strip stale tool-result
//! content, the bulkiest kind, keeping only the most recent). All strategies
//! preserve any leading system message(s), and compaction never errors -- it
//! always returns *some* retained subset in original order.
//!
//! To wire a strategy into an agent so history is compacted automatically on
//! every run, see the `compaction_provider` example.
//!
//! Runs fully offline -- no API key or network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example compaction_basics
//! ```

use agent_framework::compaction::count_message_tokens;
use agent_framework::prelude::*;
use serde_json::json;

/// Render one message as a short single line for the tables below.
fn describe(m: &Message) -> String {
    let text = m.text();
    if !text.is_empty() {
        return format!("{:<9} {text}", m.role.as_str());
    }
    let tool_results = m.function_results();
    if !tool_results.is_empty() {
        return format!(
            "{:<9} [tool result: {}]",
            m.role.as_str(),
            tool_results[0].call_id
        );
    }
    format!("{:<9} [non-text content]", m.role.as_str())
}

fn print_messages(label: &str, messages: &[Message]) {
    let tokenizer = ApproxTokenizer;
    let total: usize = messages
        .iter()
        .map(|m| count_message_tokens(&tokenizer, m))
        .sum();
    println!(
        "{label} ({} message(s), ~{total} text tokens):",
        messages.len()
    );
    for m in messages {
        println!("    {}", describe(m));
    }
    println!();
}

/// A tool-role message carrying one `Content::FunctionResult`.
fn tool_result(call_id: &str, payload: &str) -> Message {
    Message::with_contents(
        Role::tool(),
        vec![Content::FunctionResult(FunctionResultContent::new(
            call_id,
            Some(json!(payload)),
        ))],
    )
}

fn main() -> Result<()> {
    // A conversation with a system prompt, a few user/assistant turns, and
    // two (bulky) tool results from earlier lookups.
    let messages = vec![
        Message::system("You are a travel assistant."),
        Message::user("Find flights from Oslo to Lisbon next Friday."),
        tool_result(
            "call_flights",
            "flight OS1234 08:15, flight TP567 11:40, ...",
        ),
        Message::assistant("Two options: OS1234 at 08:15 or TP567 at 11:40."),
        Message::user("What's the weather like in Lisbon then?"),
        tool_result("call_weather", "22C, sunny, light breeze"),
        Message::assistant("Around 22C and sunny."),
        Message::user("Great -- book the 11:40 one."),
    ];
    print_messages("-- original conversation", &messages);

    let tokenizer = ApproxTokenizer;

    // Truncation: keep the most recent `max_messages` overall (the leading
    // system message always survives and counts against the budget).
    let strategy = Truncation::new(4);
    let out = compact(&messages, &strategy, &tokenizer);
    print_messages("-- Truncation::new(4): system + the 3 most recent", &out);

    // SlidingWindow: keep the last `window` *non-system* messages -- the
    // system prefix rides along for free, unlike Truncation.
    let strategy = SlidingWindow::new(3);
    let out = compact(&messages, &strategy, &tokenizer);
    print_messages("-- SlidingWindow::new(3): system + the 3 most recent", &out);

    // TokenBudget: walk newest-to-oldest keeping whatever fits `max_tokens`
    // (the newest non-system message is always kept, even over budget).
    let strategy = TokenBudget::new(30);
    let out = compact(&messages, &strategy, &tokenizer);
    print_messages(
        "-- TokenBudget::new(30): newest messages that fit ~30 tokens",
        &out,
    );

    // SelectiveToolResult: strip `Content::FunctionResult` from all but the
    // last `keep_last` tool-result-bearing messages -- stale tool output is
    // the bulkiest, least useful history. Messages left empty are dropped;
    // other content is untouched.
    let strategy = SelectiveToolResult::new(1);
    let out = compact(&messages, &strategy, &tokenizer);
    print_messages(
        "-- SelectiveToolResult::new(1): only the newest tool result kept",
        &out,
    );
    assert_eq!(
        out.len(),
        messages.len() - 1,
        "one stale tool result dropped"
    );

    println!("(every strategy preserved the leading system message and original order)");
    Ok(())
}
