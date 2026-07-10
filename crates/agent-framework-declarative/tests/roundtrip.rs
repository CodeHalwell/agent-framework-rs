//! YAML round-trip stability: `spec -> yaml -> spec` is the identity for the
//! real reference samples.

use agent_framework_declarative::AgentSpec;

const GET_WEATHER: &str = include_str!("specs/GetWeather.yaml");
const OPENAI_CHAT: &str = include_str!("specs/OpenAIChat.yaml");
const MS_LEARN: &str = include_str!("specs/MicrosoftLearnAgent.yaml");

fn assert_round_trips(yaml: &str, label: &str) {
    let spec = AgentSpec::from_yaml(yaml).unwrap_or_else(|e| panic!("{label}: parse: {e}"));
    let serialized = spec
        .to_yaml()
        .unwrap_or_else(|e| panic!("{label}: serialize: {e}"));
    let reparsed =
        AgentSpec::from_yaml(&serialized).unwrap_or_else(|e| panic!("{label}: reparse: {e}"));
    assert_eq!(spec, reparsed, "{label}: spec changed across round-trip");
}

#[test]
fn get_weather_round_trips() {
    assert_round_trips(GET_WEATHER, "GetWeather");
}

#[test]
fn openai_chat_round_trips() {
    assert_round_trips(OPENAI_CHAT, "OpenAIChat");
}

#[test]
fn microsoft_learn_round_trips() {
    assert_round_trips(MS_LEARN, "MicrosoftLearnAgent");
}

#[test]
fn constructed_spec_round_trips() {
    let yaml = "\
kind: Prompt
name: Builder
description: A built agent
instructions: Do the thing.
additionalInstructions: And do it well.
model:
  id: gpt-4o
  provider: OpenAI
  apiType: Chat
  options:
    temperature: 0.5
    maxOutputTokens: 512
    stopSequences:
      - STOP
    additionalProperties:
      customFlag: true
tools:
  - kind: web_search
    name: search
    description: search the web
outputSchema:
  strict: true
  properties:
    answer:
      kind: string
      required: true
";
    let spec = AgentSpec::from_yaml(yaml).expect("parse");
    let reparsed = AgentSpec::from_yaml(&spec.to_yaml().expect("serialize")).expect("reparse");
    assert_eq!(spec, reparsed);

    // Spot-check a couple of the trickier mappings survived.
    let options = spec.model.unwrap().options.unwrap();
    assert_eq!(options.max_output_tokens, Some(512));
    assert_eq!(
        options.additional_properties.get("customFlag"),
        Some(&serde_json::json!(true))
    );
}
