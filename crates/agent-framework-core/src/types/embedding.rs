//! Embedding generation types.
//!
//! Rust equivalent of upstream's `Embedding` / `GeneratedEmbeddings` /
//! `EmbeddingGenerationOptions` (`_types.py`), plus [`EmbeddingInput`] for
//! upstream's `EmbeddingInputT`. The client-side counterpart —
//! the [`EmbeddingClient`](crate::client::EmbeddingClient) trait mirroring
//! upstream's `SupportsGetEmbeddings` protocol — lives in
//! [`crate::client`], next to `ChatClient`.
//!
//! Upstream is generic over the vector element type (`list[float]`,
//! `list[int]`, `bytes`, …); this port fixes vectors to `Vec<f32>` — the
//! shape every wire API here actually returns — rather than threading a
//! type parameter through the whole trait surface.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::content::{Content, UsageDetails};
use crate::error::{Error, Result};

/// One value to embed.
///
/// Upstream is generic over the input type (`EmbeddingInputT`): text-only
/// clients take `str`, while Foundry takes `Content | str` and Gemini takes
/// `str` or a multimodal `Content`. This port uses one input type for every
/// client so the trait stays object-safe. An input is a list of
/// [`Content`] items embedded together as **one** vector: usually a single
/// text, or a single image as [`Content::Data`] or [`Content::Uri`], and
/// several items where a provider can combine them (Gemini embeds text and
/// media parts together; Foundry pairs an image with a caption).
///
/// Strings convert directly, so text callers write
/// `vec!["hello".into()]`:
///
/// ```
/// # use agent_framework_core::types::{Content, DataContent, EmbeddingInput};
/// let text: EmbeddingInput = "a cat on a mat".into();
/// assert_eq!(text.as_text(), Some("a cat on a mat"));
///
/// let png = [0x89, 0x50, 0x4e, 0x47];
/// let image = EmbeddingInput::from(Content::Data(DataContent::from_bytes(&png, "image/png")));
/// assert_eq!(image.as_text(), None);
/// ```
///
/// A client that cannot embed some input returns an error naming its index
/// rather than skipping it, so results always line up with inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingInput {
    /// The items embedded together into one vector.
    pub contents: Vec<Content>,
}

impl EmbeddingInput {
    /// An input from content items embedded together.
    pub fn new(contents: Vec<Content>) -> Self {
        Self { contents }
    }

    /// A text input.
    pub fn text(text: impl Into<String>) -> Self {
        Self::new(vec![Content::text(text)])
    }

    /// The text, when this input is exactly one text item.
    pub fn as_text(&self) -> Option<&str> {
        match self.contents.as_slice() {
            [Content::Text(t)] => Some(&t.text),
            _ => None,
        }
    }

    /// Converts a batch to plain strings for a client that embeds text only.
    ///
    /// Fails on the first input that is not exactly one text item, naming
    /// `provider` and the input's index.
    pub fn into_texts(values: Vec<EmbeddingInput>, provider: &str) -> Result<Vec<String>> {
        values
            .into_iter()
            .enumerate()
            .map(
                |(i, value)| match <[Content; 1]>::try_from(value.contents) {
                    Ok([Content::Text(t)]) => Ok(t.text),
                    _ => Err(Error::Content(format!(
                    "{provider} embeddings accept text only; input {i} is not a single text item"
                ))),
                },
            )
            .collect()
    }
}

impl From<&str> for EmbeddingInput {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

impl From<String> for EmbeddingInput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&String> for EmbeddingInput {
    fn from(text: &String) -> Self {
        Self::text(text.as_str())
    }
}

impl From<Content> for EmbeddingInput {
    fn from(content: Content) -> Self {
        Self::new(vec![content])
    }
}

impl From<Vec<Content>> for EmbeddingInput {
    fn from(contents: Vec<Content>) -> Self {
        Self::new(contents)
    }
}

/// Common request settings for embedding generation.
///
/// All fields are optional. Provider-specific settings (e.g. OpenAI's
/// `encoding_format` or `user`) ride in `additional_properties`, exactly
/// like [`ChatOptions::additional_properties`](super::ChatOptions) —
/// upstream expresses the same extension point as per-provider TypedDict
/// subclasses.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingGenerationOptions {
    /// The embedding model to use; falls back to the client's default.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model: Option<String>,
    /// Requested output dimensionality, for models that support shortening
    /// (e.g. `text-embedding-3-*`).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub dimensions: Option<u32>,
    /// Provider-specific extras, forwarded by the provider converters that
    /// understand them.
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_properties: HashMap<String, Value>,
}

impl EmbeddingGenerationOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Builder: set the requested output dimensionality.
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

/// A single embedding vector with metadata.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// The model that generated this embedding, when known.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model: Option<String>,
}

impl Embedding {
    /// An embedding from a bare vector.
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            model: None,
        }
    }

    /// The number of dimensions (the vector's length — upstream computes the
    /// same when no explicit count is supplied).
    pub fn dimensions(&self) -> usize {
        self.vector.len()
    }
}

/// A batch of generated embeddings plus usage metadata.
///
/// Upstream subclasses `list`; here the batch derefs to `[Embedding]`, so
/// indexing and iteration work directly on the result:
///
/// ```
/// # use agent_framework_core::types::{Embedding, GeneratedEmbeddings};
/// let batch = GeneratedEmbeddings::new(vec![
///     Embedding::new(vec![0.1, 0.2]),
///     Embedding::new(vec![0.3, 0.4]),
/// ]);
/// assert_eq!(batch.len(), 2);
/// assert_eq!(batch[0].dimensions(), 2);
/// for e in &batch {
///     assert_eq!(e.dimensions(), 2);
/// }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GeneratedEmbeddings {
    /// The embeddings, in input order.
    pub embeddings: Vec<Embedding>,
    /// Token usage for the batch, when the service reports it.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub usage: Option<UsageDetails>,
    /// Provider-specific extras.
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_properties: HashMap<String, Value>,
}

impl GeneratedEmbeddings {
    /// A batch from a list of embeddings, with no usage metadata.
    pub fn new(embeddings: Vec<Embedding>) -> Self {
        Self {
            embeddings,
            usage: None,
            additional_properties: HashMap::new(),
        }
    }

    /// Builder: attach usage metadata.
    pub fn with_usage(mut self, usage: UsageDetails) -> Self {
        self.usage = Some(usage);
        self
    }
}

impl std::ops::Deref for GeneratedEmbeddings {
    type Target = [Embedding];
    fn deref(&self) -> &Self::Target {
        &self.embeddings
    }
}

impl IntoIterator for GeneratedEmbeddings {
    type Item = Embedding;
    type IntoIter = std::vec::IntoIter<Embedding>;
    fn into_iter(self) -> Self::IntoIter {
        self.embeddings.into_iter()
    }
}

impl<'a> IntoIterator for &'a GeneratedEmbeddings {
    type Item = &'a Embedding;
    type IntoIter = std::slice::Iter<'a, Embedding>;
    fn into_iter(self) -> Self::IntoIter {
        self.embeddings.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataContent, UriContent};

    #[test]
    fn embedding_dimensions_is_the_vector_length() {
        assert_eq!(Embedding::new(vec![0.0; 7]).dimensions(), 7);
        assert_eq!(Embedding::new(vec![]).dimensions(), 0);
    }

    #[test]
    fn generated_embeddings_deref_and_iterate() {
        let batch =
            GeneratedEmbeddings::new(vec![Embedding::new(vec![0.1]), Embedding::new(vec![0.2])]);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert_eq!(batch[1].vector, vec![0.2]);
        let collected: Vec<usize> = (&batch).into_iter().map(Embedding::dimensions).collect();
        assert_eq!(collected, vec![1, 1]);
    }

    #[test]
    fn options_builders_set_fields() {
        let options = EmbeddingGenerationOptions::new()
            .with_model("text-embedding-3-small")
            .with_dimensions(256);
        assert_eq!(options.model.as_deref(), Some("text-embedding-3-small"));
        assert_eq!(options.dimensions, Some(256));
    }

    #[test]
    fn generated_embeddings_serialize_round_trip() {
        let batch = GeneratedEmbeddings::new(vec![Embedding::new(vec![0.5, -0.5])]).with_usage(
            UsageDetails {
                input_token_count: Some(3),
                total_token_count: Some(3),
                ..Default::default()
            },
        );
        let json = serde_json::to_value(&batch).unwrap();
        assert_eq!(
            json["embeddings"][0]["vector"],
            serde_json::json!([0.5, -0.5])
        );
        let back: GeneratedEmbeddings = serde_json::from_value(json).unwrap();
        assert_eq!(back, batch);
    }

    #[test]
    fn strings_become_single_text_inputs() {
        let inputs: Vec<EmbeddingInput> = vec!["a".into(), String::from("b").into()];
        assert_eq!(
            EmbeddingInput::into_texts(inputs, "Test").unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn a_text_only_client_names_the_first_non_text_input() {
        let image = Content::Uri(UriContent {
            uri: "https://example.com/a.png".into(),
            media_type: "image/png".into(),
        });
        let inputs = vec![
            EmbeddingInput::text("a"),
            image.clone().into(),
            EmbeddingInput::new(vec![Content::text("b"), Content::text("c")]),
        ];
        let err = EmbeddingInput::into_texts(inputs, "Test").unwrap_err();
        assert!(err.to_string().contains("input 1"), "{err}");

        let two_texts = vec![EmbeddingInput::new(vec![
            Content::text("b"),
            Content::text("c"),
        ])];
        assert!(EmbeddingInput::into_texts(two_texts, "Test").is_err());
        assert!(EmbeddingInput::into_texts(vec![EmbeddingInput::new(vec![])], "Test").is_err());
    }

    #[test]
    fn as_text_is_none_for_media_and_multi_part_inputs() {
        let data = Content::Data(DataContent::from_bytes(&[1, 2, 3], "image/png"));
        assert_eq!(EmbeddingInput::from(data.clone()).as_text(), None);
        assert_eq!(
            EmbeddingInput::new(vec![Content::text("caption"), data]).as_text(),
            None
        );
    }
}
