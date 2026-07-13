//! Embedding generation types.
//!
//! Rust equivalent of upstream's `Embedding` / `GeneratedEmbeddings` /
//! `EmbeddingGenerationOptions` (`_types.py`). The client-side counterpart —
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

use super::content::UsageDetails;

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
}
