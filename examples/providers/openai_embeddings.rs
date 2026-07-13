//! Generate text embeddings with the OpenAI embeddings API.
//!
//! The same `EmbeddingClient` trait is implemented by
//! `AzureOpenAIEmbeddingClient` (deployment-scoped, api-key or Entra ID),
//! `OllamaEmbeddingClient` (local, no key), and `MistralEmbeddingClient`
//! (`mistral-embed`) — swap the constructor and the rest is identical.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example openai_embeddings
//! ```

use agent_framework::prelude::*;

/// Cosine similarity between two equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIEmbeddingClient::from_env("text-embedding-3-small")?;

    let inputs = vec![
        "The cat sat on the mat.".to_string(),
        "A feline rested on the rug.".to_string(),
        "Quarterly revenue grew by 12%.".to_string(),
    ];
    let batch = client
        .get_embeddings(
            inputs.clone(),
            // Shortened vectors: text-embedding-3-* supports requesting a
            // lower dimensionality.
            Some(EmbeddingGenerationOptions::new().with_dimensions(256)),
        )
        .await?;

    for (text, embedding) in inputs.iter().zip(&batch) {
        println!("{} dims  <-  {text}", embedding.dimensions());
    }
    if let Some(usage) = &batch.usage {
        println!("tokens: {:?}", usage.input_token_count);
    }

    // The two cat sentences should be far more similar to each other than to
    // the finance one.
    println!(
        "similarity(cat, feline)  = {:.3}",
        cosine(&batch[0].vector, &batch[1].vector)
    );
    println!(
        "similarity(cat, revenue) = {:.3}",
        cosine(&batch[0].vector, &batch[2].vector)
    );

    Ok(())
}
