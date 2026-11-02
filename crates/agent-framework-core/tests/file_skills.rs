//! File-based skills discovered from disk (`file-skills`).
#![cfg(feature = "file-skills")]

use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::skills::SkillsProvider;
use agent_framework_core::skills_files::FileSkillsSource;
use serde_json::json;

fn write(path: &std::path::Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

#[tokio::test]
async fn skills_are_discovered_with_resources_and_served() {
    let root = std::env::temp_dir().join(format!("af-skills-{}", uuid::Uuid::new_v4().simple()));
    write(
        &root.join("pdf/SKILL.md"),
        "---\nname: pdf\ndescription: Read and fill PDF forms.\n---\nUse the reference.\n",
    );
    write(
        &root.join("pdf/reference.md"),
        "Form fields live in AcroForm.",
    );
    write(&root.join("pdf/docs/fields.json"), "{\"a\": 1}");
    write(&root.join("pdf/docs/deep/ignored.md"), "too deep");
    write(&root.join("pdf/script.py"), "print('no')");
    write(
        &root.join("pdf/nested/SKILL.md"),
        "---\nname: nested\ndescription: d\n---\n",
    );
    write(
        &root.join("group/weather/SKILL.md"),
        "---\nname: weather\ndescription: Forecasts.\n---\nCall the API.\n",
    );
    write(
        &root.join("broken/SKILL.md"),
        "---\nname: Not Valid\ndescription: d\n---\n",
    );
    #[cfg(unix)]
    {
        let outside = root.with_extension("outside");
        write(
            &outside.join("SKILL.md"),
            "---\nname: evil\ndescription: d\n---\n",
        );
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
        std::os::unix::fs::symlink(root.join("pdf/reference.md"), root.join("pdf/alias.md"))
            .unwrap();
    }

    let skills = FileSkillsSource::new([&root]).load();
    let mut names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
    names.sort();
    assert_eq!(
        names,
        ["pdf", "weather"],
        "invalid, nested and linked skills are skipped"
    );
    let pdf = skills.iter().find(|s| s.name == "pdf").unwrap();
    let mut resources: Vec<&String> = pdf.resources.keys().collect();
    resources.sort();
    assert_eq!(resources, ["docs/fields.json", "reference.md"]);
    assert!(pdf.instructions.starts_with("Use the reference."));
    assert!(pdf
        .instructions
        .contains("<resource name=\"reference.md\" />"));

    let provider = SkillsProvider::from_paths([&root]).unwrap();
    let mut ctx = SessionContext::new(vec![]);
    provider.before_run(&mut ctx).await.unwrap();
    assert!(ctx
        .instructions
        .unwrap()
        .contains("- pdf: Read and fill PDF forms."));
    let read = ctx
        .tools
        .iter()
        .find(|t| t.name == "read_skill_resource")
        .unwrap();
    let out = read
        .executor
        .clone()
        .unwrap()
        .invoke(json!({"skill_name": "pdf", "resource_name": "reference.md"}))
        .await
        .unwrap();
    assert_eq!(out, "Form fields live in AcroForm.");

    let shallow = FileSkillsSource::new([root.join("pdf")])
        .with_search_depth(1)
        .unwrap()
        .load();
    assert_eq!(
        shallow[0].resources.len(),
        1,
        "depth 1 is the skill root only"
    );
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(root.with_extension("outside"));
}
