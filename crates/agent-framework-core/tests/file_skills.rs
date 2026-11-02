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

#[tokio::test]
async fn file_scripts_are_discovered_and_run_by_the_runner() {
    use agent_framework_core::skills::SkillScriptSource;
    let root = std::env::temp_dir().join(format!("af-scripts-{}", uuid::Uuid::new_v4().simple()));
    write(
        &root.join("data/SKILL.md"),
        "---\nname: data\ndescription: Crunch data.\n---\nRun the script.\n",
    );
    write(&root.join("data/scripts/process.py"), "print('hi')");
    write(&root.join("data/notes.txt"), "not a script");

    let skills = FileSkillsSource::new([&root]).load();
    let data = &skills[0];
    assert_eq!(data.path.as_deref(), Some(root.join("data").as_path()));
    let names: Vec<&str> = data.scripts.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["scripts/process.py"]);
    assert!(matches!(
        data.scripts[0].source,
        SkillScriptSource::File { .. }
    ));
    assert!(data
        .instructions
        .contains("<script name=\"scripts/process.py\">"));
    assert!(!data.resources.contains_key("scripts/process.py"));

    // Without a runner, running a file script is an error of the call.
    let provider = SkillsProvider::new(skills.clone());
    let mut ctx = SessionContext::new(vec![]);
    provider.before_run(&mut ctx).await.unwrap();
    let run = ctx
        .tools
        .iter()
        .find(|t| t.name == "run_skill_script")
        .unwrap()
        .clone();
    assert!(
        run.requires_approval(),
        "running code needs approval by default"
    );
    let call =
        json!({"skill_name": "data", "script_name": "SCRIPTS/process.py", "args": ["--fast"]});
    assert!(run
        .executor
        .clone()
        .unwrap()
        .invoke(call.clone())
        .await
        .is_err());

    // With one, the runner gets the skill, the script and the args.
    let provider = SkillsProvider::new(skills)
        .with_script_runner(|skill, script, args| async move {
            Ok(json!({
                "skill": skill.name,
                "file": script.full_path().unwrap().file_name().unwrap().to_string_lossy(),
                "args": args,
            }))
        })
        .with_script_approval(false);
    let mut ctx = SessionContext::new(vec![]);
    provider.before_run(&mut ctx).await.unwrap();
    let run = ctx
        .tools
        .iter()
        .find(|t| t.name == "run_skill_script")
        .unwrap()
        .clone();
    assert!(!run.requires_approval());
    let out = run.executor.clone().unwrap().invoke(call).await.unwrap();
    assert_eq!(
        out,
        json!({"skill": "data", "file": "process.py", "args": ["--fast"]})
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn inline_scripts_run_in_process() {
    use agent_framework_core::skills::{Skill, SkillScript};
    let skill = Skill::new("pw", "Passwords.").with_script(
        SkillScript::inline("generate", |args| async move {
            let length = args
                .as_ref()
                .and_then(|a| a.get("length"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(8);
            Ok(json!("x".repeat(length as usize)))
        })
        .with_description("Generate a password."),
    );
    let provider = SkillsProvider::new(vec![skill]);
    let mut ctx = SessionContext::new(vec![]);
    provider.before_run(&mut ctx).await.unwrap();
    assert!(ctx
        .instructions
        .as_deref()
        .unwrap()
        .contains("run_skill_script"));
    let tool = |name: &str| ctx.tools.iter().find(|t| t.name == name).unwrap().clone();
    let loaded = tool("load_skill")
        .executor
        .unwrap()
        .invoke(json!({"skill_name": "pw"}))
        .await
        .unwrap();
    assert!(loaded
        .as_str()
        .unwrap()
        .contains("<script name=\"generate\" description=\"Generate a password.\" />"));
    let run = tool("run_skill_script").executor.unwrap();
    let out = run
        .invoke(json!({"skill_name": "pw", "script_name": "generate", "args": {"length": 3}}))
        .await
        .unwrap();
    assert_eq!(out, "xxx");
    let missing = run
        .invoke(json!({"skill_name": "pw", "script_name": "nope"}))
        .await
        .unwrap();
    assert_eq!(missing, "Error: Script 'nope' not found in skill 'pw'.");

    // A provider whose skills have no scripts adds no script tool.
    let plain = SkillsProvider::new(vec![Skill::new("a", "b")]);
    let mut ctx = SessionContext::new(vec![]);
    plain.before_run(&mut ctx).await.unwrap();
    assert!(ctx.tools.iter().all(|t| t.name != "run_skill_script"));
}
