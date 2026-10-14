//! Workflow visualization: Mermaid `flowchart TD` and Graphviz DOT output.
//!
//! Rust equivalent of Python's `_viz.py`. Pure string generation with no extra
//! dependencies. The start node is highlighted, conditional edges are dashed,
//! switch cases are labeled, and fan-in barriers render as dedicated join nodes.

use std::collections::{HashMap, HashSet};

use super::edge::EdgeGroup;
use super::runner::Workflow;

/// Map an id onto Mermaid's node-id alphabet: `[0-9A-Za-z_]`, starting with
/// a letter. Lossy — `step-a` and `step_a` both become `step_a` — so it is
/// only ever applied through [`WorkflowViz::mermaid_ids`], which resolves the
/// collisions it creates.
fn san(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if !out.starts_with(|c: char| c.is_ascii_alphabetic()) {
        out = format!("n_{out}");
    }
    out
}

/// The first `{base}_{n}` (from `n = 2`) that is in neither set.
fn first_free(base: &str, used: &HashSet<String>, reserved: &HashSet<String>) -> String {
    (2..)
        .map(|n| format!("{base}_{n}"))
        .find(|c| !used.contains(c) && !reserved.contains(c))
        .expect("an unbounded range always yields a free suffix")
}

/// A visualization helper bound to a built [`Workflow`].
pub struct WorkflowViz<'a> {
    workflow: &'a Workflow,
}

impl<'a> WorkflowViz<'a> {
    /// Create a visualizer for `workflow`.
    pub fn new(workflow: &'a Workflow) -> Self {
        Self { workflow }
    }

    /// Executor ids other than the start, sorted for deterministic output.
    fn other_nodes(&self) -> Vec<String> {
        let start = self.workflow.start_executor_id();
        let mut ids: Vec<String> = self
            .workflow
            .shared
            .executors
            .keys()
            .filter(|id| id.as_str() != start)
            .cloned()
            .collect();
        ids.sort();
        ids
    }

    /// Fan-in descriptors: `(node_id, sources, target)`.
    ///
    /// `node_id` is `fan_in_{target}_{n}`, suffixed further when an executor
    /// already uses that id — otherwise the join node and the executor would
    /// render as one node in both Mermaid and DOT (upstream #9082).
    fn fan_in_nodes(&self) -> Vec<(String, Vec<String>, String)> {
        let executors = &self.workflow.shared.executors;
        let mut taken: HashSet<String> = HashSet::new();
        let mut result = Vec::new();
        for group in &self.workflow.shared.edge_groups {
            if let EdgeGroup::FanIn { sources, target } = group {
                let base = format!("fan_in_{target}_{}", result.len());
                let node_id = if executors.contains_key(&base) || taken.contains(&base) {
                    (2..)
                        .map(|n| format!("{base}_{n}"))
                        .find(|c| !executors.contains_key(c) && !taken.contains(c))
                        .expect("an unbounded range always yields a free suffix")
                } else {
                    base
                };
                taken.insert(node_id.clone());
                result.push((node_id, sources.clone(), target.clone()));
            }
        }
        result
    }

    /// Mermaid node ids for every executor and fan-in node, keyed by raw id.
    ///
    /// [`san`] is lossy, so mapping each id independently let distinct
    /// executors (`step-a` / `step_a`, `1step` / `n_1step`, any two ids in a
    /// non-Latin script) collapse into one node — edges between them drew as
    /// a self-loop (upstream #9082). Aliases are therefore assigned for the
    /// whole graph at once: ids that are already valid Mermaid ids keep their
    /// name, and the rest take their sanitized form, or the first
    /// `_2`, `_3`, … suffix that is neither used nor some other id's
    /// sanitized form. Raw executor and fan-in ids are disjoint (see
    /// [`Self::fan_in_nodes`]), so one map covers both. Iteration is sorted,
    /// so output is deterministic.
    fn mermaid_ids(&self, fan_in: &[(String, Vec<String>, String)]) -> HashMap<String, String> {
        let mut executors: Vec<&String> = self.workflow.shared.executors.keys().collect();
        executors.sort();
        let ids: Vec<&String> = executors
            .into_iter()
            .chain(fan_in.iter().map(|(id, _, _)| id))
            .collect();

        let reserved: HashSet<String> = ids.iter().map(|id| san(id)).collect();
        let mut used: HashSet<String> = HashSet::new();
        let mut aliases: HashMap<String, String> = HashMap::new();
        for id in ids.iter().filter(|id| san(id) == ***id) {
            used.insert((*id).clone());
            aliases.insert((*id).clone(), (*id).clone());
        }
        for id in ids.iter().filter(|id| san(id) != ***id) {
            let candidate = san(id);
            let alias = if used.contains(&candidate) {
                first_free(&candidate, &used, &reserved)
            } else {
                candidate
            };
            used.insert(alias.clone());
            aliases.insert((*id).clone(), alias);
        }
        aliases
    }

    /// Render the workflow as a Mermaid `flowchart TD` string.
    pub fn to_mermaid(&self) -> String {
        let start = self.workflow.start_executor_id();
        let fan_in = self.fan_in_nodes();
        let aliases = self.mermaid_ids(&fan_in);
        // Every id rendered is an executor or a fan-in node, both in the map;
        // the fallback only covers an edge naming an unregistered executor,
        // which `WorkflowBuilder::build` rejects.
        let san = |id: &str| aliases.get(id).cloned().unwrap_or_else(|| san(id));
        let mut lines: Vec<String> = vec!["flowchart TD".to_string()];

        // Nodes.
        lines.push(format!("  {}[\"{} (Start)\"]", san(start), start));
        for id in self.other_nodes() {
            lines.push(format!("  {}[\"{}\"]", san(&id), id));
        }

        // Fan-in join nodes and their edges.
        for (node_id, _, _) in &fan_in {
            lines.push(format!("  {}((fan-in))", san(node_id)));
        }
        for (node_id, sources, target) in &fan_in {
            for src in sources {
                lines.push(format!("  {} --> {}", san(src), san(node_id)));
            }
            lines.push(format!("  {} --> {}", san(node_id), san(target)));
        }

        // Normal edges (Single + FanOut), labeled by kind.
        for group in &self.workflow.shared.edge_groups {
            match group {
                EdgeGroup::Single {
                    source,
                    target,
                    condition,
                } => {
                    if condition.is_some() {
                        lines.push(format!(
                            "  {} -. conditional .-> {}",
                            san(source),
                            san(target)
                        ));
                    } else {
                        lines.push(format!("  {} --> {}", san(source), san(target)));
                    }
                }
                EdgeGroup::FanOut {
                    source,
                    targets,
                    case_labels,
                    ..
                } => {
                    for (i, target) in targets.iter().enumerate() {
                        match case_labels.as_ref().and_then(|l| l.get(i)) {
                            Some(label) => lines.push(format!(
                                "  {} -- \"{}\" --> {}",
                                san(source),
                                label,
                                san(target)
                            )),
                            None => lines.push(format!("  {} --> {}", san(source), san(target))),
                        }
                    }
                }
                EdgeGroup::FanIn { .. } => {}
            }
        }

        lines.join("\n")
    }

    /// Render the workflow as a Graphviz DOT digraph string.
    pub fn to_dot(&self) -> String {
        let start = self.workflow.start_executor_id();
        let mut lines: Vec<String> = vec![
            "digraph Workflow {".to_string(),
            "  rankdir=TD;".to_string(),
            "  node [shape=box, style=filled, fillcolor=lightblue];".to_string(),
            "  edge [color=black, arrowhead=vee];".to_string(),
            String::new(),
        ];

        // Nodes.
        lines.push(format!(
            "  \"{start}\" [fillcolor=lightgreen, label=\"{start}\\n(Start)\"];"
        ));
        for id in self.other_nodes() {
            lines.push(format!("  \"{id}\" [label=\"{id}\"];"));
        }

        // Fan-in join nodes and edges.
        let fan_in = self.fan_in_nodes();
        for (node_id, _, _) in &fan_in {
            lines.push(format!(
                "  \"{node_id}\" [shape=ellipse, fillcolor=lightgoldenrod, label=\"fan-in\"];"
            ));
        }
        for (node_id, sources, target) in &fan_in {
            for src in sources {
                lines.push(format!("  \"{src}\" -> \"{node_id}\";"));
            }
            lines.push(format!("  \"{node_id}\" -> \"{target}\";"));
        }

        // Normal edges.
        for group in &self.workflow.shared.edge_groups {
            match group {
                EdgeGroup::Single {
                    source,
                    target,
                    condition,
                } => {
                    let attr = if condition.is_some() {
                        " [style=dashed, label=\"conditional\"]"
                    } else {
                        ""
                    };
                    lines.push(format!("  \"{source}\" -> \"{target}\"{attr};"));
                }
                EdgeGroup::FanOut {
                    source,
                    targets,
                    case_labels,
                    ..
                } => {
                    for (i, target) in targets.iter().enumerate() {
                        match case_labels.as_ref().and_then(|l| l.get(i)) {
                            Some(label) => lines.push(format!(
                                "  \"{source}\" -> \"{target}\" [label=\"{label}\"];"
                            )),
                            None => lines.push(format!("  \"{source}\" -> \"{target}\";")),
                        }
                    }
                }
                EdgeGroup::FanIn { .. } => {}
            }
        }

        lines.push("}".to_string());
        lines.join("\n")
    }
}
