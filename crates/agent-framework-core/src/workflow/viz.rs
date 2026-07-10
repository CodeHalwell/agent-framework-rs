//! Workflow visualization: Mermaid `flowchart TD` and Graphviz DOT output.
//!
//! Rust equivalent of Python's `_viz.py`. Pure string generation with no extra
//! dependencies. The start node is highlighted, conditional edges are dashed,
//! switch cases are labeled, and fan-in barriers render as dedicated join nodes.

use super::edge::EdgeGroup;
use super::runner::Workflow;

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
    fn fan_in_nodes(&self) -> Vec<(String, Vec<String>, String)> {
        let mut result = Vec::new();
        for group in &self.workflow.shared.edge_groups {
            if let EdgeGroup::FanIn { sources, target } = group {
                let node_id = format!("fan_in_{target}_{}", result.len());
                result.push((node_id, sources.clone(), target.clone()));
            }
        }
        result
    }

    /// Render the workflow as a Mermaid `flowchart TD` string.
    pub fn to_mermaid(&self) -> String {
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
            if out.is_empty() || !out.chars().next().unwrap().is_ascii_alphabetic() {
                out = format!("n_{out}");
            }
            out
        }

        let start = self.workflow.start_executor_id();
        let mut lines: Vec<String> = vec!["flowchart TD".to_string()];

        // Nodes.
        lines.push(format!("  {}[\"{} (Start)\"]", san(start), start));
        for id in self.other_nodes() {
            lines.push(format!("  {}[\"{}\"]", san(&id), id));
        }

        // Fan-in join nodes and their edges.
        let fan_in = self.fan_in_nodes();
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
