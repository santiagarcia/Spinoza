//! `list` subcommand: prints discovered packs and their capabilities.

use std::collections::BTreeMap;

use serde::Serialize;
use spinoza_core::{MethodRegistry, PackInfo};

/// JSON output shape for the `list` command.
#[derive(Debug, Serialize)]
pub struct ListJsonOutput {
    pub packs: Vec<PackInfo>,
    pub operator_factories: Vec<BTreeMap<String, String>>,
    pub preconditioner_factories: Vec<BTreeMap<String, String>>,
    pub solver_factories: Vec<BTreeMap<String, String>>,
}

/// Render the pack list as human-readable text.
pub fn render_list_text(registry: &MethodRegistry) -> String {
    let packs = registry.pack_info();
    let mut out = String::new();
    if packs.is_empty() {
        out.push_str("No packs discovered.\n");
        return out;
    }
    out.push_str("Discovered packs:\n");
    for pack in &packs {
        out.push_str(&format!("  {} v{}\n", pack.name, pack.version));
        for cap in &pack.capabilities {
            out.push_str(&format!("    - {cap}\n"));
        }
    }
    let builders = registry.builder_info();
    if !builders.is_empty() {
        out.push_str("\nRegistered builders:\n");
        for b in &builders {
            let name = b.get("name").map(|s| s.as_str()).unwrap_or("?");
            let pack = b.get("pack").map(|s| s.as_str()).unwrap_or("?");
            let eq = b.get("equation_type").map(|s| s.as_str()).unwrap_or("?");
            let dims = b.get("dimensions").map(|s| s.as_str()).unwrap_or("?");
            let backends = b.get("backends").map(|s| s.as_str()).unwrap_or("?");
            out.push_str(&format!(
                "  {name} (pack={pack}) equation={eq} dims={dims} backends={backends}\n"
            ));
        }
    }

    // Operator factories
    let ops = registry.operator_factory_info();
    if !ops.is_empty() {
        out.push_str("\nRegistered operator factories:\n");
        for f in &ops {
            let name = f.get("name").map(|s| s.as_str()).unwrap_or("?");
            let pack = f.get("pack").map(|s| s.as_str()).unwrap_or("?");
            let keys = f.get("keys").map(|s| s.as_str()).unwrap_or("?");
            out.push_str(&format!("  {name} (pack={pack}) keys=[{keys}]\n"));
        }
    }

    // Preconditioner factories
    let pcs = registry.preconditioner_factory_info();
    if !pcs.is_empty() {
        out.push_str("\nRegistered preconditioner factories:\n");
        for f in &pcs {
            let name = f.get("name").map(|s| s.as_str()).unwrap_or("?");
            let pack = f.get("pack").map(|s| s.as_str()).unwrap_or("?");
            let keys = f.get("keys").map(|s| s.as_str()).unwrap_or("?");
            out.push_str(&format!("  {name} (pack={pack}) keys=[{keys}]\n"));
        }
    }

    // Solver factories
    let solvers = registry.solver_factory_info();
    if !solvers.is_empty() {
        out.push_str("\nRegistered solver factories:\n");
        for f in &solvers {
            let name = f.get("name").map(|s| s.as_str()).unwrap_or("?");
            let pack = f.get("pack").map(|s| s.as_str()).unwrap_or("?");
            let keys = f.get("keys").map(|s| s.as_str()).unwrap_or("?");
            out.push_str(&format!("  {name} (pack={pack}) keys=[{keys}]\n"));
        }
    }

    out
}

/// Build the JSON output for the `list` command.
pub fn build_list_json(registry: &MethodRegistry) -> ListJsonOutput {
    ListJsonOutput {
        packs: registry.pack_info(),
        operator_factories: registry.operator_factory_info(),
        preconditioner_factories: registry.preconditioner_factory_info(),
        solver_factories: registry.solver_factory_info(),
    }
}
