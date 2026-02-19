//! spinoza-devtools: Developer utilities for the Spinoza framework.
//!
//! Provides a CLI for auditing case spec files against the current
//! capability registry.

mod audit;
mod list_packs;
mod new_pack;
mod solve_case;

pub use audit::{audit_spec, load_and_validate, AuditJsonOutput, AuditReport, AuditStatus};
pub use list_packs::{build_list_json, render_list_text, ListJsonOutput};
pub use new_pack::generate_new_pack;
pub use solve_case::{solve_case_file, SolveBackend, SolveOutput, SolvePreconditioner};
