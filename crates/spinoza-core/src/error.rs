//! Common error type for Spinoza.

use crate::Capability;
use std::collections::BTreeSet;

/// Errors that can occur during spec validation or capability auditing.
#[derive(Debug, thiserror::Error)]
pub enum SpinozaError {
    /// The case spec file could not be parsed as valid TOML.
    #[error("failed to parse case spec: {0}")]
    ParseError(String),

    /// The case spec was syntactically valid TOML but contains values that
    /// violate the schema (unknown equation type, bad dimension, etc.).
    #[error("validation error: {0}")]
    ValidationError(String),

    /// One or more required capabilities are not yet implemented.
    #[error("missing capabilities: {}", format_caps(.0))]
    MissingCapabilities(BTreeSet<Capability>),

    /// A linear operator related error.
    #[error("operator error: {0}")]
    OperatorError(String),
}

fn format_caps(caps: &BTreeSet<Capability>) -> String {
    caps.iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
