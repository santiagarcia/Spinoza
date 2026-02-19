//! Audit logic: parse, validate, compile, and compare against the registry.

use std::collections::BTreeSet;
use std::path::Path;

use serde::Serialize;
use spinoza_core::{Capability, CapabilityRegistry, CaseSpec, CompiledPlan, SpinozaError};

/// A structured report produced by the audit workflow.
#[derive(Debug)]
pub struct AuditReport {
    pub problem_name: String,
    pub required: BTreeSet<Capability>,
    pub missing: BTreeSet<Capability>,
}

impl AuditReport {
    /// Returns `true` when every required capability is available.
    pub fn is_ok(&self) -> bool {
        self.missing.is_empty()
    }

    fn sorted_capability_names(caps: &BTreeSet<Capability>) -> Vec<String> {
        let mut items: Vec<String> = caps.iter().map(|cap| cap.to_string()).collect();
        items.sort();
        items
    }

    pub fn sorted_required_capability_names(&self) -> Vec<String> {
        Self::sorted_capability_names(&self.required)
    }

    pub fn sorted_missing_capability_names(&self) -> Vec<String> {
        Self::sorted_capability_names(&self.missing)
    }

    /// Render the report as a human readable string.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Problem: {}\n", self.problem_name));

        out.push_str("Required capabilities:\n");
        for cap in self.sorted_required_capability_names() {
            out.push_str(&format!("  {cap}\n"));
        }

        if self.missing.is_empty() {
            out.push_str("All capabilities available.\n");
        } else {
            out.push_str("Missing capabilities:\n");
            for cap in self.sorted_missing_capability_names() {
                out.push_str(&format!("  {cap}\n"));
            }
        }
        out
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuditStatus {
    Ok,
    MissingCapabilities,
    ValidationError,
}

#[derive(Debug, Serialize)]
pub struct AuditJsonOutput {
    pub problem_name: String,
    pub case_path: String,
    pub required_capabilities: Vec<String>,
    pub missing_capabilities: Vec<String>,
    pub status: AuditStatus,
    pub exit_code: i32,
    pub validation_errors: Vec<String>,
}

impl AuditJsonOutput {
    pub fn from_success(path: &Path, report: &AuditReport) -> Self {
        let status = if report.is_ok() {
            AuditStatus::Ok
        } else {
            AuditStatus::MissingCapabilities
        };
        let exit_code = if report.is_ok() { 0 } else { 1 };

        Self {
            problem_name: report.problem_name.clone(),
            case_path: path.display().to_string(),
            required_capabilities: report.sorted_required_capability_names(),
            missing_capabilities: report.sorted_missing_capability_names(),
            status,
            exit_code,
            validation_errors: Vec::new(),
        }
    }

    pub fn from_validation_error(path: &Path, message: String) -> Self {
        Self {
            problem_name: String::new(),
            case_path: path.display().to_string(),
            required_capabilities: Vec::new(),
            missing_capabilities: Vec::new(),
            status: AuditStatus::ValidationError,
            exit_code: 2,
            validation_errors: vec![message],
        }
    }
}

/// Load a TOML case file from disk, deserialize it, and validate it.
///
/// Returns the parsed and validated [`CaseSpec`] or a [`SpinozaError`].
pub fn load_and_validate(path: &Path) -> Result<CaseSpec, SpinozaError> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| SpinozaError::ParseError(format!("cannot read file: {e}")))?;
    let spec: CaseSpec =
        toml::from_str(&contents).map_err(|e| SpinozaError::ParseError(e.to_string()))?;
    spec.validate()?;
    Ok(spec)
}

/// Run a full audit of the case spec at `path` against the given registry.
///
/// Returns an [`AuditReport`] on success, or a [`SpinozaError`] if the spec
/// is unreadable or invalid.
pub fn audit_spec(path: &Path, registry: &CapabilityRegistry) -> Result<AuditReport, SpinozaError> {
    let spec = load_and_validate(path)?;
    let plan = CompiledPlan::from_spec(&spec);
    let missing = registry.missing(&plan.required);

    Ok(AuditReport {
        problem_name: plan.problem_name,
        required: plan.required,
        missing,
    })
}
