//! Typed case specification schema and compiled plan.
//!
//! A [`CaseSpec`] is the direct deserialization target for the TOML case file.
//! After validation it can be compiled into a [`CompiledPlan`] that lists the
//! concrete [`Capability`] values the simulation requires.

use std::collections::BTreeSet;

use serde::Deserialize;

use crate::capability::Capability;
use crate::error::SpinozaError;

// ---------------------------------------------------------------------------
// Raw TOML schema types
// ---------------------------------------------------------------------------

/// Top level case specification.
#[derive(Debug, Deserialize)]
pub struct CaseSpec {
    pub problem: ProblemSpec,
    pub mesh: MeshSpec,
    pub fields: Vec<FieldSpec>,
    pub equation: EquationSpec,
    pub bcs: Vec<BcSpec>,
    pub solver: SolverSpec,
}

/// Metadata about the problem.
#[derive(Debug, Deserialize)]
pub struct ProblemSpec {
    pub name: String,
}

/// Mesh description.
#[derive(Debug, Deserialize)]
pub struct MeshSpec {
    pub dimension: u8,
}

/// A field (unknown) to be solved for.
#[derive(Debug, Deserialize)]
pub struct FieldSpec {
    pub name: String,
    pub kind: FieldKind,
    pub space: Space,
    pub complex: bool,
}

/// Whether the field is scalar or vector valued.
#[derive(Debug, Deserialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum FieldKind {
    Scalar,
    Vector,
}

/// Finite element function space family.
#[derive(Debug, Deserialize, PartialEq, Eq, Clone, Copy)]
pub enum Space {
    H1,
}

/// The governing equation.
#[derive(Debug, Deserialize)]
pub struct EquationSpec {
    #[serde(rename = "type")]
    pub eq_type: String,
    /// Optional coupling parameter alpha for coupled systems.
    pub alpha: Option<f64>,
    /// Optional coupling parameter beta for coupled systems.
    pub beta: Option<f64>,
}

/// A single boundary condition entry.
#[derive(Debug, Deserialize)]
pub struct BcSpec {
    #[serde(rename = "type")]
    pub bc_type: String,
    pub field: String,
    pub value: f64,
}

/// Solver configuration.
#[derive(Debug, Deserialize)]
pub struct SolverSpec {
    pub linear: String,
    pub precond: String,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

impl CaseSpec {
    /// Validate schema constraints that go beyond TOML deserialization.
    ///
    /// Returns `Ok(())` when the spec is well formed, or a
    /// [`SpinozaError::ValidationError`] describing the first problem found.
    pub fn validate(&self) -> Result<(), SpinozaError> {
        // Dimension must be 2 or 3.
        if self.mesh.dimension != 2 && self.mesh.dimension != 3 {
            return Err(SpinozaError::ValidationError(format!(
                "mesh.dimension must be 2 or 3, got {}",
                self.mesh.dimension
            )));
        }

        // Equation type allow list.
        let allowed_equations = ["poisson", "coupled_elliptic_2x2"];
        if !allowed_equations.contains(&self.equation.eq_type.as_str()) {
            return Err(SpinozaError::ValidationError(format!(
                "equation.type '{}' is not supported (allowed: {})",
                self.equation.eq_type,
                allowed_equations.join(", ")
            )));
        }

        // BC type allow list.
        for bc in &self.bcs {
            if bc.bc_type != "dirichlet" {
                return Err(SpinozaError::ValidationError(format!(
                    "bc type '{}' is not supported (allowed: dirichlet)",
                    bc.bc_type
                )));
            }
        }

        // Solver allow list.
        let allowed_solvers = ["cg", "block_pcg"];
        if !allowed_solvers.contains(&self.solver.linear.as_str()) {
            return Err(SpinozaError::ValidationError(format!(
                "solver.linear '{}' is not supported (allowed: {})",
                self.solver.linear,
                allowed_solvers.join(", ")
            )));
        }
        let allowed_preconds = ["jacobi", "block_jacobi", "mg"];
        if !allowed_preconds.contains(&self.solver.precond.as_str()) {
            return Err(SpinozaError::ValidationError(format!(
                "solver.precond '{}' is not supported (allowed: {})",
                self.solver.precond,
                allowed_preconds.join(", ")
            )));
        }

        // Fields must reference only known spaces (enforced by enum, but we
        // double check the list is non empty).
        if self.fields.is_empty() {
            return Err(SpinozaError::ValidationError(
                "at least one field is required".to_string(),
            ));
        }

        // Every BC must reference a declared field.
        let field_names: BTreeSet<&str> = self.fields.iter().map(|f| f.name.as_str()).collect();
        for bc in &self.bcs {
            if !field_names.contains(bc.field.as_str()) {
                return Err(SpinozaError::ValidationError(format!(
                    "bc references unknown field '{}'",
                    bc.field
                )));
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CompiledPlan
// ---------------------------------------------------------------------------

/// The result of compiling a validated [`CaseSpec`] into a set of required
/// capabilities.
#[derive(Debug)]
pub struct CompiledPlan {
    /// Human readable problem name taken from the spec.
    pub problem_name: String,
    /// The set of capabilities the simulation case requires.
    pub required: BTreeSet<Capability>,
}

impl CompiledPlan {
    /// Compile a validated `CaseSpec` into a plan.
    ///
    /// This function assumes [`CaseSpec::validate`] has already been called.
    pub fn from_spec(spec: &CaseSpec) -> Self {
        let mut required = BTreeSet::new();

        // Derive space capabilities from fields.
        for field in &spec.fields {
            match (field.space, field.kind) {
                (Space::H1, FieldKind::Scalar) => {
                    required.insert(Capability::SpaceH1Scalar);
                }
                (Space::H1, FieldKind::Vector) => {
                    required.insert(Capability::SpaceH1Vector);
                }
            }
        }

        // Derive operator capabilities from equation type.
        if spec.equation.eq_type.as_str() == "poisson" {
            required.insert(Capability::OperatorLaplacian);
        }
        if spec.equation.eq_type.as_str() == "coupled_elliptic_2x2" {
            required.insert(Capability::OperatorLaplacian);
            required.insert(Capability::EquationCoupledElliptic2x2);
            required.insert(Capability::BlockOperator2x2);
        }

        // Derive BC capabilities.
        for bc in &spec.bcs {
            if bc.bc_type.as_str() == "dirichlet" {
                required.insert(Capability::BcDirichlet);
            }
        }

        // Derive solver capabilities.
        if spec.solver.linear == "cg" {
            required.insert(Capability::SolverCG);
        }
        if spec.solver.linear == "block_pcg" {
            required.insert(Capability::SolverBlockPCG);
        }
        if spec.solver.precond == "jacobi" {
            required.insert(Capability::PrecondJacobi);
        }
        if spec.solver.precond == "block_jacobi" {
            required.insert(Capability::PrecondBlockJacobi);
        }
        if spec.solver.precond == "mg" {
            required.insert(Capability::PrecondMG);
        }

        CompiledPlan {
            problem_name: spec.problem.name.clone(),
            required,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_toml() -> &'static str {
        r#"
[problem]
name = "unit test"

[mesh]
dimension = 2

[[fields]]
name = "u"
kind = "scalar"
space = "H1"
complex = false

[equation]
type = "poisson"

[[bcs]]
type = "dirichlet"
field = "u"
value = 0.0

[solver]
linear = "cg"
precond = "jacobi"
"#
    }

    #[test]
    fn parse_minimal_spec() {
        let spec: CaseSpec = toml::from_str(minimal_toml()).expect("parse failed");
        assert_eq!(spec.problem.name, "unit test");
        assert_eq!(spec.mesh.dimension, 2);
        assert_eq!(spec.fields.len(), 1);
    }

    #[test]
    fn validate_ok() {
        let spec: CaseSpec = toml::from_str(minimal_toml()).unwrap();
        spec.validate().expect("validation should pass");
    }

    #[test]
    fn validate_bad_dimension() {
        let bad = minimal_toml().replace("dimension = 2", "dimension = 4");
        let spec: CaseSpec = toml::from_str(&bad).unwrap();
        let err = spec.validate().unwrap_err();
        assert!(err.to_string().contains("dimension must be 2 or 3"));
    }

    #[test]
    fn validate_bad_equation() {
        let bad = minimal_toml().replace("type = \"poisson\"", "type = \"navier_stokes\"");
        let spec: CaseSpec = toml::from_str(&bad).unwrap();
        let err = spec.validate().unwrap_err();
        assert!(err.to_string().contains("not supported"));
    }

    #[test]
    fn compiled_plan_from_spec() {
        let spec: CaseSpec = toml::from_str(minimal_toml()).unwrap();
        spec.validate().unwrap();
        let plan = CompiledPlan::from_spec(&spec);
        assert!(plan.required.contains(&Capability::SpaceH1Scalar));
        assert!(plan.required.contains(&Capability::OperatorLaplacian));
        assert!(plan.required.contains(&Capability::BcDirichlet));
        assert!(plan.required.contains(&Capability::SolverCG));
        assert!(plan.required.contains(&Capability::PrecondJacobi));
    }
}
