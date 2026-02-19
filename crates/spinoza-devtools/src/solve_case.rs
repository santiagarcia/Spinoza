use std::path::Path;

use serde::Serialize;
use spinoza_core::{BuilderLookup, ComponentLookup, MethodRegistry, SolverKey, SpinozaError,
    infer_block_structure};

use crate::audit::load_and_validate;

const DEFAULT_NX: usize = 8;
const DEFAULT_NY: usize = 8;
const DEFAULT_NZ: usize = 8;

#[derive(Debug, Serialize)]
pub struct SolveOutput {
    pub problem_name: String,
    pub selected_pack: String,
    pub selected_builder: String,
    pub backend: String,
    pub preconditioner: String,
    pub dimension: u8,
    pub nx: usize,
    pub ny: usize,
    pub nz: Option<usize>,
    pub field_names: Vec<String>,
    pub field_sizes: Vec<usize>,
    pub selected_operator: Option<String>,
    pub selected_preconditioner: Option<String>,
    pub selected_solver: String,
    pub solution_vector: Vec<f64>,
    pub residual_norm: f64,
    pub iteration_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum SolveBackend {
    Assembled,
    MatrixFree,
}

impl SolveBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            SolveBackend::Assembled => "assembled",
            SolveBackend::MatrixFree => "matrixfree",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SolvePreconditioner {
    Jacobi,
    Mg,
    BlockJacobi,
}

impl SolvePreconditioner {
    pub fn as_str(self) -> &'static str {
        match self {
            SolvePreconditioner::Jacobi => "jacobi",
            SolvePreconditioner::Mg => "mg",
            SolvePreconditioner::BlockJacobi => "block_jacobi",
        }
    }
}

/// Build and solve a case using the method registry.
///
/// The registry discovers all linked method packs via `inventory` and finds
/// a suitable builder for the equation type, dimension, and backend.
pub fn solve_case_file(
    path: &Path,
    backend: SolveBackend,
    precond: SolvePreconditioner,
) -> Result<SolveOutput, SpinozaError> {
    // Ensure pack crates are linked (prevents dead-code elimination).
    // The `use` statement alone is insufficient; we reference a symbol.
    let _ = spinoza_pack_poisson::PoissonPack;
    let _ = spinoza_pack_block2x2::Block2x2Pack;
    let _ = spinoza_pack_elasticity::ElasticityPack;
    let _ = spinoza_pack_linear_solvers::LinearSolversPack;

    let spec = load_and_validate(path)?;
    let dim = spec.mesh.dimension;
    let backend_str = backend.as_str();
    let precond_str = precond.as_str();

    // Build the method registry from all discovered packs.
    let registry = MethodRegistry::from_packs();

    let (builder, selection) =
        match registry.find_builder_checked(&spec.equation.eq_type, dim, backend_str) {
            BuilderLookup::Found { builder, selection } => (builder, selection),
            BuilderLookup::NotFound => {
                return Err(SpinozaError::ValidationError(format!(
                    "no builder found for equation='{}', dim={}, backend='{}'",
                    spec.equation.eq_type, dim, backend_str,
                )));
            }
            BuilderLookup::Ambiguous(err) => {
                return Err(SpinozaError::ValidationError(err.to_string()));
            }
        };

    let nz = if dim == 3 { DEFAULT_NZ } else { 1 };
    let built = builder
        .build(
            &spec,
            backend_str,
            precond_str,
            DEFAULT_NX,
            DEFAULT_NY,
            nz,
            &registry,
        )
        .map_err(SpinozaError::ValidationError)?;

    // Try to find a solver factory from the registry; fall back to solve_cg.
    let solver_type = spec.solver.linear.as_str();
    let block_str = infer_block_structure(&spec.equation.eq_type);
    let is_spd = solver_type != "gmres";
    let solver_key = SolverKey {
        solver_type: solver_type.to_string(),
        spd: is_spd,
        complex: false,
        block_structure: block_str,
    };

    let (solution, residual_norm, iterations, selected_solver) =
        match registry.find_solver(&solver_key) {
            ComponentLookup::Found {
                component,
                selection,
            } => {
                let res = component
                    .solve(
                        &solver_key,
                        built.operator.as_ref(),
                        &built.rhs,
                        built.preconditioner.as_deref(),
                    )
                    .map_err(SpinozaError::ValidationError)?;
                (
                    res.solution,
                    res.residual_norm,
                    res.iterations,
                    format!(
                        "{}::{}",
                        selection.pack_name, selection.component_name
                    ),
                )
            }
            ComponentLookup::NotFound => {
                return Err(SpinozaError::ValidationError(format!(
                    "COMPONENT_NOT_FOUND: no solver factory registered for key ({}). \
                     Ensure spinoza-pack-linear-solvers is linked.",
                    solver_key
                )));
            }
            ComponentLookup::Ambiguous(err) => {
                return Err(SpinozaError::ValidationError(err.to_string()));
            }
        };

    Ok(SolveOutput {
        problem_name: spec.problem.name,
        selected_pack: selection.pack_name,
        selected_builder: selection.builder_name,
        backend: backend_str.to_string(),
        preconditioner: precond_str.to_string(),
        dimension: dim,
        nx: DEFAULT_NX,
        ny: DEFAULT_NY,
        nz: if dim == 3 { Some(DEFAULT_NZ) } else { None },
        field_names: built.field_names,
        field_sizes: built.field_sizes,
        selected_operator: None,
        selected_preconditioner: None,
        selected_solver,
        solution_vector: solution,
        residual_norm,
        iteration_count: iterations,
    })
}
