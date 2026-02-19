//! spinoza-pack-linear-solvers: Solver algorithm pack for Spinoza.
//!
//! This pack owns all generic linear solver factory registrations.
//! Physics packs must **not** register solver factories — they request
//! solvers from the registry instead.
//!
//! Registered solver factories:
//!
//! | Solver | Key | Systems |
//! |--------|-----|---------|
//! | CG     | `{cg, spd=true, complex=false, single_field}` | SPD single-field |
//! | GMRES  | `{gmres, spd=false, complex=false, single_field}` | General single-field |
//!
//! The actual algorithm implementations live in `spinoza-solve`.
//! This pack creates thin factory wrappers that call into those routines.

use spinoza_core::{
    Capability, LinearOperator, MethodPack, MethodRegistry, Preconditioner,
    SolveResult, SolverFactory, SolverKey,
};
use spinoza_solve::{solve_cg, solve_gmres};

// ---------------------------------------------------------------------------
// Pack definition
// ---------------------------------------------------------------------------

/// The linear-solvers method pack.
///
/// Provides CG and GMRES solver factories for single-field systems.
#[derive(Default)]
pub struct LinearSolversPack;

static CAPABILITIES: &[Capability] = &[
    Capability::SolverCG,
    Capability::SolverGMRES,
];

impl MethodPack for LinearSolversPack {
    fn name(&self) -> &'static str {
        "linear_solvers"
    }
    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
    fn capabilities(&self) -> &'static [Capability] {
        CAPABILITIES
    }
    fn register(&self, registry: &mut MethodRegistry) {
        registry.register_solver_factory(Box::new(CgSolverFactory));
        registry.register_solver_factory(Box::new(GmresSolverFactory));
    }
}

spinoza_core::spinoza_register_pack!(LinearSolversPack);

// ---------------------------------------------------------------------------
// CG solver factory
// ---------------------------------------------------------------------------

/// Factory for the Conjugate Gradient solver.
///
/// Suitable for symmetric positive-definite single-field systems.
struct CgSolverFactory;

impl SolverFactory for CgSolverFactory {
    fn name(&self) -> &str {
        "CgSolver"
    }

    fn supported_keys(&self) -> Vec<SolverKey> {
        vec![SolverKey {
            solver_type: "cg".into(),
            spd: true,
            complex: false,
            block_structure: "single_field".into(),
        }]
    }

    fn capabilities_provided(&self) -> &[Capability] {
        &[Capability::SolverCG]
    }

    fn capabilities_required(&self) -> &[Capability] {
        &[]
    }

    fn solve(
        &self,
        _key: &SolverKey,
        operator: &dyn LinearOperator,
        rhs: &[f64],
        preconditioner: Option<&dyn Preconditioner>,
    ) -> Result<SolveResult, String> {
        let result = solve_cg(operator, rhs, 1e-10, 20_000, preconditioner)?;
        Ok(SolveResult {
            solution: result.solution,
            residual_norm: result.residual_norm,
            iterations: result.iterations,
        })
    }
}

// ---------------------------------------------------------------------------
// GMRES solver factory
// ---------------------------------------------------------------------------

/// Factory for the restarted GMRES solver.
///
/// Suitable for general (non-symmetric) single-field systems.
struct GmresSolverFactory;

impl SolverFactory for GmresSolverFactory {
    fn name(&self) -> &str {
        "GmresSolver"
    }

    fn supported_keys(&self) -> Vec<SolverKey> {
        vec![SolverKey {
            solver_type: "gmres".into(),
            spd: false,
            complex: false,
            block_structure: "single_field".into(),
        }]
    }

    fn capabilities_provided(&self) -> &[Capability] {
        &[Capability::SolverGMRES]
    }

    fn capabilities_required(&self) -> &[Capability] {
        &[]
    }

    fn solve(
        &self,
        _key: &SolverKey,
        operator: &dyn LinearOperator,
        rhs: &[f64],
        preconditioner: Option<&dyn Preconditioner>,
    ) -> Result<SolveResult, String> {
        let result = solve_gmres(operator, rhs, 1e-10, 20_000, 50, preconditioner)?;
        Ok(SolveResult {
            solution: result.solution,
            residual_norm: result.residual_norm,
            iterations: result.iterations,
        })
    }
}
