//! spinoza-pack-block2x2: Method pack for 2×2 block systems and
//! coupled elliptic PDEs.
//!
//! Registers:
//! - BlockOperator_2x2
//! - Precond_BlockJacobi
//! - Solver_BlockPCG
//! - Equation_CoupledElliptic2x2

mod block_operator;
mod block_precond;
mod coupled_builder;

pub use block_operator::BlockOperator2x2;
pub use block_precond::BlockJacobiPreconditioner;

use spinoza_core::{
    Capability, LinearOperator, MethodPack, MethodRegistry, Preconditioner, SolveResult,
    SolverFactory, SolverKey,
};
use spinoza_solve::solve_cg;

// ---------------------------------------------------------------------------
// Pack definition
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct Block2x2Pack;

static CAPABILITIES: &[Capability] = &[
    Capability::BlockOperator2x2,
    Capability::PrecondBlockJacobi,
    Capability::SolverBlockPCG,
    Capability::EquationCoupledElliptic2x2,
];

impl MethodPack for Block2x2Pack {
    fn name(&self) -> &'static str {
        "block2x2"
    }
    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
    fn capabilities(&self) -> &'static [Capability] {
        CAPABILITIES
    }
    fn register(&self, registry: &mut MethodRegistry) {
        registry.register_builder(Box::new(coupled_builder::CoupledEllipticBuilder));
        registry.register_solver_factory(Box::new(BlockPcgSolverFactory));
    }
}

spinoza_core::spinoza_register_pack!(Block2x2Pack);

// ---------------------------------------------------------------------------
// Solver factory for block PCG
// ---------------------------------------------------------------------------

struct BlockPcgSolverFactory;

impl SolverFactory for BlockPcgSolverFactory {
    fn name(&self) -> &str {
        "BlockPcgSolver"
    }

    fn supported_keys(&self) -> Vec<SolverKey> {
        vec![SolverKey {
            solver_type: "block_pcg".into(),
            spd: true,
            complex: false,
            block_structure: "block2x2".into(),
        }]
    }

    fn capabilities_provided(&self) -> &[Capability] {
        &[Capability::SolverBlockPCG]
    }

    fn capabilities_required(&self) -> &[Capability] {
        &[Capability::BlockOperator2x2]
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
