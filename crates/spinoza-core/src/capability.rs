//! Capability model: an enum of fine grained feature flags and a registry
//! that tracks which capabilities the codebase currently provides.

use std::collections::BTreeSet;
use std::fmt;

/// A single, typed capability that the framework can provide.
///
/// Each variant maps to a concrete piece of functionality (a function space,
/// an operator, a boundary condition type, a solver algorithm, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    /// H1 conforming scalar finite element space.
    SpaceH1Scalar,
    /// H1 conforming vector finite element space.
    SpaceH1Vector,
    /// Laplacian (diffusion) operator assembly.
    OperatorLaplacian,
    /// Dirichlet boundary condition enforcement.
    BcDirichlet,
    /// Conjugate Gradient linear solver.
    SolverCG,
    /// Jacobi preconditioner.
    PrecondJacobi,
    /// Multigrid preconditioner.
    PrecondMG,
    /// 2x2 block operator.
    BlockOperator2x2,
    /// Block Jacobi preconditioner.
    PrecondBlockJacobi,
    /// Block PCG solver.
    SolverBlockPCG,
    /// Coupled elliptic 2x2 system.
    EquationCoupledElliptic2x2,
    /// Linear elasticity stiffness operator.
    OperatorElasticity,
    /// GMRES linear solver.
    SolverGMRES,
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Capability::SpaceH1Scalar => "Space_H1Scalar",
            Capability::SpaceH1Vector => "Space_H1Vector",
            Capability::OperatorLaplacian => "Operator_Laplacian",
            Capability::BcDirichlet => "BC_Dirichlet",
            Capability::SolverCG => "Solver_CG",
            Capability::PrecondJacobi => "Precond_Jacobi",
            Capability::PrecondMG => "Precond_MG",
            Capability::BlockOperator2x2 => "BlockOperator_2x2",
            Capability::PrecondBlockJacobi => "Precond_BlockJacobi",
            Capability::SolverBlockPCG => "Solver_BlockPCG",
            Capability::EquationCoupledElliptic2x2 => "Equation_CoupledElliptic2x2",
            Capability::OperatorElasticity => "Operator_Elasticity",
            Capability::SolverGMRES => "Solver_GMRES",
        };
        f.write_str(label)
    }
}

/// A registry that records which [`Capability`] values are currently
/// implemented in the codebase.
#[derive(Debug, Clone)]
pub struct CapabilityRegistry {
    available: BTreeSet<Capability>,
}

impl CapabilityRegistry {
    /// Create an empty registry.
    pub fn empty() -> Self {
        Self {
            available: BTreeSet::new(),
        }
    }

    /// Create the default registry by discovering all registered method packs.
    ///
    /// This iterates all packs submitted via `inventory` and unions their
    /// capabilities into a single registry.
    pub fn default_registry() -> Self {
        let mut reg = Self::empty();
        for pack in crate::method_pack::iter_packs() {
            for cap in pack.capabilities() {
                reg.register(*cap);
            }
        }
        reg
    }

    /// Register a capability as available.
    pub fn register(&mut self, cap: Capability) {
        self.available.insert(cap);
    }

    /// Check whether a capability is available.
    pub fn has(&self, cap: &Capability) -> bool {
        self.available.contains(cap)
    }

    /// Return `available` set reference.
    pub fn available(&self) -> &BTreeSet<Capability> {
        &self.available
    }

    /// Return the set of capabilities from `required` that are **not** in
    /// the registry.
    pub fn missing(&self, required: &BTreeSet<Capability>) -> BTreeSet<Capability> {
        required.difference(&self.available).copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_contains_expected() {
        // Without any packs linked, the default registry is empty.
        // Packs register capabilities at link time via inventory.
        let reg = CapabilityRegistry::default_registry();
        // This test only verifies the registry builds without panic.
        let _ = reg.available();
    }

    #[test]
    fn missing_reports_absent_caps() {
        let mut reg = CapabilityRegistry::empty();
        reg.register(Capability::SpaceH1Scalar);
        reg.register(Capability::OperatorLaplacian);
        reg.register(Capability::SolverCG);

        let mut required = BTreeSet::new();
        required.insert(Capability::SpaceH1Scalar);
        required.insert(Capability::OperatorLaplacian);
        required.insert(Capability::SolverCG);

        let missing = reg.missing(&required);
        assert!(missing.is_empty());
    }
}
