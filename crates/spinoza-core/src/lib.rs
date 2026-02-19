//! spinoza-core: Foundational types for the Spinoza multiphysics framework.
//!
//! This crate defines the capability model, case specification schema,
//! compiled plan, plugin infrastructure, and common error types shared
//! across all Spinoza crates.

mod block;
mod capability;
pub mod conformance;
mod error;
pub mod method_pack;
pub mod method_registry;
mod operator;
mod preconditioner;
mod spec;

pub use block::BlockLayout;
pub use capability::{Capability, CapabilityRegistry};
pub use error::SpinozaError;
pub use method_pack::{MethodPack, PackEntry};
pub use method_registry::{
    AmbiguousBuilderError, AmbiguousComponentError, BuilderCandidate, BuilderLookup,
    BuilderSelection, BuiltOperator, BuiltProblem, ComponentCandidate, ComponentLookup,
    ComponentSelection, FactoryKey, MethodRegistry, OperatorFactory, OperatorKey, PackInfo,
    PreconditionerFactory, PreconditionerKey, ProblemBuilder, SolveResult, SolverFactory,
    SolverKey,
};
pub use operator::LinearOperator;
pub use preconditioner::Preconditioner;
pub use spec::CompiledPlan;
pub use spec::{
    BcSpec, CaseSpec, EquationSpec, FieldKind, FieldSpec, MeshSpec, ProblemSpec, SolverSpec, Space,
};

// Re-export inventory so packs can use the macro without depending on it directly.
pub use inventory;
