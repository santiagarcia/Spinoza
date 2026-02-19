//! spinoza-disc: Minimal discretization support for Poisson in 2D.

mod matrix;
mod matrixfree;
mod mesh;
mod poisson;

pub use matrix::{CsrMatrix, SparseMatrixBuilder};
pub use matrixfree::{
    apply_dirichlet_rhs_correction_2d, apply_dirichlet_rhs_correction_3d, DirichletConstraints,
    PoissonLaplacianMF2D, PoissonLaplacianMF3D,
};
pub use mesh::{StructuredHexMesh3D, StructuredQuadMesh2D};
pub use poisson::{
    apply_constant_dirichlet, apply_constant_dirichlet_on_boundary,
    apply_constant_dirichlet_on_boundary_3d, apply_dirichlet_values, assemble_poisson_system,
    assemble_poisson_system_3d, assemble_poisson_system_3d_with_source,
    assemble_poisson_system_with_source,
};
