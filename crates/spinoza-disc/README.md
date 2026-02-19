# spinoza-disc

Structured mesh generation and finite element assembly for Spinoza.

## What this crate provides

- **`StructuredQuadMesh2D`** — unit-square mesh of bilinear quads with boundary node queries.
- **`StructuredHexMesh3D`** — unit-cube mesh of trilinear hexahedra.
- **`assemble_poisson_system`** — global stiffness matrix and load vector assembly for Poisson (2D/3D), with optional manufactured source terms.
- **`assemble_elasticity_system`** — global stiffness matrix assembly for linear isotropic elasticity (plane strain 2D, full 3D).
- **`DirichletConstraints`** — strong Dirichlet BC enforcement and RHS correction.
- **Matrix-free operators** — `PoissonLaplacianMF2D`, `PoissonLaplacianMF3D`, `ElasticityMF2D`, `ElasticityMF3D` implementing `LinearOperator` via element-by-element apply.
- **CSR sparse matrix** — `CsrMatrix` with `LinearOperator` implementation.

## Numerical scope

Q1 finite elements on structured meshes. 2×2 Gauss quadrature (2D) and 2×2×2 (3D). Dirichlet BCs applied strongly via row/column modification (assembled) or constraint injection (matrix-free).

## Dependencies

Depends on `spinoza-core` for `LinearOperator` and `SpinozaError` types.
