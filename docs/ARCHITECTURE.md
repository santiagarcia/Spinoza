# Spinoza Architecture

Spinoza is organized as a Rust workspace with strict crate boundaries.
The current implementation delivers working Poisson, coupled elliptic 2x2, and linear elasticity paths.

## Workspace crates

1. spinoza-core defines shared types, capabilities, spec schema, and errors.
2. spinoza-disc defines structured mesh generation and finite element assembly.
3. spinoza-solve defines linear solver routines (CG, GMRES).
4. spinoza-kernels is reserved for future low level kernels.
5. spinoza-app is reserved for future high level orchestration.
6. spinoza-devtools provides CLI workflows such as audit and solve.
7. spinoza-pack-poisson provides scalar Poisson operators and preconditioners.
8. spinoza-pack-block2x2 provides coupled 2x2 block operators and solvers.
9. spinoza-pack-elasticity provides linear elasticity operators and preconditioners.
10. spinoza-pack-linear-solvers provides generic solver algorithms (CG, GMRES).

## Dependency rules

Dependencies flow from higher level crates to lower level crates.
spinoza-core is the foundation and has no internal crate dependencies.
spinoza-disc depends on spinoza-core.
spinoza-solve depends on spinoza-core and spinoza-disc.
spinoza-devtools depends on spinoza-core, spinoza-disc, and spinoza-solve.
spinoza-kernels and spinoza-app stay as placeholders in this stage.

## Implemented numerical scope

The first implemented discretization is Q1 finite elements on structured unit domain meshes.
The 2D path uses bilinear quads on an Nx by Ny unit square grid.
The 3D path uses trilinear hexes on an Nx by Ny by Nz unit cube grid.
The assembled operator is the scalar Laplacian using 2 by 2 Gauss quadrature in 2D and 2 by 2 by 2 Gauss quadrature in 3D.
The matrix free operator uses the same basis and quadrature rules and applies the action element by element without building CSR.
Dirichlet boundary conditions are applied strongly by row and column modification.
The solver path uses Conjugate Gradient for symmetric positive definite systems and restarted GMRES for general systems.
The assembled backend uses Jacobi preconditioning.
The matrixfree backend supports Jacobi and geometric multigrid V cycle preconditioning.
The multigrid hierarchy is geometric and built by halving structured mesh resolution across levels.

The elasticity pack implements plane strain (2D) and full 3D linear isotropic elasticity.
DOF numbering is interleaved: [ux0, uy0, ux1, uy1, ...] in 2D and [ux0, uy0, uz0, ...] in 3D.
Element stiffness uses B transpose D B integration with the standard constitutive matrices.
Material parameters (Young's modulus and Poisson's ratio) are read from the case spec.

## Supported command workflow

The audit command validates a case spec and checks capability coverage.
The solve command runs the first real Poisson path and writes deterministic JSON output.
The command form is spinoza-devtools solve <case.toml> --out <path.json> --backend <assembled|matrixfree> --precond <jacobi|mg>.
The default backend is assembled.
The default preconditioner is jacobi.
The matrixfree backend is used to avoid global matrix materialization and to support larger problems.
The mg preconditioner is available for matrixfree backend.

The repository also contains a simple operator and solve benchmark tool.
The command form is cargo run -p spinoza-devtools --bin spinoza-op-bench.

## Current limits

Mesh dimensions 2 and 3 are supported in the solve path.
Scalar H1, vector H1, and coupled multi-field problems are supported.
Equation types: poisson, coupled_elliptic_2x2, elasticity.
Solvers: cg, block_pcg, gmres.
Preconditioners: jacobi, block_jacobi, mg.

## Pack ownership rules

Physics packs (poisson, elasticity) own operators and preconditioners only.
Generic single-field solver factories belong to the solver algorithm pack (linear_solvers).
Block-specific solvers (block_pcg) may remain in the pack that defines the block structure.
Conformance check 12 enforces this separation at test time.

## Next evolution path

The current implementation keeps APIs small so extensions can be introduced incrementally.
Future tasks can add richer mesh controls from the case spec, additional operators, and more solver backends without changing the current audit contract.
