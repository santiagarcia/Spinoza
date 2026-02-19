# spinoza-pack-block2x2

Method pack for coupled 2×2 block systems.

## Registered components

| Component | Type | Key |
|-----------|------|-----|
| `BlockOperator2x2` | Operator | via coupled builder |
| `BlockJacobiPreconditioner` | Preconditioner | `block_jacobi / matrixfree / 2D|3D` |
| `BlockPcgSolverFactory` | Solver | `block_pcg / spd=true / block2x2` |
| `CoupledElliptic2x2Builder` | Builder | `coupled_elliptic_2x2 / 2D|3D / matrixfree` |

## Capabilities

`BlockOperator_2x2`, `Precond_BlockJacobi`, `Solver_BlockPCG`, `Equation_CoupledElliptic2x2`

## Design

This is a **block-structure pack**, not a physics pack. Because block PCG is specific to the 2×2 block structure, its solver factory stays here (conformance check 12 only restricts `single_field` solvers in physics packs).

The coupled builder composes two sub-operators from the registry (e.g. Poisson Laplacians) with configurable coupling coefficients `alpha` and `beta`.
