# spinoza-pack-poisson

Method pack for scalar Poisson problems.

## Registered components

| Component | Type | Key |
|-----------|------|-----|
| `PoissonOperator` | Operator | `poisson / assembled|matrixfree / 2D|3D / H1Scalar` |
| `JacobiPreconditioner` | Preconditioner | `jacobi / assembled|matrixfree / poisson / 2D|3D` |
| `MgPreconditioner` | Preconditioner | `mg / matrixfree / poisson / 2D|3D` |
| `PoissonBuilder` | Builder | `poisson / 2D|3D / assembled|matrixfree` |

## Capabilities

`Space_H1Scalar`, `Operator_Laplacian`, `BC_Dirichlet`, `Precond_Jacobi`, `Precond_MG`

## Note on solvers

This pack does **not** register solver factories. Solvers (CG, GMRES) are provided by `spinoza-pack-linear-solvers`. This separation is enforced by conformance check 12.
