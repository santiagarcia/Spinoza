# spinoza-pack-elasticity

Method pack for linear isotropic elasticity.

## Registered components

| Component | Type | Key |
|-----------|------|-----|
| `ElasticityOperator` | Operator | `elasticity / assembled|matrixfree / 2D|3D / H1Vector` |
| `ElasticityJacobiPrecond` | Preconditioner | `jacobi / assembled|matrixfree / elasticity / 2D|3D` |
| `ElasticityBuilder` | Builder | `elasticity / 2D|3D / assembled|matrixfree` |

## Capabilities

`Space_H1Vector`, `Operator_Elasticity`, `BC_Dirichlet`, `Precond_Jacobi`

## Physics

- **2D:** plane strain with interleaved DOF numbering `[ux0, uy0, ux1, uy1, ...]`
- **3D:** full linear elasticity `[ux0, uy0, uz0, ux1, uy1, uz1, ...]`
- Material parameters (`young_modulus`, `poisson_ratio`) read from the case spec.

## Note on solvers

This pack does **not** register solver factories. Solvers are provided by `spinoza-pack-linear-solvers`.
