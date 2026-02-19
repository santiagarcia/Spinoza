# spinoza-solve

Linear solver implementations for Spinoza.

## Solvers

| Solver | Function | Systems |
|--------|----------|---------|
| Conjugate Gradient | `solve_cg` | SPD |
| CG + Jacobi | `solve_cg_jacobi` | SPD (convenience wrapper) |
| Restarted GMRES | `solve_gmres` | General |

All solvers accept any `LinearOperator` and an optional `Preconditioner`, returning a `SolveResult` with the solution vector, residual norm, and iteration count.

## Preconditioners

- **`JacobiPreconditioner`** — diagonal scaling built from the operator's `diagonal()` method.
- **`MultigridPreconditioner2D` / `MultigridPreconditioner3D`** — geometric multigrid V-cycle with configurable levels, smoother sweeps, and coarse-grid CG solve.

## Dependencies

Depends on `spinoza-core` for trait definitions and `spinoza-disc` for mesh types used by multigrid hierarchy construction.
