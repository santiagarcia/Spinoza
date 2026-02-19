# spinoza-pack-linear-solvers

Solver algorithm pack for the Spinoza multiphysics framework.

## Purpose

This pack owns all generic linear solver factory registrations for single-field systems. Physics packs (Poisson, elasticity, etc.) must **not** register solver factories — they request solvers from the registry instead. This separation is enforced by conformance check 12.

## Registered solver factories

| Solver | Key | Systems |
|--------|-----|---------|
| **CG** | `{cg, spd=true, complex=false, single_field}` | SPD single-field |
| **GMRES** | `{gmres, spd=false, complex=false, single_field}` | General single-field |

## Capabilities

`Solver_CG`, `Solver_GMRES`

## Implementation

Both factories are thin wrappers that delegate to `spinoza_solve::solve_cg` and `spinoza_solve::solve_gmres`. The factory layer provides typed key matching and registry integration; the actual algorithm code lives in `spinoza-solve`.

Default parameters:
- **CG:** tolerance = 1e-10, max iterations = 20,000
- **GMRES:** tolerance = 1e-10, max iterations = 20,000, restart = 50

## Adding a new solver

1. Implement the algorithm in `spinoza-solve`.
2. Create a factory struct implementing `SolverFactory` in this crate.
3. Register it in `LinearSolversPack::register`.
4. Add the corresponding `Capability` variant to `CAPABILITIES`.
5. Run `cargo test -p spinoza-pack-linear-solvers` to verify conformance.
