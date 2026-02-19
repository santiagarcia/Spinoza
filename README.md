# Spinoza

**A modular multiphysics framework in Rust with a plugin-based method pack architecture.**

Spinoza separates physics, discretization, and solver concerns into independent crates that compose at link time through a typed capability registry.

---

## Highlights

- **Plugin architecture** — method packs register operators, preconditioners, and solvers via the `inventory` crate. No central dispatch code needs modification when adding new physics.
- **Typed key selection** — operator, preconditioner, and solver factories are matched through structured keys with automatic ambiguity detection.
- **Conformance harness** — every pack ships a test that checks registration, key uniqueness, and separation rules at CI time.
- **Assembled & matrix-free** — all physics support both CSR-assembled and matrix-free operator evaluation on structured Q1 meshes (2D quad, 3D hex).

## Workspace layout

```
spinoza-core             shared types, capability registry, spec schema
spinoza-disc             structured meshes and finite element assembly
spinoza-solve            CG, GMRES, multigrid implementations
spinoza-kernels          (reserved) low-level compute kernels
spinoza-app              (reserved) high-level orchestration
spinoza-devtools         CLI: audit, solve, list, new-pack

spinoza-pack-poisson     scalar Poisson operators and preconditioners
spinoza-pack-elasticity  linear elasticity operators and preconditioners
spinoza-pack-block2x2    coupled 2×2 block operators and block PCG
spinoza-pack-linear-solvers  CG and GMRES solver factories
```

## Quick start

```bash
# Audit a case spec against the capability registry
cargo run -p spinoza-devtools -- audit spec/poisson_minimal_ok.toml

# Solve a 2D Poisson problem (assembled, Jacobi preconditioner)
cargo run -p spinoza-devtools -- solve spec/poisson_minimal_ok.toml --out result.json

# Solve 3D Poisson with matrix-free multigrid
cargo run -p spinoza-devtools -- solve spec/poisson_3d_minimal_ok.toml \
    --out result.json --backend matrixfree --precond mg

# List all discovered method packs
cargo run -p spinoza-devtools -- list --format json

# Generate a new pack scaffold
cargo run -p spinoza-devtools -- new-pack my_diffusion --path crates
```

## Supported physics

| Equation | Backends | Preconditioners | Solver |
|----------|----------|-----------------|--------|
| Poisson (scalar) | assembled, matrixfree | Jacobi, MG | CG |
| Elasticity (vector) | assembled, matrixfree | Jacobi | CG, GMRES |
| Coupled elliptic 2×2 | matrixfree | Block Jacobi | Block PCG |

## Architecture rules

1. **Physics packs own operators and preconditioners.** They never register generic solver factories.
2. **Solver packs own algorithm factories.** `spinoza-pack-linear-solvers` provides CG and GMRES for single-field systems.
3. **Block-specific solvers** (e.g. Block PCG) stay in the pack that defines the block structure.
4. **Conformance check 12** enforces the physics/solver separation at test time.

## Running tests

```bash
cargo test --workspace
```

## Case spec format

Case specs are TOML files describing the problem, mesh, fields, equation, boundary conditions, and solver choices. See the [`spec/`](spec/) directory for examples.

## Documentation

- [Architecture overview](docs/ARCHITECTURE.md)
- [Core interfaces reference](docs/CORE_INTERFACES.md)

## License

MIT
