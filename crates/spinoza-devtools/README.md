# spinoza-devtools

Developer CLI for the Spinoza multiphysics framework.

## Commands

### `audit`

Validate a TOML case spec against the capability registry.

```bash
spinoza-devtools audit spec/poisson_minimal_ok.toml
spinoza-devtools audit spec/poisson_minimal_ok.toml --format json
```

### `solve`

Build and solve a problem from a case spec. Writes deterministic JSON output.

```bash
spinoza-devtools solve spec/poisson_minimal_ok.toml --out result.json
spinoza-devtools solve spec/poisson_3d_minimal_ok.toml --out result.json --backend matrixfree --precond mg
```

### `list`

List all discovered method packs with their capabilities and registered factories.

```bash
spinoza-devtools list
spinoza-devtools list --format json
```

### `new-pack`

Generate a new method pack crate scaffold.

```bash
spinoza-devtools new-pack my_diffusion --path crates
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Audit found missing capabilities |
| 2 | Validation or solve error |
