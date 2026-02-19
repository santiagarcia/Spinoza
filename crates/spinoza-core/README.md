# spinoza-core

Shared foundation for the Spinoza multiphysics framework.

## What this crate provides

- **`Capability` enum** — names every discrete feature that can be audited (spaces, operators, solvers, preconditioners).
- **`CapabilityRegistry`** — collects capabilities from all linked method packs and answers coverage queries.
- **`CaseSpec` / `CompiledPlan`** — typed TOML schema and the mapping from a validated spec to required capabilities.
- **`MethodPack` trait** — the plugin contract that every method pack implements.
- **`ProblemBuilder` trait** — factory interface for composing an operator, preconditioner, and solver from a spec.
- **Component factory traits** — `OperatorFactory`, `PreconditionerFactory`, `SolverFactory` with typed keys and ambiguity detection.
- **`MethodRegistry`** — runtime registry that discovers packs via `inventory` and resolves component lookups.
- **`conformance` module** — reusable test harness for verifying pack registration and separation rules.
- **`SpinozaError`** — unified error type for parse, validation, and runtime failures.

## Design rules

This crate has **no** internal crate dependencies — it is the dependency root of the workspace.

All types in this crate are trait objects or plain data. No discretization, assembly, or solver logic lives here.
