# Core Interfaces

This document describes the key interfaces used by the current Spinoza MVP path.

## Capability and registry

Capability is an enum that names each discrete feature that can be audited.
The current default registry marks these capabilities as implemented.

1. Space_H1Scalar
2. Space_H1Vector
3. Operator_Laplacian
4. BC_Dirichlet
5. Solver_CG
6. Precond_Jacobi
7. Precond_MG
8. BlockOperator_2x2
9. Precond_BlockJacobi
10. Solver_BlockPCG
11. Equation_CoupledElliptic2x2
12. Operator_Elasticity
13. Solver_GMRES

Capabilities are registered by method packs at link time via the inventory crate.

## CaseSpec and CompiledPlan

CaseSpec is the typed TOML schema model loaded from a case file.
CompiledPlan maps a validated CaseSpec to the required capability set.
The mapping remains deterministic and stable.

## LinearOperator

LinearOperator is the stable solver side interface for assembled and matrixfree paths.
It exposes three methods.

1. size returns the square operator size.
2. apply performs y equals A x without allocation.
3. diagonal fills the Jacobi diagonal and returns an error if unsupported.

CSR and matrixfree Poisson operators both implement this interface.

## Preconditioner

Preconditioner is the solver side interface for reusable preconditioners.
It exposes one method apply that computes z equals M inverse r without allocation.

Jacobi and geometric multigrid preconditioners implement this interface.

## Method pack plugin system

A method pack is a self-contained crate that registers new mathematics into
Spinoza without modifying core. Packs are discovered at link time via the
`inventory` crate.

### MethodPack trait

Every pack implements the MethodPack trait and registers via the
`spinoza_register_pack!` macro. The trait requires four methods.

1. name returns a human-readable pack name.
2. version returns the pack version string.
3. capabilities returns a static slice of Capability values.
4. register receives a mutable MethodRegistry and registers builders.

### ProblemBuilder trait

ProblemBuilder is the factory interface for translating a CaseSpec into an
executable problem. It coordinates component factories from the registry.
It requires six methods.

1. name returns the builder name (e.g. "PoissonBuilder").
2. equation_type returns the equation string this builder handles.
3. supported_dimensions returns a slice of supported dimensions.
4. supported_backends returns a slice of supported backend strings.
5. build receives the spec, parameters, and a MethodRegistry reference,
   then composes an operator, preconditioner, and solver from registered
   factories and returns a BuiltProblem.

### Component factory traits

Operators, preconditioners, and solvers are independently registrable
via typed factory traits. Each factory declares:

- name: a human-readable identifier.
- supported_keys: the typed keys it can handle.
- capabilities_provided: capabilities this factory adds.
- capabilities_required: capabilities that must be available.

#### OperatorFactory

Produces a linear operator and RHS vector for a given equation family,
backend, and dimension. Keyed by OperatorKey with the following fields:

- equation_family: e.g. "poisson", "elasticity"
- backend: "assembled" or "matrixfree"
- dimension: 2 or 3
- space_signature: e.g. "H1Scalar", "H1Vector"
- element_family: "quad" (2D) or "hex" (3D)
- order: polynomial order (1 for Q1)
- block_structure: "single_field" or "block2x2"

Returns a BuiltOperator containing the operator, rhs,
field_names, and field_sizes.

#### PreconditionerFactory

Produces a preconditioner given the assembled operator and case spec.
Keyed by PreconditionerKey with the following fields:

- precond_type: e.g. "jacobi", "mg", "block_jacobi"
- backend: "assembled" or "matrixfree"
- dimension: 2 or 3
- operator_family: e.g. "poisson", "elasticity"
- space_signature: e.g. "H1Scalar", "H1Vector"
- element_family: "quad" or "hex"
- order: polynomial order
- block_structure: "single_field" or "block2x2"

Returns a boxed Preconditioner.

#### SolverFactory

Runs a linear or nonlinear solver. Keyed by SolverKey with the following
fields:

- solver_type: e.g. "cg", "gmres", "block_pcg"
- spd: whether the system is symmetric positive definite
- complex: whether the system involves complex arithmetic
- block_structure: "single_field" or "block2x2"

Returns a SolveResult containing solution, residual_norm,
and iterations.

### MethodRegistry

The registry stores all builders and component factories discovered from
packs. Key operations.

1. from_packs builds the registry by iterating all registered packs.
2. find_builder returns the first matching builder.
3. find_builder_checked returns a BuilderLookup with conflict detection.
4. find_operator returns a ComponentLookup for operator factories.
5. find_preconditioner returns a ComponentLookup for preconditioner factories.
6. find_solver returns a ComponentLookup for solver factories.
7. pack_info returns serializable metadata for all packs.
8. operator_factory_info, preconditioner_factory_info, solver_factory_info
   return introspection data for the list command.

### Builder conflict detection

When find_builder_checked finds multiple builders matching the same
equation type, dimension, and backend, it returns BuilderLookup::Ambiguous
with an AmbiguousBuilderError containing the candidate list. This error
is structured and machine-readable.

### Component conflict detection

The find_operator, find_preconditioner, and find_solver methods use the
same ambiguity detection pattern. When multiple factories match a key,
ComponentLookup::Ambiguous is returned with an AmbiguousComponentError.
This ensures deterministic selection: if only one factory matches, it is
used; if multiple match, an actionable error is reported.

### Registering components without a ProblemBuilder

A pack can register individual factories (operator, preconditioner, solver)
without providing a ProblemBuilder. For example, a pack that provides a new
preconditioner can register a PreconditionerFactory and existing
ProblemBuilders in other packs will be able to use it automatically when
the matching preconditioner key is requested via the case spec.

### Creating a new pack

Use the scaffold command:

    spinoza-devtools new-pack <pack_name> --path <dir>

This generates a complete crate with Cargo.toml, a lib.rs with MethodPack
and ProblemBuilder stubs, a conformance test, and a README.

After generating the pack:

1. Add the crate to the workspace Cargo.toml members list.
2. Add it as a dependency in spinoza-devtools/Cargo.toml.
3. Anchor the pack struct in main.rs and solve_case.rs.
4. Implement the builder's build method.
5. Run conformance tests.

### Pack conformance testing

The conformance harness module (spinoza_core::conformance) provides
reusable verification for pack authors. It checks:

1. The pack is discoverable via inventory.
2. Capabilities are deterministic across repeated calls.
3. Capabilities are non-empty.
4. Version is non-empty.
5. Builders are registered and queryable.
6. No duplicate builder keys exist within the pack.
7. Operator factories are registered (informational).
8. Preconditioner factories are registered (informational).
9. Solver factories are registered (informational).
10. No duplicate operator factory keys within the pack.
11. No duplicate solver factory keys within the pack.
12. Physics packs must not register single_field solver factories
    (enforces separation between physics and solver algorithm concerns).

Usage in tests:

    spinoza_core::conformance::assert_pack_conforms("my_pack");

### Listing discovered packs

    spinoza-devtools list [--format text|json]

Text output shows discovered packs with their versions and capabilities,
plus registered builders, operator factories, preconditioner factories,
and solver factories. JSON output is a single object with a `packs` array,
plus `operator_factories`, `preconditioner_factories`, and
`solver_factories` arrays.

## Audit command contract

The command form is spinoza-devtools audit <path> with optional --format text or --format json.
Default format is text.

Text output is stable.
Section headers are fixed.
Capability names are sorted by string representation.
No timestamp or environment dependent information is included.

JSON output is a single object to stdout.
The object contains these keys.

1. problem_name
2. case_path
3. required_capabilities
4. missing_capabilities
5. status
6. exit_code
7. validation_errors

Exit codes remain fixed.

1. 0 means audit success with no missing capabilities.
2. 1 means audit success with missing capabilities.
3. 2 means parse or validation failure.

## Solve command contract

The command form is spinoza-devtools solve <case.toml> --out <path.json> --backend <assembled|matrixfree> --precond <jacobi|mg|block_jacobi>.
Default backend is assembled.
Default preconditioner is jacobi.

The solve path supports Poisson problems in 2D and 3D, coupled elliptic 2x2 systems,
and linear elasticity problems in 2D (plane strain) and 3D.
It accepts mesh dimension 2 or 3.
The registry-based dispatch selects the first matching builder from discovered packs.

The solve JSON output contains deterministic keys.

1. problem_name
2. selected_pack
3. selected_builder
4. backend
5. preconditioner
6. dimension
7. nx
8. ny
9. nz
10. field_names
11. field_sizes
12. selected_operator (nullable, present when resolved via registry)
13. selected_preconditioner (nullable, present when resolved via registry)
14. selected_solver (the pack::component that solved the system)
15. solution_vector
16. residual_norm
17. iteration_count

The selected_pack and selected_builder fields provide traceability
showing which pack and builder were used to construct the problem.

## Error model

SpinozaError is used for parse, validation, capability, and operator related failures.
