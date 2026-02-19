//! Tests for solver selection from the linear-solvers pack.
//!
//! Verifies:
//! 1. Poisson solve selects CG from linear_solvers pack.
//! 2. Elasticity GMRES solve selects GMRES from linear_solvers pack.
//! 3. No ambiguous selection for any existing spec keys.
//! 4. Physics packs do not register single_field solver factories.
//! 5. COMPONENT_NOT_FOUND when no factory matches a key.

use spinoza_core::{
    BuilderLookup, Capability, CapabilityRegistry, ComponentLookup, MethodRegistry, SolverKey,
};

fn init() {
    let _ = spinoza_pack_linear_solvers::LinearSolversPack;
    let _ = spinoza_pack_poisson::PoissonPack;
    let _ = spinoza_pack_elasticity::ElasticityPack;
    let _ = spinoza_pack_block2x2::Block2x2Pack;
}

// ---------------------------------------------------------------------------
// CG solver comes from linear_solvers, not from poisson
// ---------------------------------------------------------------------------

#[test]
fn poisson_solve_selects_cg_from_linear_solvers() {
    init();
    let registry = MethodRegistry::from_packs();

    let solver_key = SolverKey {
        solver_type: "cg".into(),
        spd: true,
        complex: false,
        block_structure: "single_field".into(),
    };
    match registry.find_solver(&solver_key) {
        ComponentLookup::Found { selection, .. } => {
            assert_eq!(
                selection.pack_name, "linear_solvers",
                "CG solver should come from linear_solvers pack, got: {}",
                selection.pack_name
            );
            assert_eq!(selection.component_name, "CgSolver");
        }
        ComponentLookup::NotFound => panic!("CG solver factory not found"),
        ComponentLookup::Ambiguous(err) => panic!("ambiguous CG solver: {err}"),
    }
}

// ---------------------------------------------------------------------------
// GMRES solver comes from linear_solvers, not from elasticity
// ---------------------------------------------------------------------------

#[test]
fn elasticity_solve_selects_gmres_from_linear_solvers() {
    init();
    let registry = MethodRegistry::from_packs();

    let solver_key = SolverKey {
        solver_type: "gmres".into(),
        spd: false,
        complex: false,
        block_structure: "single_field".into(),
    };
    match registry.find_solver(&solver_key) {
        ComponentLookup::Found { selection, .. } => {
            assert_eq!(
                selection.pack_name, "linear_solvers",
                "GMRES solver should come from linear_solvers pack, got: {}",
                selection.pack_name
            );
            assert_eq!(selection.component_name, "GmresSolver");
        }
        ComponentLookup::NotFound => panic!("GMRES solver factory not found"),
        ComponentLookup::Ambiguous(err) => panic!("ambiguous GMRES solver: {err}"),
    }
}

// ---------------------------------------------------------------------------
// Block PCG solver stays in block2x2 pack (not single_field)
// ---------------------------------------------------------------------------

#[test]
fn block_pcg_stays_in_block2x2_pack() {
    init();
    let registry = MethodRegistry::from_packs();

    let solver_key = SolverKey {
        solver_type: "block_pcg".into(),
        spd: true,
        complex: false,
        block_structure: "block2x2".into(),
    };
    match registry.find_solver(&solver_key) {
        ComponentLookup::Found { selection, .. } => {
            assert_eq!(
                selection.pack_name, "block2x2",
                "block_pcg solver should still come from block2x2 pack, got: {}",
                selection.pack_name
            );
        }
        ComponentLookup::NotFound => panic!("block_pcg solver factory not found"),
        ComponentLookup::Ambiguous(err) => panic!("ambiguous block_pcg solver: {err}"),
    }
}

// ---------------------------------------------------------------------------
// No ambiguous selection for any known solver keys
// ---------------------------------------------------------------------------

#[test]
fn no_ambiguous_solver_selection() {
    init();
    let registry = MethodRegistry::from_packs();

    let keys = [
        SolverKey {
            solver_type: "cg".into(),
            spd: true,
            complex: false,
            block_structure: "single_field".into(),
        },
        SolverKey {
            solver_type: "gmres".into(),
            spd: false,
            complex: false,
            block_structure: "single_field".into(),
        },
        SolverKey {
            solver_type: "block_pcg".into(),
            spd: true,
            complex: false,
            block_structure: "block2x2".into(),
        },
    ];

    for key in &keys {
        match registry.find_solver(key) {
            ComponentLookup::Ambiguous(err) => {
                panic!("ambiguous selection for solver key {key}: {err}");
            }
            _ => {} // Found or NotFound are both acceptable
        }
    }
}

// ---------------------------------------------------------------------------
// Physics packs don't register single_field solver factories
// ---------------------------------------------------------------------------

#[test]
fn physics_packs_have_no_single_field_solvers() {
    init();
    let registry = MethodRegistry::from_packs();

    let solver_infos = registry.solver_factory_info();

    // Check each physics pack.
    for physics_pack in &["poisson", "elasticity"] {
        let pack_solvers: Vec<_> = solver_infos
            .iter()
            .filter(|info| info.get("pack").map(|s| s.as_str()) == Some(physics_pack))
            .filter(|info| {
                info.get("keys")
                    .map(|k| k.contains("single_field"))
                    .unwrap_or(false)
            })
            .collect();

        assert!(
            pack_solvers.is_empty(),
            "physics pack '{}' should not register single_field solver factories, found: {:?}",
            physics_pack,
            pack_solvers
                .iter()
                .filter_map(|f| f.get("name"))
                .collect::<Vec<_>>()
        );
    }
}

// ---------------------------------------------------------------------------
// NotFound when requesting a non-existent solver type
// ---------------------------------------------------------------------------

#[test]
fn solver_not_found_for_nonexistent_type() {
    init();
    let registry = MethodRegistry::from_packs();

    let fake_key = SolverKey {
        solver_type: "bicgstab".into(),
        spd: false,
        complex: false,
        block_structure: "single_field".into(),
    };

    match registry.find_solver(&fake_key) {
        ComponentLookup::NotFound => {} // expected
        ComponentLookup::Found { selection, .. } => {
            panic!(
                "should not find a factory for 'bicgstab', got: {}::{}",
                selection.pack_name, selection.component_name
            );
        }
        ComponentLookup::Ambiguous(err) => panic!("unexpected ambiguity: {err}"),
    }
}

// ---------------------------------------------------------------------------
// Capabilities are provided by linear_solvers pack
// ---------------------------------------------------------------------------

#[test]
fn solver_capabilities_in_registry() {
    init();
    let cap_reg = CapabilityRegistry::default_registry();
    assert!(
        cap_reg.has(&Capability::SolverCG),
        "SolverCG should be in capability registry"
    );
    assert!(
        cap_reg.has(&Capability::SolverGMRES),
        "SolverGMRES should be in capability registry"
    );
}

// ---------------------------------------------------------------------------
// End-to-end: Poisson build + solve via registry
// ---------------------------------------------------------------------------

#[test]
fn poisson_end_to_end_build_and_solve_via_registry() {
    init();
    let spec: spinoza_core::CaseSpec = toml::from_str(
        r#"
[problem]
name = "test"
[mesh]
dimension = 2
[[fields]]
name = "u"
kind = "scalar"
space = "H1"
complex = false
[equation]
type = "poisson"
[[bcs]]
type = "dirichlet"
field = "u"
value = 0.0
[solver]
linear = "cg"
precond = "jacobi"
"#,
    )
    .unwrap();
    spec.validate().unwrap();

    let registry = MethodRegistry::from_packs();
    let (builder, selection) =
        match registry.find_builder_checked("poisson", 2, "assembled") {
            BuilderLookup::Found { builder, selection } => (builder, selection),
            other => panic!("expected Found, got {other:?}"),
        };
    assert_eq!(selection.pack_name, "poisson");

    let built = builder
        .build(&spec, "assembled", "jacobi", 8, 8, 1, &registry)
        .unwrap();

    // Solve via CG from linear_solvers pack.
    let solver_key = SolverKey {
        solver_type: "cg".into(),
        spd: true,
        complex: false,
        block_structure: "single_field".into(),
    };
    let (component, solver_sel) = match registry.find_solver(&solver_key) {
        ComponentLookup::Found {
            component,
            selection,
        } => (component, selection),
        other => panic!("expected Found CG solver, got {other:?}"),
    };
    assert_eq!(solver_sel.pack_name, "linear_solvers");

    let result = component
        .solve(
            &solver_key,
            built.operator.as_ref(),
            &built.rhs,
            built.preconditioner.as_deref(),
        )
        .unwrap();

    assert!(
        result.residual_norm < 1e-8,
        "Poisson CG should converge, got residual={}",
        result.residual_norm
    );
}

// ---------------------------------------------------------------------------
// End-to-end: Elasticity GMRES solve via registry
// ---------------------------------------------------------------------------

#[test]
fn elasticity_end_to_end_gmres_via_registry() {
    init();
    let spec: spinoza_core::CaseSpec = toml::from_str(
        r#"
[problem]
name = "test elasticity"
[mesh]
dimension = 2
[[fields]]
name = "displacement"
kind = "vector"
space = "H1"
complex = false
[equation]
type = "elasticity"
young_modulus = 1.0
poisson_ratio = 0.3
[[bcs]]
type = "dirichlet"
field = "displacement"
value = 0.0
[solver]
linear = "gmres"
precond = "jacobi"
"#,
    )
    .unwrap();
    spec.validate().unwrap();

    let registry = MethodRegistry::from_packs();
    let (builder, _) = match registry.find_builder_checked("elasticity", 2, "assembled") {
        BuilderLookup::Found { builder, selection } => (builder, selection),
        other => panic!("expected Found, got {other:?}"),
    };

    let built = builder
        .build(&spec, "assembled", "jacobi", 6, 6, 1, &registry)
        .unwrap();

    // Solve via GMRES from linear_solvers pack.
    let solver_key = SolverKey {
        solver_type: "gmres".into(),
        spd: false,
        complex: false,
        block_structure: "single_field".into(),
    };
    let (component, solver_sel) = match registry.find_solver(&solver_key) {
        ComponentLookup::Found {
            component,
            selection,
        } => (component, selection),
        other => panic!("expected Found GMRES solver, got {other:?}"),
    };
    assert_eq!(solver_sel.pack_name, "linear_solvers");

    let result = component
        .solve(
            &solver_key,
            built.operator.as_ref(),
            &built.rhs,
            built.preconditioner.as_deref(),
        )
        .unwrap();

    let max_val: f64 = result
        .solution
        .iter()
        .map(|v| v.abs())
        .fold(0.0, f64::max);
    assert!(
        max_val < 1e-8,
        "GMRES elasticity zero-BC solution should be zero, got max={max_val:e}"
    );
}

// ---------------------------------------------------------------------------
// Conformance: all physics packs still pass
// ---------------------------------------------------------------------------

#[test]
fn all_packs_conform() {
    init();
    spinoza_core::conformance::assert_pack_conforms("poisson");
    spinoza_core::conformance::assert_pack_conforms("elasticity");
    spinoza_core::conformance::assert_pack_conforms("block2x2");
    spinoza_core::conformance::assert_pack_conforms("linear_solvers");
}
