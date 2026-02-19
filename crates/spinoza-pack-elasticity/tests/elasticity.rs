//! Integration tests for the elasticity method pack.
//!
//! Tests:
//! 1. Operator equivalence (assembled vs matrixfree) in 2D and 3D.
//! 2. Solve convergence with CG + Jacobi (2D and 3D).
//! 3. GMRES solve works (2D).
//! 4. Pack-level registration and key selection.

use spinoza_core::{
    BuilderLookup, Capability, CapabilityRegistry, ComponentLookup, LinearOperator, MethodPack,
    MethodRegistry, OperatorKey, PreconditionerKey, SolverKey,
};

// Ensure the elasticity (and poisson, linear-solvers) packs are linked.
fn init() {
    let _ = spinoza_pack_elasticity::ElasticityPack;
    let _ = spinoza_pack_poisson::PoissonPack;
    let _ = spinoza_pack_linear_solvers::LinearSolversPack;
}

fn deterministic_vector(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        out.push(2.0 * val - 1.0);
    }
    out
}

fn relative_error(a: &[f64], b: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = av - bv;
        num += d * d;
        den += av * av;
    }
    num.sqrt() / den.sqrt().max(1e-16)
}

// ---------------------------------------------------------------------------
// Operator equivalence tests
// ---------------------------------------------------------------------------

#[test]
fn elasticity_operator_apply_equivalence_2d() {
    init();
    let spec = elasticity_spec_2d();
    let registry = MethodRegistry::from_packs();
    let nx = 6;
    let ny = 6;

    let op_key_asm = OperatorKey {
        equation_family: "elasticity".into(),
        backend: "assembled".into(),
        dimension: 2,
        space_signature: "H1Vector".into(),
        element_family: "quad".into(),
        order: 1,
        block_structure: "single_field".into(),
    };
    let op_key_mf = OperatorKey {
        equation_family: "elasticity".into(),
        backend: "matrixfree".into(),
        dimension: 2,
        space_signature: "H1Vector".into(),
        element_family: "quad".into(),
        order: 1,
        block_structure: "single_field".into(),
    };

    let built_asm = match registry.find_operator(&op_key_asm) {
        ComponentLookup::Found { component, .. } => {
            component.build(&op_key_asm, &spec, nx, ny, 1).unwrap()
        }
        other => panic!("expected Found for assembled, got {other:?}"),
    };
    let built_mf = match registry.find_operator(&op_key_mf) {
        ComponentLookup::Found { component, .. } => {
            component.build(&op_key_mf, &spec, nx, ny, 1).unwrap()
        }
        other => panic!("expected Found for matrixfree, got {other:?}"),
    };

    let ndofs = built_asm.operator.size();
    assert_eq!(ndofs, built_mf.operator.size());

    let x = deterministic_vector(ndofs);
    let mut y_asm = vec![0.0; ndofs];
    let mut y_mf = vec![0.0; ndofs];
    built_asm.operator.apply(&x, &mut y_asm);
    built_mf.operator.apply(&x, &mut y_mf);

    let rel = relative_error(&y_asm, &y_mf);
    assert!(
        rel < 1e-10,
        "2D elasticity operator apply mismatch rel={rel:e}"
    );
}

#[test]
fn elasticity_operator_apply_equivalence_3d() {
    init();
    let spec = elasticity_spec_3d();
    let registry = MethodRegistry::from_packs();
    let nx = 4;
    let ny = 4;
    let nz = 4;

    let op_key_asm = OperatorKey {
        equation_family: "elasticity".into(),
        backend: "assembled".into(),
        dimension: 3,
        space_signature: "H1Vector".into(),
        element_family: "hex".into(),
        order: 1,
        block_structure: "single_field".into(),
    };
    let op_key_mf = OperatorKey {
        equation_family: "elasticity".into(),
        backend: "matrixfree".into(),
        dimension: 3,
        space_signature: "H1Vector".into(),
        element_family: "hex".into(),
        order: 1,
        block_structure: "single_field".into(),
    };

    let built_asm = match registry.find_operator(&op_key_asm) {
        ComponentLookup::Found { component, .. } => {
            component.build(&op_key_asm, &spec, nx, ny, nz).unwrap()
        }
        other => panic!("expected Found for assembled 3d, got {other:?}"),
    };
    let built_mf = match registry.find_operator(&op_key_mf) {
        ComponentLookup::Found { component, .. } => {
            component.build(&op_key_mf, &spec, nx, ny, nz).unwrap()
        }
        other => panic!("expected Found for matrixfree 3d, got {other:?}"),
    };

    let ndofs = built_asm.operator.size();
    assert_eq!(ndofs, built_mf.operator.size());

    let x = deterministic_vector(ndofs);
    let mut y_asm = vec![0.0; ndofs];
    let mut y_mf = vec![0.0; ndofs];
    built_asm.operator.apply(&x, &mut y_asm);
    built_mf.operator.apply(&x, &mut y_mf);

    let rel = relative_error(&y_asm, &y_mf);
    assert!(
        rel < 1e-10,
        "3D elasticity operator apply mismatch rel={rel:e}"
    );
}

// ---------------------------------------------------------------------------
// Solve convergence tests
// ---------------------------------------------------------------------------

#[test]
fn elasticity_solve_cg_2d() {
    init();
    let spec = elasticity_spec_2d();
    let registry = MethodRegistry::from_packs();

    let (builder, selection) =
        match registry.find_builder_checked("elasticity", 2, "assembled") {
            BuilderLookup::Found { builder, selection } => (builder, selection),
            other => panic!("expected Found builder, got {other:?}"),
        };
    assert_eq!(selection.pack_name, "elasticity");

    let built = builder
        .build(&spec, "assembled", "jacobi", 8, 8, 1, &registry)
        .unwrap();

    // Solve via CG solver from registry.
    let solver_key = SolverKey {
        solver_type: "cg".into(),
        spd: true,
        complex: false,
        block_structure: "single_field".into(),
    };
    let res = match registry.find_solver(&solver_key) {
        ComponentLookup::Found { component, .. } => component
            .solve(
                &solver_key,
                built.operator.as_ref(),
                &built.rhs,
                built.preconditioner.as_deref(),
            )
            .unwrap(),
        other => panic!("expected Found solver, got {other:?}"),
    };

    // With zero BCs and zero body force, solution should be zero.
    let max_val: f64 = res
        .solution
        .iter()
        .map(|v| v.abs())
        .fold(0.0, f64::max);
    assert!(
        max_val < 1e-8,
        "2D elasticity zero-BC solution should be zero, got max={max_val:e}"
    );
    assert!(
        res.residual_norm < 1e-8,
        "residual should be small, got {}",
        res.residual_norm
    );
}

#[test]
fn elasticity_solve_cg_3d() {
    init();
    let spec = elasticity_spec_3d();
    let registry = MethodRegistry::from_packs();

    let (builder, _) = match registry.find_builder_checked("elasticity", 3, "assembled") {
        BuilderLookup::Found { builder, selection } => (builder, selection),
        other => panic!("expected Found builder 3d, got {other:?}"),
    };

    let built = builder
        .build(&spec, "assembled", "jacobi", 4, 4, 4, &registry)
        .unwrap();

    let solver_key = SolverKey {
        solver_type: "cg".into(),
        spd: true,
        complex: false,
        block_structure: "single_field".into(),
    };
    let res = match registry.find_solver(&solver_key) {
        ComponentLookup::Found { component, .. } => component
            .solve(
                &solver_key,
                built.operator.as_ref(),
                &built.rhs,
                built.preconditioner.as_deref(),
            )
            .unwrap(),
        other => panic!("expected Found solver 3d, got {other:?}"),
    };

    let max_val: f64 = res
        .solution
        .iter()
        .map(|v| v.abs())
        .fold(0.0, f64::max);
    assert!(
        max_val < 1e-8,
        "3D elasticity zero-BC solution should be zero, got max={max_val:e}"
    );
}

// ---------------------------------------------------------------------------
// GMRES solve test
// ---------------------------------------------------------------------------

#[test]
fn elasticity_solve_gmres_2d() {
    init();
    let spec = elasticity_spec_2d_gmres();
    let registry = MethodRegistry::from_packs();

    let (builder, _) = match registry.find_builder_checked("elasticity", 2, "assembled") {
        BuilderLookup::Found { builder, selection } => (builder, selection),
        other => panic!("expected Found builder for GMRES test, got {other:?}"),
    };

    let built = builder
        .build(&spec, "assembled", "jacobi", 6, 6, 1, &registry)
        .unwrap();

    let solver_key = SolverKey {
        solver_type: "gmres".into(),
        spd: false,
        complex: false,
        block_structure: "single_field".into(),
    };
    let res = match registry.find_solver(&solver_key) {
        ComponentLookup::Found { component, .. } => component
            .solve(
                &solver_key,
                built.operator.as_ref(),
                &built.rhs,
                built.preconditioner.as_deref(),
            )
            .unwrap(),
        other => panic!("expected Found GMRES solver, got {other:?}"),
    };

    let max_val: f64 = res
        .solution
        .iter()
        .map(|v| v.abs())
        .fold(0.0, f64::max);
    assert!(
        max_val < 1e-8,
        "GMRES 2D elasticity zero-BC solution should be zero, got max={max_val:e}"
    );
}

// ---------------------------------------------------------------------------
// Key selection: no collision between Poisson and Elasticity
// ---------------------------------------------------------------------------

#[test]
fn no_key_collision_between_packs() {
    init();
    let registry = MethodRegistry::from_packs();

    // Elasticity operator key is distinct from Poisson
    let elast_op_key = OperatorKey {
        equation_family: "elasticity".into(),
        backend: "assembled".into(),
        dimension: 2,
        space_signature: "H1Vector".into(),
        element_family: "quad".into(),
        order: 1,
        block_structure: "single_field".into(),
    };
    match registry.find_operator(&elast_op_key) {
        ComponentLookup::Found { selection, .. } => {
            assert!(
                selection.component_name.contains("Elasticity"),
                "should find Elasticity operator, got: {}",
                selection.component_name
            );
        }
        other => panic!("expected Found for elasticity op, got {other:?}"),
    }

    // Poisson operator key is distinct
    let poisson_op_key = OperatorKey {
        equation_family: "poisson".into(),
        backend: "assembled".into(),
        dimension: 2,
        space_signature: "H1Scalar".into(),
        element_family: "quad".into(),
        order: 1,
        block_structure: "single_field".into(),
    };
    match registry.find_operator(&poisson_op_key) {
        ComponentLookup::Found { selection, .. } => {
            assert!(
                selection.component_name.contains("Poisson"),
                "should find Poisson operator, got: {}",
                selection.component_name
            );
        }
        other => panic!("expected Found for poisson op, got {other:?}"),
    }

    // Preconditioner keys don't collide
    let elast_pc_key = PreconditionerKey {
        precond_type: "jacobi".into(),
        backend: "assembled".into(),
        dimension: 2,
        operator_family: "elasticity".into(),
        space_signature: "H1Vector".into(),
        element_family: "quad".into(),
        order: 1,
        block_structure: "single_field".into(),
    };
    match registry.find_preconditioner(&elast_pc_key) {
        ComponentLookup::Found { selection, .. } => {
            assert!(
                selection.component_name.contains("Elasticity"),
                "should find Elasticity precond, got: {}",
                selection.component_name
            );
        }
        other => panic!("expected Found for elasticity precond, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Capability audit for elasticity specs
// ---------------------------------------------------------------------------

#[test]
fn elasticity_capabilities_in_registry() {
    init();
    let cap_reg = CapabilityRegistry::default_registry();
    assert!(
        cap_reg.has(Capability::OperatorElasticity),
        "registry should have OperatorElasticity"
    );
    assert!(
        cap_reg.has(Capability::SolverGMRES),
        "registry should have SolverGMRES"
    );
    assert!(
        cap_reg.has(Capability::SpaceH1Vector),
        "registry should have SpaceH1Vector"
    );
}

// ---------------------------------------------------------------------------
// Helpers to build in-memory specs
// ---------------------------------------------------------------------------

fn elasticity_spec_2d() -> spinoza_core::CaseSpec {
    let toml_str = r#"
[problem]
name = "Test elasticity 2D"

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
linear = "cg"
precond = "jacobi"
"#;
    let spec: spinoza_core::CaseSpec = toml::from_str(toml_str).unwrap();
    spec.validate().expect("elasticity 2d spec should validate");
    spec
}

fn elasticity_spec_3d() -> spinoza_core::CaseSpec {
    let toml_str = r#"
[problem]
name = "Test elasticity 3D"

[mesh]
dimension = 3

[[fields]]
name = "displacement"
kind = "vector"
space = "H1"
complex = false

[equation]
type = "elasticity"
young_modulus = 210000.0
poisson_ratio = 0.3

[[bcs]]
type = "dirichlet"
field = "displacement"
value = 0.0

[solver]
linear = "cg"
precond = "jacobi"
"#;
    let spec: spinoza_core::CaseSpec = toml::from_str(toml_str).unwrap();
    spec.validate().expect("elasticity 3d spec should validate");
    spec
}

fn elasticity_spec_2d_gmres() -> spinoza_core::CaseSpec {
    let toml_str = r#"
[problem]
name = "Test elasticity 2D GMRES"

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
"#;
    let spec: spinoza_core::CaseSpec = toml::from_str(toml_str).unwrap();
    spec.validate()
        .expect("elasticity 2d gmres spec should validate");
    spec
}
