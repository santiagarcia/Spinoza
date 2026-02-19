//! Tests for the 2×2 block operator, block Jacobi preconditioner,
//! and coupled elliptic build + solve.

use spinoza_core::{CaseSpec, LinearOperator, MethodRegistry, Preconditioner};
use spinoza_disc::{DirichletConstraints, PoissonLaplacianMF2D, StructuredQuadMesh2D};
use spinoza_pack_block2x2::{BlockJacobiPreconditioner, BlockOperator2x2};
use spinoza_solve::{solve_cg, JacobiPreconditioner};

// -----------------------------------------------------------------------
// Block operator: basic apply correctness
// -----------------------------------------------------------------------

/// Build a tiny 2×2 block operator out of Laplacians and scaled-identity
/// coupling, then verify that `apply` matches element-wise reference.
#[test]
fn block_operator_apply_matches_reference() {
    let nx = 4;
    let ny = 4;
    let mesh = StructuredQuadMesh2D::unit_square(nx, ny);
    let n = mesh.num_nodes();
    let boundary: Vec<usize> = mesh.boundary_nodes().into_iter().collect();

    // Diagonal blocks: Poisson Laplacian MF with zero Dirichlet.
    let bc_vals: Vec<(usize, f64)> = boundary.iter().map(|&i| (i, 0.0)).collect();
    let c0 = DirichletConstraints::new(n, &bc_vals);
    let c1 = DirichletConstraints::new(n, &bc_vals);
    let a00 = PoissonLaplacianMF2D::new(mesh.clone(), c0);
    let a11 = PoissonLaplacianMF2D::new(mesh.clone(), c1);

    // Off-diagonal: zero coupling (identity * 0 = zero).
    let a01 = ZeroOp(n);
    let a10 = ZeroOp(n);

    let block = BlockOperator2x2::new(Box::new(a00), Box::new(a01), Box::new(a10), Box::new(a11));

    assert_eq!(block.size(), 2 * n);

    // Build input x = [1,1,...,1, 2,2,...,2].
    let mut x = vec![1.0; 2 * n];
    x[n..2 * n].fill(2.0);

    let mut y_block = vec![0.0; 2 * n];
    block.apply(&x, &mut y_block);

    // Reference: apply each sub-operator independently.
    let c_ref0 = DirichletConstraints::new(n, &bc_vals);
    let c_ref1 = DirichletConstraints::new(n, &bc_vals);
    let ref_a00 = PoissonLaplacianMF2D::new(mesh.clone(), c_ref0);
    let ref_a11 = PoissonLaplacianMF2D::new(mesh, c_ref1);
    let mut y0_ref = vec![0.0; n];
    let mut y1_ref = vec![0.0; n];
    ref_a00.apply(&x[..n], &mut y0_ref);
    ref_a11.apply(&x[n..], &mut y1_ref);

    for i in 0..n {
        assert!(
            (y_block[i] - y0_ref[i]).abs() < 1e-14,
            "block 0 mismatch at {i}: {} vs {}",
            y_block[i],
            y0_ref[i]
        );
    }
    for i in 0..n {
        assert!(
            (y_block[n + i] - y1_ref[i]).abs() < 1e-14,
            "block 1 mismatch at {i}: {} vs {}",
            y_block[n + i],
            y1_ref[i]
        );
    }
}

/// Zero operator for testing.
struct ZeroOp(usize);

impl LinearOperator for ZeroOp {
    fn size(&self) -> usize {
        self.0
    }
    fn apply(&self, _x: &[f64], y: &mut [f64]) {
        y.iter_mut().for_each(|v| *v = 0.0);
    }
    fn diagonal(&self, d: &mut [f64]) -> Result<(), spinoza_core::SpinozaError> {
        d.iter_mut().for_each(|v| *v = 0.0);
        Ok(())
    }
}

// -----------------------------------------------------------------------
// Block Jacobi preconditioner: apply correctness
// -----------------------------------------------------------------------

#[test]
fn block_jacobi_precond_applies_per_block() {
    let nx = 4;
    let ny = 4;
    let mesh = StructuredQuadMesh2D::unit_square(nx, ny);
    let n = mesh.num_nodes();
    let boundary: Vec<usize> = mesh.boundary_nodes().into_iter().collect();
    let bc_vals: Vec<(usize, f64)> = boundary.iter().map(|&i| (i, 0.0)).collect();

    let c0 = DirichletConstraints::new(n, &bc_vals);
    let c1 = DirichletConstraints::new(n, &bc_vals);
    let op0 = PoissonLaplacianMF2D::new(mesh.clone(), c0);
    let op1 = PoissonLaplacianMF2D::new(mesh, c1);

    let p0 = JacobiPreconditioner::from_operator(&op0).unwrap();
    let p1 = JacobiPreconditioner::from_operator(&op1).unwrap();

    let block_pc = BlockJacobiPreconditioner::new(Box::new(p0), Box::new(p1), n, n);

    // Input residual.
    let r: Vec<f64> = (0..(2 * n)).map(|i| (i as f64) * 0.01).collect();
    let mut z = vec![0.0; 2 * n];
    block_pc.apply(&r, &mut z);

    // Reference: apply each Jacobi independently.
    let c_r0 = DirichletConstraints::new(n, &bc_vals);
    let c_r1 = DirichletConstraints::new(n, &bc_vals);
    let op_r0 = PoissonLaplacianMF2D::new(StructuredQuadMesh2D::unit_square(nx, ny), c_r0);
    let op_r1 = PoissonLaplacianMF2D::new(StructuredQuadMesh2D::unit_square(nx, ny), c_r1);
    let pref0 = JacobiPreconditioner::from_operator(&op_r0).unwrap();
    let pref1 = JacobiPreconditioner::from_operator(&op_r1).unwrap();
    let mut z0_ref = vec![0.0; n];
    let mut z1_ref = vec![0.0; n];
    pref0.apply(&r[..n], &mut z0_ref);
    pref1.apply(&r[n..], &mut z1_ref);

    for i in 0..n {
        assert!(
            (z[i] - z0_ref[i]).abs() < 1e-14,
            "block 0 precond mismatch at {i}"
        );
    }
    for i in 0..n {
        assert!(
            (z[n + i] - z1_ref[i]).abs() < 1e-14,
            "block 1 precond mismatch at {i}"
        );
    }
}

// -----------------------------------------------------------------------
// Coupled elliptic: registry-based build and solve
// -----------------------------------------------------------------------

fn coupled_spec_toml(dim: u8) -> String {
    format!(
        r#"
[problem]
name = "coupled test"

[mesh]
dimension = {dim}

[[fields]]
name = "u"
kind = "scalar"
space = "H1"
complex = false

[[fields]]
name = "v"
kind = "scalar"
space = "H1"
complex = false

[equation]
type = "coupled_elliptic_2x2"
alpha = 0.1
beta = 0.1

[[bcs]]
type = "dirichlet"
field = "u"
value = 1.0

[[bcs]]
type = "dirichlet"
field = "v"
value = 0.0

[solver]
linear = "block_pcg"
precond = "block_jacobi"
"#
    )
}

#[test]
fn coupled_2d_build_and_solve_converges() {
    // Anchor packs.
    let _ = spinoza_pack_poisson::PoissonPack;
    let _ = spinoza_pack_block2x2::Block2x2Pack;

    let registry = MethodRegistry::from_packs();
    let builder = registry
        .find_builder("coupled_elliptic_2x2", 2, "matrixfree")
        .expect("coupled builder should be found");

    let spec: CaseSpec = toml::from_str(&coupled_spec_toml(2)).unwrap();
    spec.validate().unwrap();

    let built = builder
        .build(&spec, "matrixfree", "block_jacobi", 16, 16, 1, &registry)
        .expect("build should succeed");

    assert_eq!(built.field_names.len(), 2);
    assert_eq!(built.field_sizes.len(), 2);
    let n0 = built.field_sizes[0];
    let n1 = built.field_sizes[1];
    assert_eq!(built.rhs.len(), n0 + n1);

    let result = solve_cg(
        built.operator.as_ref(),
        &built.rhs,
        1e-10,
        10_000,
        built.preconditioner.as_deref(),
    )
    .expect("CG should converge");

    assert!(
        result.residual_norm < 1e-8,
        "residual too large: {}",
        result.residual_norm
    );
    assert!(
        result.iterations < 5000,
        "too many iterations: {}",
        result.iterations
    );

    // Solution should be finite everywhere.
    for (i, &v) in result.solution.iter().enumerate() {
        assert!(v.is_finite(), "non-finite solution at index {i}");
    }
}

#[test]
fn coupled_3d_build_and_solve_converges() {
    let _ = spinoza_pack_poisson::PoissonPack;
    let _ = spinoza_pack_block2x2::Block2x2Pack;

    let registry = MethodRegistry::from_packs();
    let builder = registry
        .find_builder("coupled_elliptic_2x2", 3, "matrixfree")
        .expect("coupled builder should be found for 3D");

    let spec: CaseSpec = toml::from_str(&coupled_spec_toml(3)).unwrap();
    spec.validate().unwrap();

    let built = builder
        .build(&spec, "matrixfree", "block_jacobi", 8, 8, 8, &registry)
        .expect("3D build should succeed");

    let n = built.rhs.len();
    assert!(n > 0);

    let result = solve_cg(
        built.operator.as_ref(),
        &built.rhs,
        1e-10,
        20_000,
        built.preconditioner.as_deref(),
    )
    .expect("3D CG should converge");

    assert!(
        result.residual_norm < 1e-8,
        "3D residual too large: {}",
        result.residual_norm
    );

    for (i, &v) in result.solution.iter().enumerate() {
        assert!(v.is_finite(), "3D non-finite solution at index {i}");
    }
}

// -----------------------------------------------------------------------
// Pack conformance: packs register without panics
// -----------------------------------------------------------------------

#[test]
fn pack_discovery_finds_both_packs() {
    let _ = spinoza_pack_poisson::PoissonPack;
    let _ = spinoza_pack_block2x2::Block2x2Pack;

    let registry = MethodRegistry::from_packs();
    let names = registry.pack_names();
    assert!(
        names.contains(&"poisson"),
        "poisson pack missing: {names:?}"
    );
    assert!(
        names.contains(&"block2x2"),
        "block2x2 pack missing: {names:?}"
    );
}

#[test]
fn coupled_spec_audit_passes() {
    let _ = spinoza_pack_poisson::PoissonPack;
    let _ = spinoza_pack_block2x2::Block2x2Pack;

    let spec: CaseSpec = toml::from_str(&coupled_spec_toml(2)).unwrap();
    spec.validate().expect("coupled spec should validate");

    let plan = spinoza_core::CompiledPlan::from_spec(&spec);
    let registry = spinoza_core::CapabilityRegistry::default_registry();

    let missing = registry.missing(&plan.required);
    assert!(
        missing.is_empty(),
        "coupled spec has missing capabilities: {missing:?}"
    );
}

// -----------------------------------------------------------------------
// Conformance harness tests
// -----------------------------------------------------------------------

#[test]
fn poisson_pack_conforms() {
    let _ = spinoza_pack_poisson::PoissonPack;
    spinoza_core::conformance::assert_pack_conforms("poisson");
}

#[test]
fn block2x2_pack_conforms() {
    let _ = spinoza_pack_block2x2::Block2x2Pack;
    spinoza_core::conformance::assert_pack_conforms("block2x2");
}
