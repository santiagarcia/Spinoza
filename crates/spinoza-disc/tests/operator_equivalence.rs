use spinoza_core::LinearOperator;
use spinoza_disc::{
    apply_constant_dirichlet_on_boundary, apply_constant_dirichlet_on_boundary_3d,
    assemble_poisson_system, assemble_poisson_system_3d, PoissonLaplacianMF2D,
    PoissonLaplacianMF3D, StructuredHexMesh3D, StructuredQuadMesh2D,
};

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

#[test]
fn operator_apply_equivalence_2d() {
    let mesh = StructuredQuadMesh2D::unit_square(8, 8);

    let (mut builder, mut rhs) = assemble_poisson_system(&mesh);
    apply_constant_dirichlet_on_boundary(&mesh, &mut builder, &mut rhs, 0.0);
    let csr = builder.to_csr();

    let mf = PoissonLaplacianMF2D::with_constant_dirichlet_boundary(mesh.clone(), 0.0);

    let x = deterministic_vector(mesh.num_nodes());

    let mut y_csr = vec![0.0; mesh.num_nodes()];
    csr.apply(&x, &mut y_csr);

    let mut y_mf = vec![0.0; mesh.num_nodes()];
    mf.apply(&x, &mut y_mf);

    let rel = relative_error(&y_csr, &y_mf);
    assert!(rel < 1e-10, "2d operator apply mismatch rel={rel:e}");
}

#[test]
fn operator_apply_equivalence_3d() {
    let mesh = StructuredHexMesh3D::unit_cube(6, 6, 6);

    let (mut builder, mut rhs) = assemble_poisson_system_3d(&mesh);
    apply_constant_dirichlet_on_boundary_3d(&mesh, &mut builder, &mut rhs, 0.0);
    let csr = builder.to_csr();

    let mf = PoissonLaplacianMF3D::with_constant_dirichlet_boundary(mesh.clone(), 0.0);

    let x = deterministic_vector(mesh.num_nodes());

    let mut y_csr = vec![0.0; mesh.num_nodes()];
    csr.apply(&x, &mut y_csr);

    let mut y_mf = vec![0.0; mesh.num_nodes()];
    mf.apply(&x, &mut y_mf);

    let rel = relative_error(&y_csr, &y_mf);
    assert!(rel < 1e-10, "3d operator apply mismatch rel={rel:e}");
}
