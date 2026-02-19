use spinoza_disc::{
    apply_constant_dirichlet_on_boundary, assemble_poisson_system, StructuredQuadMesh2D,
};
use spinoza_solve::solve_cg_jacobi;

#[test]
fn cg_solves_small_dirichlet_system() {
    let mesh = StructuredQuadMesh2D::unit_square(2, 2);
    let (mut a, mut b) = assemble_poisson_system(&mesh);
    apply_constant_dirichlet_on_boundary(&mesh, &mut a, &mut b, 0.0);
    let csr = a.to_csr();

    let result = solve_cg_jacobi(&csr, &b, 1e-12, 200).expect("cg should converge");
    assert!(result.residual_norm.is_finite());
    assert!(result.residual_norm < 1e-10);
}
