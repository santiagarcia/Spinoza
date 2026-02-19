use spinoza_disc::{
    apply_dirichlet_rhs_correction_2d, apply_dirichlet_rhs_correction_3d,
    assemble_poisson_system_3d_with_source, assemble_poisson_system_with_source,
    PoissonLaplacianMF2D, PoissonLaplacianMF3D, StructuredHexMesh3D, StructuredQuadMesh2D,
};
use spinoza_solve::{
    make_constant_boundary_constraints_2d, make_constant_boundary_constraints_3d, solve_cg,
    solve_cg_jacobi, MultigridConfig, MultigridPreconditioner2D, MultigridPreconditioner3D,
};

/// Constant source f=1 excites all eigenmodes, ensuring Jacobi-PCG needs
/// many iterations and MG-PCG can show a clear iteration reduction.
fn solve_pair_2d(nx: usize, ny: usize) -> (usize, usize) {
    let mesh = StructuredQuadMesh2D::unit_square(nx, ny);
    let (_, base_rhs) = assemble_poisson_system_with_source(&mesh, |_, _| 1.0);

    let constraints = make_constant_boundary_constraints_2d(&mesh, 0.0);

    let mut rhs = base_rhs;
    apply_dirichlet_rhs_correction_2d(&mesh, &mut rhs, &constraints);

    let op = PoissonLaplacianMF2D::new(mesh, constraints);

    let jac = solve_cg_jacobi(&op, &rhs, 1e-10, 50_000).expect("jacobi cg 2d should converge");

    let mg = MultigridPreconditioner2D::new(nx, ny, 0.0, MultigridConfig::default())
        .expect("mg preconditioner 2d should build");
    let mg_result =
        solve_cg(&op, &rhs, 1e-10, 50_000, Some(&mg)).expect("mg cg 2d should converge");

    // Solutions must match to high relative accuracy.
    let diff: f64 = jac
        .solution
        .iter()
        .zip(mg_result.solution.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = jac.solution.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!(
        diff / norm.max(1e-15) < 1e-4,
        "jacobi and mg solutions should match"
    );

    (jac.iterations, mg_result.iterations)
}

fn solve_pair_3d(nx: usize, ny: usize, nz: usize) -> (usize, usize) {
    let mesh = StructuredHexMesh3D::unit_cube(nx, ny, nz);
    let (_, base_rhs) = assemble_poisson_system_3d_with_source(&mesh, |_, _, _| 1.0);

    let constraints = make_constant_boundary_constraints_3d(&mesh, 0.0);

    let mut rhs = base_rhs;
    apply_dirichlet_rhs_correction_3d(&mesh, &mut rhs, &constraints);

    let op = PoissonLaplacianMF3D::new(mesh, constraints);

    let jac = solve_cg_jacobi(&op, &rhs, 1e-9, 100_000).expect("jacobi cg 3d should converge");

    let mg = MultigridPreconditioner3D::new(nx, ny, nz, 0.0, MultigridConfig::default())
        .expect("mg preconditioner 3d should build");
    let mg_result =
        solve_cg(&op, &rhs, 1e-9, 100_000, Some(&mg)).expect("mg cg 3d should converge");

    let diff: f64 = jac
        .solution
        .iter()
        .zip(mg_result.solution.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = jac.solution.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!(
        diff / norm.max(1e-15) < 1e-3,
        "jacobi and mg solutions should match"
    );

    (jac.iterations, mg_result.iterations)
}

#[test]
fn mg_reduces_iterations_in_2d_and_solution_matches() {
    // 64x64 gives 6 MG levels (64→32→16→8→4→2) and enough DOFs
    // that Jacobi-PCG needs ~50-100 iterations while MG-PCG needs ~10.
    let (jac_it, mg_it) = solve_pair_2d(64, 64);
    assert!(
        mg_it < jac_it,
        "expected mg to reduce iterations: jac={jac_it}, mg={mg_it}"
    );
    assert!(
        mg_it * 2 < jac_it,
        "expected meaningful margin (>2x speedup): jac={jac_it}, mg={mg_it}"
    );
}

#[test]
fn mg_reduces_iterations_in_3d_and_solution_matches() {
    // 16x16x16 gives 4 MG levels (16→8→4→2).
    let (jac_it, mg_it) = solve_pair_3d(16, 16, 16);
    assert!(
        mg_it < jac_it,
        "expected mg to reduce iterations: jac={jac_it}, mg={mg_it}"
    );
    assert!(
        mg_it * 2 < jac_it,
        "expected meaningful margin (>2x speedup): jac={jac_it}, mg={mg_it}"
    );
}
