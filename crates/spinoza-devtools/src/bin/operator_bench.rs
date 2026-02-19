use std::time::Instant;

use spinoza_core::{LinearOperator, Preconditioner};
use spinoza_disc::{
    apply_constant_dirichlet_on_boundary, apply_constant_dirichlet_on_boundary_3d,
    assemble_poisson_system, assemble_poisson_system_3d, PoissonLaplacianMF2D,
    PoissonLaplacianMF3D, StructuredHexMesh3D, StructuredQuadMesh2D,
};
use spinoza_solve::{
    solve_cg, solve_cg_jacobi, MultigridConfig, MultigridPreconditioner2D,
    MultigridPreconditioner3D,
};

fn deterministic_vector(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x0ddc_0ffe_e123_4567;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        out.push(2.0 * val - 1.0);
    }
    out
}

fn bench_operator(name: &str, op: &dyn LinearOperator, repeats: usize) {
    let n = op.size();
    let x = deterministic_vector(n);
    let mut y = vec![0.0; n];

    op.apply(&x, &mut y);

    let start = Instant::now();
    for _ in 0..repeats {
        op.apply(&x, &mut y);
    }
    let elapsed = start.elapsed().as_secs_f64();

    let avg_ms = 1000.0 * elapsed / repeats as f64;
    let dofs_per_sec = (n as f64 * repeats as f64) / elapsed.max(1e-12);

    println!(
        "{name}: size={n}, repeats={repeats}, avg_apply_ms={avg_ms:.6}, dofs_per_sec={dofs_per_sec:.3e}"
    );
}

fn bench_preconditioner(name: &str, precond: &dyn Preconditioner, size: usize, repeats: usize) {
    let r = deterministic_vector(size);
    let mut z = vec![0.0; size];

    precond.apply(&r, &mut z);

    let start = Instant::now();
    for _ in 0..repeats {
        precond.apply(&r, &mut z);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let avg_ms = 1000.0 * elapsed / repeats as f64;
    let dofs_per_sec = (size as f64 * repeats as f64) / elapsed.max(1e-12);

    println!(
        "{name}: size={size}, repeats={repeats}, avg_apply_ms={avg_ms:.6}, dofs_per_sec={dofs_per_sec:.3e}"
    );
}

fn bench_solve(name: &str, op: &dyn LinearOperator, rhs: &[f64], use_mg: bool) {
    let start = Instant::now();
    let result = if use_mg {
        if rhs.len() == (64 + 1) * (64 + 1) {
            let mg = MultigridPreconditioner2D::new(64, 64, 0.0, MultigridConfig::default())
                .expect("failed to build 2d mg preconditioner");
            solve_cg(op, rhs, 1e-8, 20_000, Some(&mg))
        } else {
            let mg = MultigridPreconditioner3D::new(16, 16, 16, 0.0, MultigridConfig::default())
                .expect("failed to build 3d mg preconditioner");
            solve_cg(op, rhs, 1e-8, 20_000, Some(&mg))
        }
    } else {
        solve_cg_jacobi(op, rhs, 1e-8, 20_000)
    }
    .expect("benchmark solve failed");
    let elapsed = start.elapsed().as_secs_f64();

    println!(
        "{name}: iterations={}, residual_norm={:.3e}, wall_s={:.6}",
        result.iterations, result.residual_norm, elapsed
    );
}

fn main() {
    let mesh2 = StructuredQuadMesh2D::unit_square(64, 64);
    let (mut a2, mut b2) = assemble_poisson_system(&mesh2);
    apply_constant_dirichlet_on_boundary(&mesh2, &mut a2, &mut b2, 0.0);
    let csr2 = a2.to_csr();
    let mf2 = PoissonLaplacianMF2D::with_constant_dirichlet_boundary(mesh2.clone(), 0.0);
    let mg2 = MultigridPreconditioner2D::new(64, 64, 0.0, MultigridConfig::default())
        .expect("failed to build 2d mg preconditioner");

    let mesh3 = StructuredHexMesh3D::unit_cube(16, 16, 16);
    let (mut a3, mut b3) = assemble_poisson_system_3d(&mesh3);
    apply_constant_dirichlet_on_boundary_3d(&mesh3, &mut a3, &mut b3, 0.0);
    let csr3 = a3.to_csr();
    let mf3 = PoissonLaplacianMF3D::with_constant_dirichlet_boundary(mesh3.clone(), 0.0);
    let mg3 = MultigridPreconditioner3D::new(16, 16, 16, 0.0, MultigridConfig::default())
        .expect("failed to build 3d mg preconditioner");

    println!("Poisson operator apply microbenchmark");
    bench_operator("2d assembled", &csr2, 30);
    bench_operator("2d matrixfree", &mf2, 30);
    bench_operator("3d assembled", &csr3, 10);
    bench_operator("3d matrixfree", &mf3, 10);

    println!("Poisson preconditioner apply microbenchmark");
    bench_preconditioner("2d mg apply", &mg2, mf2.size(), 30);
    bench_preconditioner("3d mg apply", &mg3, mf3.size(), 10);

    let rhs2 = deterministic_vector(mf2.size());
    let rhs3 = deterministic_vector(mf3.size());

    println!("Poisson full solve microbenchmark");
    bench_solve("2d matrixfree jacobi solve", &mf2, &rhs2, false);
    bench_solve("2d matrixfree mg solve", &mf2, &rhs2, true);
    bench_solve("3d matrixfree jacobi solve", &mf3, &rhs3, false);
    bench_solve("3d matrixfree mg solve", &mf3, &rhs3, true);
}
