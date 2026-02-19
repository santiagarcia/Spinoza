use std::f64::consts::PI;
use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;
use spinoza_disc::{
    apply_dirichlet_rhs_correction_2d, apply_dirichlet_rhs_correction_3d, apply_dirichlet_values,
    assemble_poisson_system_3d_with_source, assemble_poisson_system_with_source,
    DirichletConstraints, PoissonLaplacianMF2D, PoissonLaplacianMF3D, StructuredHexMesh3D,
    StructuredQuadMesh2D,
};
use spinoza_solve::solve_cg_jacobi;

fn manufactured_source(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
}

fn exact_solution(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

fn manufactured_source_3d(x: f64, y: f64, z: f64) -> f64 {
    3.0 * PI * PI * (PI * x).sin() * (PI * y).sin() * (PI * z).sin()
}

fn exact_solution_3d(x: f64, y: f64, z: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin() * (PI * z).sin()
}

fn solve_pair_2d(nx: usize, ny: usize) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let mesh = StructuredQuadMesh2D::unit_square(nx, ny);
    let (mut assembled_matrix, base_rhs) =
        assemble_poisson_system_with_source(&mesh, manufactured_source);

    let boundary_values: Vec<(usize, f64)> = mesh
        .boundary_nodes()
        .into_iter()
        .map(|idx| {
            let xy = mesh.nodes[idx];
            (idx, exact_solution(xy[0], xy[1]))
        })
        .collect();

    let mut rhs_assembled = base_rhs.clone();
    apply_dirichlet_values(&mut assembled_matrix, &mut rhs_assembled, &boundary_values);
    let csr = assembled_matrix.to_csr();
    let assembled = solve_cg_jacobi(&csr, &rhs_assembled, 1e-10, 20_000)
        .expect("assembled 2d solve should converge");

    let mut rhs_mf = base_rhs;
    let constraints = DirichletConstraints::new(mesh.num_nodes(), &boundary_values);
    apply_dirichlet_rhs_correction_2d(&mesh, &mut rhs_mf, &constraints);
    let op = PoissonLaplacianMF2D::new(mesh.clone(), constraints);
    let matrixfree =
        solve_cg_jacobi(&op, &rhs_mf, 1e-10, 20_000).expect("matrixfree 2d solve should converge");

    let mut err_a = 0.0;
    let mut err_m = 0.0;
    for (idx, xy) in mesh.nodes.iter().enumerate() {
        let ua = assembled.solution[idx] - exact_solution(xy[0], xy[1]);
        let um = matrixfree.solution[idx] - exact_solution(xy[0], xy[1]);
        err_a += ua * ua;
        err_m += um * um;
    }

    (
        assembled.solution,
        matrixfree.solution,
        (err_a / mesh.num_nodes() as f64).sqrt(),
        (err_m / mesh.num_nodes() as f64).sqrt(),
    )
}

fn solve_pair_3d(nx: usize, ny: usize, nz: usize) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let mesh = StructuredHexMesh3D::unit_cube(nx, ny, nz);
    let (mut assembled_matrix, base_rhs) =
        assemble_poisson_system_3d_with_source(&mesh, manufactured_source_3d);

    let boundary_values: Vec<(usize, f64)> = mesh
        .boundary_nodes()
        .into_iter()
        .map(|idx| {
            let xyz = mesh.nodes[idx];
            (idx, exact_solution_3d(xyz[0], xyz[1], xyz[2]))
        })
        .collect();

    let mut rhs_assembled = base_rhs.clone();
    apply_dirichlet_values(&mut assembled_matrix, &mut rhs_assembled, &boundary_values);
    let csr = assembled_matrix.to_csr();
    let assembled = solve_cg_jacobi(&csr, &rhs_assembled, 1e-9, 40_000)
        .expect("assembled 3d solve should converge");

    let mut rhs_mf = base_rhs;
    let constraints = DirichletConstraints::new(mesh.num_nodes(), &boundary_values);
    apply_dirichlet_rhs_correction_3d(&mesh, &mut rhs_mf, &constraints);
    let op = PoissonLaplacianMF3D::new(mesh.clone(), constraints);
    let matrixfree =
        solve_cg_jacobi(&op, &rhs_mf, 1e-9, 40_000).expect("matrixfree 3d solve should converge");

    let mut err_a = 0.0;
    let mut err_m = 0.0;
    for (idx, xyz) in mesh.nodes.iter().enumerate() {
        let ua = assembled.solution[idx] - exact_solution_3d(xyz[0], xyz[1], xyz[2]);
        let um = matrixfree.solution[idx] - exact_solution_3d(xyz[0], xyz[1], xyz[2]);
        err_a += ua * ua;
        err_m += um * um;
    }

    (
        assembled.solution,
        matrixfree.solution,
        (err_a / mesh.num_nodes() as f64).sqrt(),
        (err_m / mesh.num_nodes() as f64).sqrt(),
    )
}

fn relative_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = av - bv;
        num += d * d;
        den += av * av;
    }
    num.sqrt() / den.sqrt().max(1e-16)
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

#[test]
fn assembled_and_matrixfree_2d_solutions_match_and_refine() {
    let (a8, m8, e8a, e8m) = solve_pair_2d(8, 8);
    let (_a16, _m16, e16a, e16m) = solve_pair_2d(16, 16);

    assert!(relative_diff(&a8, &m8) < 1e-8);
    assert!(e16a < e8a);
    assert!(e16m < e8m);
}

#[test]
fn assembled_and_matrixfree_3d_solutions_match_and_refine() {
    let (a6, m6, e6a, e6m) = solve_pair_3d(6, 6, 6);
    let (_a10, _m10, e10a, e10m) = solve_pair_3d(10, 10, 10);

    assert!(relative_diff(&a6, &m6) < 1e-8);
    assert!(e10a < e6a);
    assert!(e10m < e6m);
}

#[test]
fn solve_command_writes_expected_json_shape() {
    let exe = env!("CARGO_BIN_EXE_spinoza-devtools");
    let out_path = std::env::temp_dir().join("spinoza_solve_regression_output.json");

    let status = Command::new(exe)
        .arg("solve")
        .arg("spec/poisson_minimal_ok.toml")
        .arg("--out")
        .arg(&out_path)
        .arg("--backend")
        .arg("assembled")
        .current_dir(workspace_root())
        .status()
        .expect("failed to execute solve command");

    assert!(status.success(), "solve command should exit successfully");

    let text = std::fs::read_to_string(&out_path).expect("solve output file should exist");
    let json: Value = serde_json::from_str(&text).expect("solve output must be valid json");

    assert_eq!(
        json.get("backend").and_then(Value::as_str),
        Some("assembled")
    );
    assert!(json.get("problem_name").is_some());
    assert!(json.get("dimension").is_some());
    assert!(json.get("field_names").is_some());
    assert!(json.get("solution_vector").is_some());
    assert!(json.get("residual_norm").is_some());
    assert!(json.get("iteration_count").is_some());
    // Traceability metadata.
    assert_eq!(
        json.get("selected_pack").and_then(Value::as_str),
        Some("poisson")
    );
    assert_eq!(
        json.get("selected_builder").and_then(Value::as_str),
        Some("PoissonBuilder")
    );
    // Solver should now come from linear_solvers pack.
    let selected_solver = json
        .get("selected_solver")
        .and_then(Value::as_str)
        .expect("selected_solver must be a string");
    assert!(
        selected_solver.starts_with("linear_solvers::"),
        "solver should come from linear_solvers pack, got: {selected_solver}"
    );

    let residual = json
        .get("residual_norm")
        .and_then(Value::as_f64)
        .expect("residual norm must be f64");
    assert!(residual.is_finite());
}

#[test]
fn solve_command_runs_for_3d_spec_matrixfree_and_has_expected_solution_size() {
    let exe = env!("CARGO_BIN_EXE_spinoza-devtools");
    let out_path = std::env::temp_dir().join("spinoza_solve_3d_regression_output.json");

    let status = Command::new(exe)
        .arg("solve")
        .arg("spec/poisson_3d_minimal_ok.toml")
        .arg("--out")
        .arg(&out_path)
        .arg("--backend")
        .arg("matrixfree")
        .current_dir(workspace_root())
        .status()
        .expect("failed to execute solve command for 3d spec");

    assert!(
        status.success(),
        "3d solve command should exit successfully"
    );

    let text = std::fs::read_to_string(&out_path).expect("solve output file should exist");
    let json: Value = serde_json::from_str(&text).expect("solve output must be valid json");

    assert_eq!(
        json.get("backend").and_then(Value::as_str),
        Some("matrixfree")
    );

    let residual = json
        .get("residual_norm")
        .and_then(Value::as_f64)
        .expect("residual norm must be f64");
    assert!(residual.is_finite());

    let solution = json
        .get("solution_vector")
        .and_then(Value::as_array)
        .expect("solution vector must be an array");
    assert_eq!(solution.len(), (8 + 1) * (8 + 1) * (8 + 1));
}

#[test]
fn solve_command_runs_with_matrixfree_mg_preconditioner() {
    let exe = env!("CARGO_BIN_EXE_spinoza-devtools");
    let out_path = std::env::temp_dir().join("spinoza_solve_mg_regression_output.json");

    let status = Command::new(exe)
        .arg("solve")
        .arg("spec/poisson_minimal_ok.toml")
        .arg("--out")
        .arg(&out_path)
        .arg("--backend")
        .arg("matrixfree")
        .arg("--precond")
        .arg("mg")
        .current_dir(workspace_root())
        .status()
        .expect("failed to execute solve command with mg preconditioner");

    assert!(
        status.success(),
        "matrixfree mg solve command should succeed"
    );

    let text = std::fs::read_to_string(&out_path).expect("solve output file should exist");
    let json: Value = serde_json::from_str(&text).expect("solve output must be valid json");

    assert_eq!(
        json.get("backend").and_then(Value::as_str),
        Some("matrixfree")
    );
    let residual = json
        .get("residual_norm")
        .and_then(Value::as_f64)
        .expect("residual norm must be f64");
    assert!(residual.is_finite());
}
