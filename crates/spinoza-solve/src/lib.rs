//! spinoza-solve: Minimal linear solver support.

mod multigrid;

pub use multigrid::{
    make_constant_boundary_constraints_2d, make_constant_boundary_constraints_3d, MultigridConfig,
    MultigridPreconditioner2D, MultigridPreconditioner3D,
};

use spinoza_core::{LinearOperator, Preconditioner};

#[derive(Debug, Clone)]
pub struct LinearSolveResult {
    pub solution: Vec<f64>,
    pub residual_norm: f64,
    pub iterations: usize,
}

pub struct JacobiPreconditioner {
    inv_diag: Vec<f64>,
}

impl JacobiPreconditioner {
    pub fn from_operator(operator: &dyn LinearOperator) -> Result<Self, String> {
        let mut diag = vec![0.0; operator.size()];
        operator
            .diagonal(&mut diag)
            .map_err(|e| format!("failed to extract diagonal for jacobi: {e}"))?;
        if diag.iter().any(|d| d.abs() < 1e-14) {
            return Err("jacobi preconditioner encountered zero diagonal".to_string());
        }
        let inv_diag = diag.into_iter().map(|d| 1.0 / d).collect();
        Ok(Self { inv_diag })
    }
}

impl Preconditioner for JacobiPreconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        assert_eq!(r.len(), self.inv_diag.len());
        assert_eq!(z.len(), self.inv_diag.len());
        for i in 0..r.len() {
            z[i] = self.inv_diag[i] * r[i];
        }
    }
}

pub fn solve_cg(
    operator: &dyn LinearOperator,
    rhs: &[f64],
    tol: f64,
    max_iter: usize,
    preconditioner: Option<&dyn Preconditioner>,
) -> Result<LinearSolveResult, String> {
    if rhs.len() != operator.size() {
        return Err("rhs length does not match matrix size".to_string());
    }

    let n = rhs.len();
    let mut x = vec![0.0; n];
    let mut r = rhs.to_vec();
    let mut z = vec![0.0; n];

    if let Some(precond) = preconditioner {
        precond.apply(&r, &mut z);
    } else {
        z.copy_from_slice(&r);
    }
    if dot(&r, &z) <= 0.0 {
        z.copy_from_slice(&r);
    }

    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);
    let mut res_norm = norm2(&r);

    if res_norm <= tol {
        return Ok(LinearSolveResult {
            solution: x,
            residual_norm: res_norm,
            iterations: 0,
        });
    }

    let mut ax = vec![0.0; n];
    let mut ap = vec![0.0; n];
    for k in 0..max_iter {
        operator.apply(&p, &mut ap);
        let mut denom = dot(&p, &ap);
        if denom.abs() < 1e-40 || !denom.is_finite() {
            // Recompute true residual to avoid accumulated drift.
            operator.apply(&x, &mut ax);
            for i in 0..n {
                r[i] = rhs[i] - ax[i];
            }

            res_norm = norm2(&r);
            if res_norm <= tol {
                return Ok(LinearSolveResult {
                    solution: x,
                    residual_norm: res_norm,
                    iterations: k,
                });
            }

            // Hard restart: rebuild preconditioned direction from true residual.
            if let Some(precond) = preconditioner {
                precond.apply(&r, &mut z);
            } else {
                z.copy_from_slice(&r);
            }
            let rz_check = dot(&r, &z);
            if rz_check <= 0.0 || !rz_check.is_finite() {
                z.copy_from_slice(&r);
            }

            p.copy_from_slice(&z);
            rz_old = dot(&r, &z);

            operator.apply(&p, &mut ap);
            denom = dot(&p, &ap);
            if denom.abs() < 1e-40 || !denom.is_finite() {
                return Err("cg breakdown due to near zero denominator".to_string());
            }
        }

        let alpha = rz_old / denom;

        axpy(alpha, &p, &mut x);
        axpy(-alpha, &ap, &mut r);

        res_norm = norm2(&r);
        if res_norm <= tol {
            return Ok(LinearSolveResult {
                solution: x,
                residual_norm: res_norm,
                iterations: k + 1,
            });
        }

        if let Some(precond) = preconditioner {
            precond.apply(&r, &mut z);
        } else {
            z.copy_from_slice(&r);
        }
        let rz_new = dot(&r, &z);
        if rz_new <= 0.0 || !rz_new.is_finite() {
            z.copy_from_slice(&r);
        }
        let rz_new = dot(&r, &z);
        let beta = if rz_old.abs() > 0.0 {
            rz_new / rz_old
        } else {
            0.0
        };

        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
    }

    Ok(LinearSolveResult {
        solution: x,
        residual_norm: res_norm,
        iterations: max_iter,
    })
}

pub fn solve_cg_jacobi(
    operator: &dyn LinearOperator,
    rhs: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<LinearSolveResult, String> {
    let jacobi = JacobiPreconditioner::from_operator(operator)?;
    solve_cg(operator, rhs, tol, max_iter, Some(&jacobi))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}
