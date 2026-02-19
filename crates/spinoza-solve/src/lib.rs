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

/// Solve a linear system using restarted GMRES.
///
/// Suitable for non-symmetric and indefinite systems. Falls back gracefully
/// for SPD systems where CG would be preferred.
pub fn solve_gmres(
    operator: &dyn LinearOperator,
    rhs: &[f64],
    tol: f64,
    max_iter: usize,
    restart: usize,
    preconditioner: Option<&dyn Preconditioner>,
) -> Result<LinearSolveResult, String> {
    if rhs.len() != operator.size() {
        return Err("rhs length does not match operator size".to_string());
    }

    let n = rhs.len();
    let m = restart.min(n).max(1);
    let mut x = vec![0.0; n];
    let mut total_iters = 0;
    let mut ax = vec![0.0; n];
    let mut z = vec![0.0; n];

    for _cycle in 0..(max_iter / m + 1) {
        // Compute residual r = b - A*x
        operator.apply(&x, &mut ax);
        let mut r: Vec<f64> = rhs.iter().zip(ax.iter()).map(|(b, a)| b - a).collect();

        let r_norm = norm2(&r);
        if r_norm <= tol {
            return Ok(LinearSolveResult {
                solution: x,
                residual_norm: r_norm,
                iterations: total_iters,
            });
        }

        // Arnoldi basis vectors V[0..m], Hessenberg matrix H[(m+1) x m]
        let mut v_basis: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        let scale = 1.0 / r_norm;
        let v0: Vec<f64> = r.iter().map(|&ri| ri * scale).collect();
        v_basis.push(v0);

        // g = r_norm * e_1
        let mut g = vec![0.0; m + 1];
        g[0] = r_norm;

        // Givens rotation coefficients
        let mut cs = vec![0.0; m];
        let mut sn = vec![0.0; m];

        // Upper Hessenberg matrix stored column-major: h[j] has entries 0..j+2
        let mut h: Vec<Vec<f64>> = Vec::with_capacity(m);

        let mut k = 0;
        while k < m && total_iters < max_iter {
            total_iters += 1;

            // w = A * M^{-1} * v_k  (right preconditioned)
            if let Some(pc) = preconditioner {
                pc.apply(&v_basis[k], &mut z);
                operator.apply(&z, &mut r);
            } else {
                operator.apply(&v_basis[k], &mut r);
            }

            // Modified Gram-Schmidt
            let mut hcol = vec![0.0; k + 2];
            for i in 0..=k {
                let d = dot(&r, &v_basis[i]);
                hcol[i] = d;
                for (rj, vj) in r.iter_mut().zip(v_basis[i].iter()) {
                    *rj -= d * vj;
                }
            }
            let w_norm = norm2(&r);
            hcol[k + 1] = w_norm;

            // Apply previous Givens rotations to hcol
            for i in 0..k {
                let temp = cs[i] * hcol[i] + sn[i] * hcol[i + 1];
                hcol[i + 1] = -sn[i] * hcol[i] + cs[i] * hcol[i + 1];
                hcol[i] = temp;
            }

            // Compute new Givens rotation
            let a_val = hcol[k];
            let b_val = hcol[k + 1];
            let r_val = (a_val * a_val + b_val * b_val).sqrt();
            if r_val > 1e-40 {
                cs[k] = a_val / r_val;
                sn[k] = b_val / r_val;
            } else {
                cs[k] = 1.0;
                sn[k] = 0.0;
            }
            hcol[k] = cs[k] * a_val + sn[k] * b_val;
            hcol[k + 1] = 0.0;

            // Apply rotation to g
            let temp = cs[k] * g[k] + sn[k] * g[k + 1];
            g[k + 1] = -sn[k] * g[k] + cs[k] * g[k + 1];
            g[k] = temp;

            h.push(hcol);

            let res_est = g[k + 1].abs();
            if res_est <= tol {
                k += 1;
                break;
            }

            // New basis vector
            if w_norm > 1e-40 {
                let scale = 1.0 / w_norm;
                let vk1: Vec<f64> = r.iter().map(|&ri| ri * scale).collect();
                v_basis.push(vk1);
            } else {
                k += 1;
                break;
            }

            k += 1;
        }

        // Back substitution to solve H * y = g
        let mut y = vec![0.0; k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for j in (i + 1)..k {
                sum -= h[j][i] * y[j];
            }
            if h[i][i].abs() > 1e-40 {
                y[i] = sum / h[i][i];
            }
        }

        // Update x: x += M^{-1} * V * y
        for (j, yj) in y.iter().enumerate() {
            if let Some(pc) = preconditioner {
                pc.apply(&v_basis[j], &mut z);
                axpy(*yj, &z, &mut x);
            } else {
                axpy(*yj, &v_basis[j], &mut x);
            }
        }

        // Check convergence
        operator.apply(&x, &mut ax);
        let res: f64 = rhs
            .iter()
            .zip(ax.iter())
            .map(|(b, a)| (b - a) * (b - a))
            .sum::<f64>()
            .sqrt();
        if res <= tol || total_iters >= max_iter {
            return Ok(LinearSolveResult {
                solution: x,
                residual_norm: res,
                iterations: total_iters,
            });
        }
    }

    // Final residual
    operator.apply(&x, &mut ax);
    let res: f64 = rhs
        .iter()
        .zip(ax.iter())
        .map(|(b, a)| (b - a) * (b - a))
        .sum::<f64>()
        .sqrt();
    Ok(LinearSolveResult {
        solution: x,
        residual_norm: res,
        iterations: total_iters,
    })
}
