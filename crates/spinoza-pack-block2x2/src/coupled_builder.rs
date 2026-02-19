//! Builder for 2×2 coupled elliptic systems.
//!
//! The PDE system is:
//! ```text
//!   -∇²u + α v = f
//!   -∇²v + β u = g
//! ```
//! which is assembled as the block system:
//! ```text
//!   [ L   αI ] [ u ]   [ f ]
//!   [ βI   L ] [ v ] = [ g ]
//! ```
//! where L is the Poisson Laplacian operator with Dirichlet BCs, and αI/βI
//! are scaled identity operators that are zero on boundary nodes.

use spinoza_core::{
    BuiltProblem, CaseSpec, LinearOperator, MethodRegistry, Preconditioner, ProblemBuilder,
    SpinozaError,
};
use spinoza_disc::{
    apply_dirichlet_rhs_correction_2d, apply_dirichlet_rhs_correction_3d, DirichletConstraints,
    PoissonLaplacianMF2D, PoissonLaplacianMF3D, StructuredHexMesh3D, StructuredQuadMesh2D,
};
use spinoza_solve::{
    JacobiPreconditioner, MultigridConfig, MultigridPreconditioner2D, MultigridPreconditioner3D,
};

use crate::block_operator::BlockOperator2x2;
use crate::block_precond::BlockJacobiPreconditioner;

// ---------------------------------------------------------------------------
// Scaled identity with BC zeroing
// ---------------------------------------------------------------------------

/// A diagonal operator `y[i] = scale * x[i]` for interior nodes and
/// `y[i] = 0` for constrained (boundary) nodes.
struct ScaledIdentityWithBC {
    scale: f64,
    n: usize,
    /// Bitset of boundary indices for O(1) lookup.
    is_boundary: Vec<bool>,
}

impl ScaledIdentityWithBC {
    fn new(n: usize, scale: f64, boundary_nodes: &[usize]) -> Self {
        let mut is_boundary = vec![false; n];
        for &idx in boundary_nodes {
            is_boundary[idx] = true;
        }
        Self {
            scale,
            n,
            is_boundary,
        }
    }
}

impl LinearOperator for ScaledIdentityWithBC {
    fn size(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);
        for ((yi, &xi), &is_bc) in y.iter_mut().zip(x.iter()).zip(self.is_boundary.iter()) {
            *yi = if is_bc { 0.0 } else { self.scale * xi };
        }
    }

    fn diagonal(&self, diag: &mut [f64]) -> Result<(), SpinozaError> {
        if diag.len() != self.n {
            return Err(SpinozaError::OperatorError(
                "diagonal slice length mismatch".to_string(),
            ));
        }
        for (d, &is_bc) in diag.iter_mut().zip(self.is_boundary.iter()) {
            *d = if is_bc { 0.0 } else { self.scale };
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CoupledEllipticBuilder
// ---------------------------------------------------------------------------

pub(crate) struct CoupledEllipticBuilder;

impl ProblemBuilder for CoupledEllipticBuilder {
    fn name(&self) -> &str {
        "CoupledEllipticBuilder"
    }

    fn equation_type(&self) -> &str {
        "coupled_elliptic_2x2"
    }

    fn supported_dimensions(&self) -> &[u8] {
        &[2, 3]
    }

    fn supported_backends(&self) -> &[&str] {
        &["matrixfree"]
    }

    fn build(
        &self,
        spec: &CaseSpec,
        _backend: &str,
        precond_hint: &str,
        nx: usize,
        ny: usize,
        nz: usize,
        _registry: &MethodRegistry,
    ) -> Result<BuiltProblem, String> {
        let alpha = spec.equation.alpha.unwrap_or(0.0);
        let beta = spec.equation.beta.unwrap_or(0.0);
        let dim = spec.mesh.dimension;

        // Collect per-field boundary values (default 0.0).
        let field_names: Vec<String> = spec.fields.iter().map(|f| f.name.clone()).collect();
        if field_names.len() != 2 {
            return Err(format!(
                "coupled_elliptic_2x2 requires exactly 2 fields, got {}",
                field_names.len()
            ));
        }
        let bc_value_u = spec
            .bcs
            .iter()
            .find(|bc| bc.field == field_names[0])
            .map(|bc| bc.value)
            .unwrap_or(0.0);
        let bc_value_v = spec
            .bcs
            .iter()
            .find(|bc| bc.field == field_names[1])
            .map(|bc| bc.value)
            .unwrap_or(0.0);

        match dim {
            2 => build_coupled_2d(
                nx,
                ny,
                alpha,
                beta,
                bc_value_u,
                bc_value_v,
                precond_hint,
                field_names,
            ),
            3 => build_coupled_3d(
                nx,
                ny,
                nz,
                alpha,
                beta,
                bc_value_u,
                bc_value_v,
                precond_hint,
                field_names,
            ),
            _ => Err(format!("unsupported dimension: {dim}")),
        }
    }
}

// ---------------------------------------------------------------------------
// 2D coupled system builder
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_coupled_2d(
    nx: usize,
    ny: usize,
    alpha: f64,
    beta: f64,
    bc_value_u: f64,
    bc_value_v: f64,
    precond_hint: &str,
    field_names: Vec<String>,
) -> Result<BuiltProblem, String> {
    let mesh = StructuredQuadMesh2D::unit_square(nx, ny);
    let ndofs = mesh.num_nodes();
    let boundary: Vec<usize> = mesh.boundary_nodes().into_iter().collect();

    // --- Block 0: Laplacian for u ---
    let bc_u: Vec<(usize, f64)> = boundary.iter().map(|&idx| (idx, bc_value_u)).collect();
    let constraints_u = DirichletConstraints::new(ndofs, &bc_u);
    let mut rhs_u = vec![0.0; ndofs];
    apply_dirichlet_rhs_correction_2d(&mesh, &mut rhs_u, &constraints_u);
    let a00 = PoissonLaplacianMF2D::new(mesh.clone(), constraints_u);

    // --- Block 1: Laplacian for v ---
    let bc_v: Vec<(usize, f64)> = boundary.iter().map(|&idx| (idx, bc_value_v)).collect();
    let constraints_v = DirichletConstraints::new(ndofs, &bc_v);
    let mut rhs_v = vec![0.0; ndofs];
    apply_dirichlet_rhs_correction_2d(&mesh, &mut rhs_v, &constraints_v);
    let a11 = PoissonLaplacianMF2D::new(mesh, constraints_v);

    // --- Off-diagonal coupling ---
    let a01 = ScaledIdentityWithBC::new(ndofs, alpha, &boundary);
    let a10 = ScaledIdentityWithBC::new(ndofs, beta, &boundary);

    // --- Block operator ---
    let block_op =
        BlockOperator2x2::new(Box::new(a00), Box::new(a01), Box::new(a10), Box::new(a11));

    // --- RHS: concatenated [rhs_u ; rhs_v] ---
    let mut rhs = rhs_u;
    rhs.extend_from_slice(&rhs_v);

    // --- Preconditioner: block Jacobi with per-block preconditioners ---
    let precond =
        build_block_precond_2d(ndofs, ndofs, nx, ny, bc_value_u, bc_value_v, precond_hint)?;

    Ok(BuiltProblem {
        operator: Box::new(block_op),
        rhs,
        preconditioner: Some(precond),
        field_names,
        field_sizes: vec![ndofs, ndofs],
    })
}

fn build_block_precond_2d(
    n0: usize,
    n1: usize,
    nx: usize,
    ny: usize,
    bc_value_u: f64,
    bc_value_v: f64,
    precond_hint: &str,
) -> Result<Box<dyn Preconditioner>, String> {
    match precond_hint {
        "block_jacobi" | "mg" => {
            // Use MG on each diagonal block when explicitly requested or as
            // default for block_jacobi hint.
            let mg_cfg = MultigridConfig::default();
            let p0 = MultigridPreconditioner2D::new(nx, ny, bc_value_u, mg_cfg)
                .map_err(|e| format!("MG preconditioner build failed for block 0: {e}"))?;
            let p1 = MultigridPreconditioner2D::new(nx, ny, bc_value_v, mg_cfg)
                .map_err(|e| format!("MG preconditioner build failed for block 1: {e}"))?;
            Ok(Box::new(BlockJacobiPreconditioner::new(
                Box::new(p0),
                Box::new(p1),
                n0,
                n1,
            )))
        }
        "jacobi" => {
            // Build Jacobi on each diagonal block.
            let mesh = StructuredQuadMesh2D::unit_square(nx, ny);
            let boundary: Vec<usize> = mesh.boundary_nodes().into_iter().collect();
            let bc_u: Vec<(usize, f64)> = boundary.iter().map(|&idx| (idx, bc_value_u)).collect();
            let bc_v: Vec<(usize, f64)> = boundary.iter().map(|&idx| (idx, bc_value_v)).collect();
            let c_u = DirichletConstraints::new(n0, &bc_u);
            let c_v = DirichletConstraints::new(n1, &bc_v);
            let op_u = PoissonLaplacianMF2D::new(mesh.clone(), c_u);
            let op_v = PoissonLaplacianMF2D::new(mesh, c_v);
            let p0 = JacobiPreconditioner::from_operator(&op_u)?;
            let p1 = JacobiPreconditioner::from_operator(&op_v)?;
            Ok(Box::new(BlockJacobiPreconditioner::new(
                Box::new(p0),
                Box::new(p1),
                n0,
                n1,
            )))
        }
        _ => Err(format!("unknown preconditioner hint: {precond_hint}")),
    }
}

// ---------------------------------------------------------------------------
// 3D coupled system builder
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_coupled_3d(
    nx: usize,
    ny: usize,
    nz: usize,
    alpha: f64,
    beta: f64,
    bc_value_u: f64,
    bc_value_v: f64,
    precond_hint: &str,
    field_names: Vec<String>,
) -> Result<BuiltProblem, String> {
    let mesh = StructuredHexMesh3D::unit_cube(nx, ny, nz);
    let ndofs = mesh.num_nodes();
    let boundary: Vec<usize> = mesh.boundary_nodes().into_iter().collect();

    // --- Block 0: Laplacian for u ---
    let bc_u: Vec<(usize, f64)> = boundary.iter().map(|&idx| (idx, bc_value_u)).collect();
    let constraints_u = DirichletConstraints::new(ndofs, &bc_u);
    let mut rhs_u = vec![0.0; ndofs];
    apply_dirichlet_rhs_correction_3d(&mesh, &mut rhs_u, &constraints_u);
    let a00 = PoissonLaplacianMF3D::new(mesh.clone(), constraints_u);

    // --- Block 1: Laplacian for v ---
    let bc_v: Vec<(usize, f64)> = boundary.iter().map(|&idx| (idx, bc_value_v)).collect();
    let constraints_v = DirichletConstraints::new(ndofs, &bc_v);
    let mut rhs_v = vec![0.0; ndofs];
    apply_dirichlet_rhs_correction_3d(&mesh, &mut rhs_v, &constraints_v);
    let a11 = PoissonLaplacianMF3D::new(mesh, constraints_v);

    // --- Off-diagonal coupling ---
    let a01 = ScaledIdentityWithBC::new(ndofs, alpha, &boundary);
    let a10 = ScaledIdentityWithBC::new(ndofs, beta, &boundary);

    // --- Block operator ---
    let block_op =
        BlockOperator2x2::new(Box::new(a00), Box::new(a01), Box::new(a10), Box::new(a11));

    // --- RHS ---
    let mut rhs = rhs_u;
    rhs.extend_from_slice(&rhs_v);

    // --- Preconditioner ---
    let precond = build_block_precond_3d(
        ndofs,
        ndofs,
        nx,
        ny,
        nz,
        bc_value_u,
        bc_value_v,
        precond_hint,
    )?;

    Ok(BuiltProblem {
        operator: Box::new(block_op),
        rhs,
        preconditioner: Some(precond),
        field_names,
        field_sizes: vec![ndofs, ndofs],
    })
}

#[allow(clippy::too_many_arguments)]
fn build_block_precond_3d(
    n0: usize,
    n1: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    bc_value_u: f64,
    bc_value_v: f64,
    precond_hint: &str,
) -> Result<Box<dyn Preconditioner>, String> {
    match precond_hint {
        "block_jacobi" | "mg" => {
            let mg_cfg = MultigridConfig::default();
            let p0 = MultigridPreconditioner3D::new(nx, ny, nz, bc_value_u, mg_cfg)
                .map_err(|e| format!("MG preconditioner build failed for block 0: {e}"))?;
            let p1 = MultigridPreconditioner3D::new(nx, ny, nz, bc_value_v, mg_cfg)
                .map_err(|e| format!("MG preconditioner build failed for block 1: {e}"))?;
            Ok(Box::new(BlockJacobiPreconditioner::new(
                Box::new(p0),
                Box::new(p1),
                n0,
                n1,
            )))
        }
        "jacobi" => {
            let mesh = StructuredHexMesh3D::unit_cube(nx, ny, nz);
            let boundary: Vec<usize> = mesh.boundary_nodes().into_iter().collect();
            let bc_u: Vec<(usize, f64)> = boundary.iter().map(|&idx| (idx, bc_value_u)).collect();
            let bc_v: Vec<(usize, f64)> = boundary.iter().map(|&idx| (idx, bc_value_v)).collect();
            let c_u = DirichletConstraints::new(n0, &bc_u);
            let c_v = DirichletConstraints::new(n1, &bc_v);
            let op_u = PoissonLaplacianMF3D::new(mesh.clone(), c_u);
            let op_v = PoissonLaplacianMF3D::new(mesh, c_v);
            let p0 = JacobiPreconditioner::from_operator(&op_u)?;
            let p1 = JacobiPreconditioner::from_operator(&op_v)?;
            Ok(Box::new(BlockJacobiPreconditioner::new(
                Box::new(p0),
                Box::new(p1),
                n0,
                n1,
            )))
        }
        _ => Err(format!("unknown preconditioner hint: {precond_hint}")),
    }
}
