//! spinoza-pack-elasticity: Method pack for linear isotropic elasticity.
//!
//! Supports plane strain (2D) and full 3D linear elasticity with:
//! - Q1 (bilinear quad / trilinear hex) elements
//! - H1 vector function space
//! - Assembled CSR operator
//! - Matrix-free operator
//! - Jacobi preconditioner
//! - ElasticityBuilder that composes the above from the registry
//!
//! **Note:** Solver factories (CG, GMRES) are provided by
//! `spinoza-pack-linear-solvers`, not by physics packs.
//!
//! DOF numbering: interleaved [ux0, uy0, ux1, uy1, ...] for 2D,
//! [ux0, uy0, uz0, ux1, uy1, uz1, ...] for 3D.
//!
//! Adding this crate as a dependency is sufficient for Spinoza to discover
//! it at link time via the inventory mechanism.

use spinoza_core::{
    BuiltOperator, BuiltProblem, Capability, CaseSpec, ComponentLookup, LinearOperator,
    MethodPack, MethodRegistry, OperatorFactory, OperatorKey, Preconditioner,
    PreconditionerFactory, PreconditionerKey, ProblemBuilder, SpinozaError,
};
use spinoza_disc::{
    SparseMatrixBuilder, StructuredHexMesh3D, StructuredQuadMesh2D,
};
use spinoza_solve::JacobiPreconditioner;

// ---------------------------------------------------------------------------
// Pack definition
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct ElasticityPack;

static CAPABILITIES: &[Capability] = &[
    Capability::SpaceH1Vector,
    Capability::OperatorElasticity,
    Capability::BcDirichlet,
    Capability::PrecondJacobi,
];

impl MethodPack for ElasticityPack {
    fn name(&self) -> &'static str {
        "elasticity"
    }
    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
    fn capabilities(&self) -> &'static [Capability] {
        CAPABILITIES
    }
    fn register(&self, registry: &mut MethodRegistry) {
        registry.register_builder(Box::new(ElasticityBuilder));
        registry.register_operator_factory(Box::new(ElasticityOperatorFactory));
        registry.register_preconditioner_factory(Box::new(ElasticityJacobiPrecondFactory));
    }
}

spinoza_core::spinoza_register_pack!(ElasticityPack);

// ---------------------------------------------------------------------------
// Material constants
// ---------------------------------------------------------------------------

/// Plane strain constitutive matrix (3×3 Voigt: [σ_xx, σ_yy, σ_xy]).
fn plane_strain_d(young: f64, nu: f64) -> [[f64; 3]; 3] {
    let factor = young / ((1.0 + nu) * (1.0 - 2.0 * nu));
    [
        [factor * (1.0 - nu), factor * nu, 0.0],
        [factor * nu, factor * (1.0 - nu), 0.0],
        [0.0, 0.0, factor * 0.5 * (1.0 - 2.0 * nu)],
    ]
}

/// 3D constitutive matrix (6×6 Voigt: [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz]).
fn full_3d_d(young: f64, nu: f64) -> [[f64; 6]; 6] {
    let factor = young / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let a = factor * (1.0 - nu);
    let b = factor * nu;
    let c = factor * 0.5 * (1.0 - 2.0 * nu);
    [
        [a, b, b, 0.0, 0.0, 0.0],
        [b, a, b, 0.0, 0.0, 0.0],
        [b, b, a, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, c],
    ]
}

// ---------------------------------------------------------------------------
// 2D Q1 shape functions (same convention as poisson)
// ---------------------------------------------------------------------------

fn dshape_q1_dxi(_xi: f64, eta: f64) -> [f64; 4] {
    [
        -0.25 * (1.0 - eta),
        0.25 * (1.0 - eta),
        0.25 * (1.0 + eta),
        -0.25 * (1.0 + eta),
    ]
}

fn dshape_q1_deta(xi: f64, _eta: f64) -> [f64; 4] {
    [
        -0.25 * (1.0 - xi),
        -0.25 * (1.0 + xi),
        0.25 * (1.0 + xi),
        0.25 * (1.0 - xi),
    ]
}

// ---------------------------------------------------------------------------
// 3D Q1 hex shape functions
// ---------------------------------------------------------------------------

const HEX_SIGNS: [(f64, f64, f64); 8] = [
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
];

fn dshape_hex_dxi(_xi: f64, eta: f64, zeta: f64) -> [f64; 8] {
    let mut out = [0.0; 8];
    for a in 0..8 {
        let s = HEX_SIGNS[a];
        out[a] = 0.125 * s.0 * (1.0 + s.1 * eta) * (1.0 + s.2 * zeta);
    }
    out
}

fn dshape_hex_deta(xi: f64, _eta: f64, zeta: f64) -> [f64; 8] {
    let mut out = [0.0; 8];
    for a in 0..8 {
        let s = HEX_SIGNS[a];
        out[a] = 0.125 * (1.0 + s.0 * xi) * s.1 * (1.0 + s.2 * zeta);
    }
    out
}

fn dshape_hex_dzeta(xi: f64, eta: f64, _zeta: f64) -> [f64; 8] {
    let mut out = [0.0; 8];
    for a in 0..8 {
        let s = HEX_SIGNS[a];
        out[a] = 0.125 * (1.0 + s.0 * xi) * (1.0 + s.1 * eta) * s.2;
    }
    out
}

// ---------------------------------------------------------------------------
// 2D element stiffness: plane strain elasticity
// ---------------------------------------------------------------------------

/// Compute element stiffness matrix for plane strain Q1 quad.
///
/// DOFs per element: 8 (interleaved [ux0, uy0, ux1, uy1, ux2, uy2, ux3, uy3]).
fn element_stiffness_2d(
    coords: [[f64; 2]; 4],
    d_mat: &[[f64; 3]; 3],
) -> [[f64; 8]; 8] {
    let g = 1.0 / 3.0_f64.sqrt();
    let gauss = [(-g, 1.0), (g, 1.0)];

    let mut ke = [[0.0; 8]; 8];

    for (xi, wx) in gauss {
        for (eta, wy) in gauss {
            let dndxi = dshape_q1_dxi(xi, eta);
            let dndeta = dshape_q1_deta(xi, eta);

            // Jacobian
            let mut j11 = 0.0;
            let mut j12 = 0.0;
            let mut j21 = 0.0;
            let mut j22 = 0.0;
            for a in 0..4 {
                j11 += dndxi[a] * coords[a][0];
                j12 += dndeta[a] * coords[a][0];
                j21 += dndxi[a] * coords[a][1];
                j22 += dndeta[a] * coords[a][1];
            }
            let detj = j11 * j22 - j12 * j21;
            let invj = [[j22 / detj, -j12 / detj], [-j21 / detj, j11 / detj]];

            // Physical gradients dN/dx, dN/dy
            let mut grad = [[0.0; 2]; 4];
            for a in 0..4 {
                grad[a][0] = invj[0][0] * dndxi[a] + invj[0][1] * dndeta[a];
                grad[a][1] = invj[1][0] * dndxi[a] + invj[1][1] * dndeta[a];
            }

            let weight = wx * wy * detj;

            // B^T * D * B integration.
            // B_a is 3×2: [[dN_a/dx, 0], [0, dN_a/dy], [dN_a/dy, dN_a/dx]]
            // ke(2a+i, 2b+j) += sum over Voigt components
            for a in 0..4 {
                // Build B_a (3×2)
                let b_a = [
                    [grad[a][0], 0.0],
                    [0.0, grad[a][1]],
                    [grad[a][1], grad[a][0]],
                ];
                for b in 0..4 {
                    let b_b = [
                        [grad[b][0], 0.0],
                        [0.0, grad[b][1]],
                        [grad[b][1], grad[b][0]],
                    ];
                    // ke_ab = B_a^T * D * B_b  (2×2 block)
                    for i in 0..2 {
                        for j in 0..2 {
                            let mut val = 0.0;
                            for p in 0..3 {
                                for q in 0..3 {
                                    val += b_a[p][i] * d_mat[p][q] * b_b[q][j];
                                }
                            }
                            ke[2 * a + i][2 * b + j] += val * weight;
                        }
                    }
                }
            }
        }
    }
    ke
}

// ---------------------------------------------------------------------------
// 3D element stiffness: full 3D elasticity
// ---------------------------------------------------------------------------

fn determinant_3x3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inverse_3x3(m: [[f64; 3]; 3], det: f64) -> [[f64; 3]; 3] {
    let inv_det = 1.0 / det;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}

/// Compute element stiffness matrix for 3D linear elasticity with hex Q1.
///
/// DOFs per element: 24 (interleaved [ux0, uy0, uz0, ux1, ...]).
fn element_stiffness_3d(
    coords: [[f64; 3]; 8],
    d_mat: &[[f64; 6]; 6],
) -> [[f64; 24]; 24] {
    let g = 1.0 / 3.0_f64.sqrt();
    let gauss = [(-g, 1.0), (g, 1.0)];

    let mut ke = [[0.0; 24]; 24];

    for (xi, wx) in gauss {
        for (eta, wy) in gauss {
            for (zeta, wz) in gauss {
                let dndxi = dshape_hex_dxi(xi, eta, zeta);
                let dndeta = dshape_hex_deta(xi, eta, zeta);
                let dndzeta = dshape_hex_dzeta(xi, eta, zeta);

                // Jacobian
                let mut j = [[0.0; 3]; 3];
                for a in 0..8 {
                    j[0][0] += dndxi[a] * coords[a][0];
                    j[0][1] += dndeta[a] * coords[a][0];
                    j[0][2] += dndzeta[a] * coords[a][0];
                    j[1][0] += dndxi[a] * coords[a][1];
                    j[1][1] += dndeta[a] * coords[a][1];
                    j[1][2] += dndzeta[a] * coords[a][1];
                    j[2][0] += dndxi[a] * coords[a][2];
                    j[2][1] += dndeta[a] * coords[a][2];
                    j[2][2] += dndzeta[a] * coords[a][2];
                }

                let detj = determinant_3x3(j);
                let invj = inverse_3x3(j, detj);

                // Physical gradients
                let mut grad = [[0.0; 3]; 8];
                for a in 0..8 {
                    grad[a][0] = invj[0][0] * dndxi[a]
                        + invj[0][1] * dndeta[a]
                        + invj[0][2] * dndzeta[a];
                    grad[a][1] = invj[1][0] * dndxi[a]
                        + invj[1][1] * dndeta[a]
                        + invj[1][2] * dndzeta[a];
                    grad[a][2] = invj[2][0] * dndxi[a]
                        + invj[2][1] * dndeta[a]
                        + invj[2][2] * dndzeta[a];
                }

                let weight = wx * wy * wz * detj;

                // B^T * D * B integration.
                // B_a is 6×3:
                // [[dN_a/dx,       0,       0],
                //  [      0, dN_a/dy,       0],
                //  [      0,       0, dN_a/dz],
                //  [dN_a/dy, dN_a/dx,       0],
                //  [      0, dN_a/dz, dN_a/dy],
                //  [dN_a/dz,       0, dN_a/dx]]
                for a in 0..8 {
                    let gx = grad[a][0];
                    let gy = grad[a][1];
                    let gz = grad[a][2];
                    let b_a: [[f64; 3]; 6] = [
                        [gx, 0.0, 0.0],
                        [0.0, gy, 0.0],
                        [0.0, 0.0, gz],
                        [gy, gx, 0.0],
                        [0.0, gz, gy],
                        [gz, 0.0, gx],
                    ];
                    for b in 0..8 {
                        let bx = grad[b][0];
                        let by = grad[b][1];
                        let bz = grad[b][2];
                        let b_b: [[f64; 3]; 6] = [
                            [bx, 0.0, 0.0],
                            [0.0, by, 0.0],
                            [0.0, 0.0, bz],
                            [by, bx, 0.0],
                            [0.0, bz, by],
                            [bz, 0.0, bx],
                        ];
                        // ke_ab = B_a^T * D * B_b  (3×3 block)
                        for i in 0..3 {
                            for jj in 0..3 {
                                let mut val = 0.0;
                                for p in 0..6 {
                                    for q in 0..6 {
                                        val += b_a[p][i] * d_mat[p][q] * b_b[q][jj];
                                    }
                                }
                                ke[3 * a + i][3 * b + jj] += val * weight;
                            }
                        }
                    }
                }
            }
        }
    }
    ke
}

// ---------------------------------------------------------------------------
// Assembly routines
// ---------------------------------------------------------------------------

/// Assemble 2D plane strain elasticity stiffness matrix and zero RHS.
fn assemble_elasticity_2d(
    mesh: &StructuredQuadMesh2D,
    young: f64,
    nu: f64,
) -> (SparseMatrixBuilder, Vec<f64>) {
    let ndofs = mesh.num_nodes() * 2; // 2 DOFs per node
    let mut matrix = SparseMatrixBuilder::new(ndofs, ndofs);
    let rhs = vec![0.0; ndofs];
    let d_mat = plane_strain_d(young, nu);

    for element in &mesh.elements {
        let coords = [
            mesh.nodes[element[0]],
            mesh.nodes[element[1]],
            mesh.nodes[element[2]],
            mesh.nodes[element[3]],
        ];

        let ke = element_stiffness_2d(coords, &d_mat);

        // Scatter: global DOF for node a, component i = 2 * element[a] + i
        for a in 0..4 {
            for i in 0..2 {
                let row = 2 * element[a] + i;
                for b in 0..4 {
                    for j in 0..2 {
                        let col = 2 * element[b] + j;
                        matrix.add_entry(row, col, ke[2 * a + i][2 * b + j]);
                    }
                }
            }
        }
    }

    (matrix, rhs)
}

/// Assemble 3D elasticity stiffness matrix and zero RHS.
fn assemble_elasticity_3d(
    mesh: &StructuredHexMesh3D,
    young: f64,
    nu: f64,
) -> (SparseMatrixBuilder, Vec<f64>) {
    let ndofs = mesh.num_nodes() * 3; // 3 DOFs per node
    let mut matrix = SparseMatrixBuilder::new(ndofs, ndofs);
    let rhs = vec![0.0; ndofs];
    let d_mat = full_3d_d(young, nu);

    for element in &mesh.elements {
        let coords = [
            mesh.nodes[element[0]],
            mesh.nodes[element[1]],
            mesh.nodes[element[2]],
            mesh.nodes[element[3]],
            mesh.nodes[element[4]],
            mesh.nodes[element[5]],
            mesh.nodes[element[6]],
            mesh.nodes[element[7]],
        ];

        let ke = element_stiffness_3d(coords, &d_mat);

        for a in 0..8 {
            for i in 0..3 {
                let row = 3 * element[a] + i;
                for b in 0..8 {
                    for j in 0..3 {
                        let col = 3 * element[b] + j;
                        matrix.add_entry(row, col, ke[3 * a + i][3 * b + j]);
                    }
                }
            }
        }
    }

    (matrix, rhs)
}

// ---------------------------------------------------------------------------
// Dirichlet BC for vector fields
// ---------------------------------------------------------------------------

/// Apply zero Dirichlet on all displacement DOFs at boundary nodes (2D).
fn apply_dirichlet_elasticity_2d(
    mesh: &StructuredQuadMesh2D,
    matrix: &mut SparseMatrixBuilder,
    rhs: &mut [f64],
    value: f64,
) {
    let boundary_nodes = mesh.boundary_nodes();
    let mut dof_values = Vec::new();
    for &node in &boundary_nodes {
        for c in 0..2 {
            dof_values.push((2 * node + c, value));
        }
    }
    apply_dirichlet_values(matrix, rhs, &dof_values);
}

/// Apply zero Dirichlet on all displacement DOFs at boundary nodes (3D).
fn apply_dirichlet_elasticity_3d(
    mesh: &StructuredHexMesh3D,
    matrix: &mut SparseMatrixBuilder,
    rhs: &mut [f64],
    value: f64,
) {
    let boundary_nodes = mesh.boundary_nodes();
    let mut dof_values = Vec::new();
    for &node in &boundary_nodes {
        for c in 0..3 {
            dof_values.push((3 * node + c, value));
        }
    }
    apply_dirichlet_values(matrix, rhs, &dof_values);
}

/// Apply Dirichlet conditions by row/column elimination.
fn apply_dirichlet_values(
    matrix: &mut SparseMatrixBuilder,
    rhs: &mut [f64],
    node_values: &[(usize, f64)],
) {
    for &(i, value) in node_values {
        for (row, rhs_row) in rhs.iter_mut().enumerate().take(matrix.nrows()) {
            if row == i {
                continue;
            }
            let a_ri = matrix.get(row, i);
            if a_ri != 0.0 {
                *rhs_row -= a_ri * value;
                matrix.set(row, i, 0.0);
            }
        }

        {
            let row_map = &mut matrix.rows_mut()[i];
            row_map.clear();
            row_map.insert(i, 1.0);
        }
        rhs[i] = value;
    }
}

// ---------------------------------------------------------------------------
// Matrix-free elasticity operator (2D)
// ---------------------------------------------------------------------------

/// Matrix-free 2D plane strain elasticity operator.
///
/// Stores the mesh and material, and applies element-by-element without
/// assembling a global matrix. Dirichlet DOFs are zeroed via constraints.
pub struct ElasticityMF2D {
    mesh: StructuredQuadMesh2D,
    d_mat: [[f64; 3]; 3],
    boundary_dofs: Vec<bool>,
}

impl ElasticityMF2D {
    pub fn new(mesh: StructuredQuadMesh2D, young: f64, nu: f64) -> Self {
        let ndofs = mesh.num_nodes() * 2;
        let mut boundary_dofs = vec![false; ndofs];
        for &node in &mesh.boundary_nodes() {
            boundary_dofs[2 * node] = true;
            boundary_dofs[2 * node + 1] = true;
        }
        let d_mat = plane_strain_d(young, nu);
        Self {
            mesh,
            d_mat,
            boundary_dofs,
        }
    }
}

impl LinearOperator for ElasticityMF2D {
    fn size(&self) -> usize {
        self.mesh.num_nodes() * 2
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let n = self.size();
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);

        for yi in y.iter_mut() {
            *yi = 0.0;
        }

        for element in &self.mesh.elements {
            let coords = [
                self.mesh.nodes[element[0]],
                self.mesh.nodes[element[1]],
                self.mesh.nodes[element[2]],
                self.mesh.nodes[element[3]],
            ];

            let ke = element_stiffness_2d(coords, &self.d_mat);

            // Scatter-gather
            for a in 0..4 {
                for i in 0..2 {
                    let row = 2 * element[a] + i;
                    if self.boundary_dofs[row] {
                        continue;
                    }
                    for b in 0..4 {
                        for j in 0..2 {
                            let col = 2 * element[b] + j;
                            y[row] += ke[2 * a + i][2 * b + j] * x[col];
                        }
                    }
                }
            }
        }

        // Boundary DOFs: identity action
        for (idx, &is_boundary) in self.boundary_dofs.iter().enumerate() {
            if is_boundary {
                y[idx] = x[idx];
            }
        }
    }

    fn diagonal(&self, diag: &mut [f64]) -> Result<(), SpinozaError> {
        let n = self.size();
        if diag.len() != n {
            return Err(SpinozaError::OperatorError(
                "diagonal slice length mismatch".to_string(),
            ));
        }

        for d in diag.iter_mut() {
            *d = 0.0;
        }

        for element in &self.mesh.elements {
            let coords = [
                self.mesh.nodes[element[0]],
                self.mesh.nodes[element[1]],
                self.mesh.nodes[element[2]],
                self.mesh.nodes[element[3]],
            ];

            let ke = element_stiffness_2d(coords, &self.d_mat);

            for a in 0..4 {
                for i in 0..2 {
                    let row = 2 * element[a] + i;
                    if !self.boundary_dofs[row] {
                        diag[row] += ke[2 * a + i][2 * a + i];
                    }
                }
            }
        }

        for (idx, &is_boundary) in self.boundary_dofs.iter().enumerate() {
            if is_boundary {
                diag[idx] = 1.0;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Matrix-free elasticity operator (3D)
// ---------------------------------------------------------------------------

/// Matrix-free 3D elasticity operator.
pub struct ElasticityMF3D {
    mesh: StructuredHexMesh3D,
    d_mat: [[f64; 6]; 6],
    boundary_dofs: Vec<bool>,
}

impl ElasticityMF3D {
    pub fn new(mesh: StructuredHexMesh3D, young: f64, nu: f64) -> Self {
        let ndofs = mesh.num_nodes() * 3;
        let mut boundary_dofs = vec![false; ndofs];
        for &node in &mesh.boundary_nodes() {
            boundary_dofs[3 * node] = true;
            boundary_dofs[3 * node + 1] = true;
            boundary_dofs[3 * node + 2] = true;
        }
        let d_mat = full_3d_d(young, nu);
        Self {
            mesh,
            d_mat,
            boundary_dofs,
        }
    }
}

impl LinearOperator for ElasticityMF3D {
    fn size(&self) -> usize {
        self.mesh.num_nodes() * 3
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let n = self.size();
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);

        for yi in y.iter_mut() {
            *yi = 0.0;
        }

        for element in &self.mesh.elements {
            let coords = [
                self.mesh.nodes[element[0]],
                self.mesh.nodes[element[1]],
                self.mesh.nodes[element[2]],
                self.mesh.nodes[element[3]],
                self.mesh.nodes[element[4]],
                self.mesh.nodes[element[5]],
                self.mesh.nodes[element[6]],
                self.mesh.nodes[element[7]],
            ];

            let ke = element_stiffness_3d(coords, &self.d_mat);

            for a in 0..8 {
                for i in 0..3 {
                    let row = 3 * element[a] + i;
                    if self.boundary_dofs[row] {
                        continue;
                    }
                    for b in 0..8 {
                        for j in 0..3 {
                            let col = 3 * element[b] + j;
                            y[row] += ke[3 * a + i][3 * b + j] * x[col];
                        }
                    }
                }
            }
        }

        // Boundary DOFs: identity
        for (idx, &is_boundary) in self.boundary_dofs.iter().enumerate() {
            if is_boundary {
                y[idx] = x[idx];
            }
        }
    }

    fn diagonal(&self, diag: &mut [f64]) -> Result<(), SpinozaError> {
        let n = self.size();
        if diag.len() != n {
            return Err(SpinozaError::OperatorError(
                "diagonal slice length mismatch".to_string(),
            ));
        }

        for d in diag.iter_mut() {
            *d = 0.0;
        }

        for element in &self.mesh.elements {
            let coords = [
                self.mesh.nodes[element[0]],
                self.mesh.nodes[element[1]],
                self.mesh.nodes[element[2]],
                self.mesh.nodes[element[3]],
                self.mesh.nodes[element[4]],
                self.mesh.nodes[element[5]],
                self.mesh.nodes[element[6]],
                self.mesh.nodes[element[7]],
            ];

            let ke = element_stiffness_3d(coords, &self.d_mat);

            for a in 0..8 {
                for i in 0..3 {
                    let row = 3 * element[a] + i;
                    if !self.boundary_dofs[row] {
                        diag[row] += ke[3 * a + i][3 * a + i];
                    }
                }
            }
        }

        for (idx, &is_boundary) in self.boundary_dofs.iter().enumerate() {
            if is_boundary {
                diag[idx] = 1.0;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Operator factory
// ---------------------------------------------------------------------------

struct ElasticityOperatorFactory;

impl OperatorFactory for ElasticityOperatorFactory {
    fn name(&self) -> &str {
        "ElasticityOperator"
    }

    fn supported_keys(&self) -> Vec<OperatorKey> {
        vec![
            OperatorKey {
                equation_family: "elasticity".into(),
                backend: "assembled".into(),
                dimension: 2,
                space_signature: "H1Vector".into(),
                element_family: "quad".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            OperatorKey {
                equation_family: "elasticity".into(),
                backend: "assembled".into(),
                dimension: 3,
                space_signature: "H1Vector".into(),
                element_family: "hex".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            OperatorKey {
                equation_family: "elasticity".into(),
                backend: "matrixfree".into(),
                dimension: 2,
                space_signature: "H1Vector".into(),
                element_family: "quad".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            OperatorKey {
                equation_family: "elasticity".into(),
                backend: "matrixfree".into(),
                dimension: 3,
                space_signature: "H1Vector".into(),
                element_family: "hex".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
        ]
    }

    fn capabilities_provided(&self) -> &[Capability] {
        &[Capability::OperatorElasticity]
    }

    fn capabilities_required(&self) -> &[Capability] {
        &[Capability::SpaceH1Vector, Capability::BcDirichlet]
    }

    fn build(
        &self,
        key: &OperatorKey,
        spec: &CaseSpec,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<BuiltOperator, String> {
        let young = spec
            .equation
            .young_modulus
            .unwrap_or(1.0);
        let nu = spec
            .equation
            .poisson_ratio
            .unwrap_or(0.3);
        let boundary_value = spec.bcs.last().map(|bc| bc.value).unwrap_or(0.0);
        let dim = key.dimension;

        // Field names: displacement components
        let field_names: Vec<String> = if dim == 2 {
            vec!["ux".into(), "uy".into()]
        } else {
            vec!["ux".into(), "uy".into(), "uz".into()]
        };

        match (dim, key.backend.as_str()) {
            (2, "assembled") => {
                let mesh = StructuredQuadMesh2D::unit_square(nx, ny);
                let ndofs = mesh.num_nodes() * 2;
                let (mut matrix, mut rhs) = assemble_elasticity_2d(&mesh, young, nu);
                apply_dirichlet_elasticity_2d(&mesh, &mut matrix, &mut rhs, boundary_value);
                let csr = matrix.to_csr();
                Ok(BuiltOperator {
                    operator: Box::new(csr),
                    rhs,
                    field_names,
                    field_sizes: vec![ndofs],
                })
            }
            (2, "matrixfree") => {
                let mesh = StructuredQuadMesh2D::unit_square(nx, ny);
                let ndofs = mesh.num_nodes() * 2;
                let mut rhs = vec![0.0; ndofs];
                // For matrixfree with zero BCs, rhs stays zero
                if boundary_value != 0.0 {
                    // Assemble just to get BC corrections, then discard matrix
                    let (mut tmp_mat, _) = assemble_elasticity_2d(&mesh, young, nu);
                    apply_dirichlet_elasticity_2d(&mesh, &mut tmp_mat, &mut rhs, boundary_value);
                    // Only rhs corrections matter; the MF operator handles identity on boundary
                }
                let op = ElasticityMF2D::new(mesh, young, nu);
                Ok(BuiltOperator {
                    operator: Box::new(op),
                    rhs,
                    field_names,
                    field_sizes: vec![ndofs],
                })
            }
            (3, "assembled") => {
                let mesh = StructuredHexMesh3D::unit_cube(nx, ny, nz);
                let ndofs = mesh.num_nodes() * 3;
                let (mut matrix, mut rhs) = assemble_elasticity_3d(&mesh, young, nu);
                apply_dirichlet_elasticity_3d(&mesh, &mut matrix, &mut rhs, boundary_value);
                let csr = matrix.to_csr();
                Ok(BuiltOperator {
                    operator: Box::new(csr),
                    rhs,
                    field_names,
                    field_sizes: vec![ndofs],
                })
            }
            (3, "matrixfree") => {
                let mesh = StructuredHexMesh3D::unit_cube(nx, ny, nz);
                let ndofs = mesh.num_nodes() * 3;
                let mut rhs = vec![0.0; ndofs];
                if boundary_value != 0.0 {
                    let (mut tmp_mat, _) = assemble_elasticity_3d(&mesh, young, nu);
                    apply_dirichlet_elasticity_3d(&mesh, &mut tmp_mat, &mut rhs, boundary_value);
                }
                let op = ElasticityMF3D::new(mesh, young, nu);
                Ok(BuiltOperator {
                    operator: Box::new(op),
                    rhs,
                    field_names,
                    field_sizes: vec![ndofs],
                })
            }
            _ => Err(format!(
                "unsupported dimension/backend: dim={}, backend={}",
                dim, key.backend
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Preconditioner factory
// ---------------------------------------------------------------------------

struct ElasticityJacobiPrecondFactory;

impl PreconditionerFactory for ElasticityJacobiPrecondFactory {
    fn name(&self) -> &str {
        "ElasticityJacobiPreconditioner"
    }

    fn supported_keys(&self) -> Vec<PreconditionerKey> {
        vec![
            PreconditionerKey {
                precond_type: "jacobi".into(),
                backend: "assembled".into(),
                dimension: 2,
                operator_family: "elasticity".into(),
                space_signature: "H1Vector".into(),
                element_family: "quad".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            PreconditionerKey {
                precond_type: "jacobi".into(),
                backend: "assembled".into(),
                dimension: 3,
                operator_family: "elasticity".into(),
                space_signature: "H1Vector".into(),
                element_family: "hex".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            PreconditionerKey {
                precond_type: "jacobi".into(),
                backend: "matrixfree".into(),
                dimension: 2,
                operator_family: "elasticity".into(),
                space_signature: "H1Vector".into(),
                element_family: "quad".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            PreconditionerKey {
                precond_type: "jacobi".into(),
                backend: "matrixfree".into(),
                dimension: 3,
                operator_family: "elasticity".into(),
                space_signature: "H1Vector".into(),
                element_family: "hex".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
        ]
    }

    fn capabilities_provided(&self) -> &[Capability] {
        &[Capability::PrecondJacobi]
    }

    fn capabilities_required(&self) -> &[Capability] {
        &[]
    }

    fn build(
        &self,
        _key: &PreconditionerKey,
        operator: &dyn LinearOperator,
        _spec: &CaseSpec,
        _nx: usize,
        _ny: usize,
        _nz: usize,
    ) -> Result<Box<dyn Preconditioner>, String> {
        Ok(Box::new(JacobiPreconditioner::from_operator(operator)?))
    }
}

// ---------------------------------------------------------------------------
// Problem builder
// ---------------------------------------------------------------------------

struct ElasticityBuilder;

impl ProblemBuilder for ElasticityBuilder {
    fn name(&self) -> &str {
        "ElasticityBuilder"
    }

    fn equation_type(&self) -> &str {
        "elasticity"
    }

    fn supported_dimensions(&self) -> &[u8] {
        &[2, 3]
    }

    fn supported_backends(&self) -> &[&str] {
        &["assembled", "matrixfree"]
    }

    fn build(
        &self,
        spec: &CaseSpec,
        backend: &str,
        precond_hint: &str,
        nx: usize,
        ny: usize,
        nz: usize,
        registry: &MethodRegistry,
    ) -> Result<BuiltProblem, String> {
        let dim = spec.mesh.dimension;
        let elem = if dim == 2 { "quad" } else { "hex" };

        // 1. Request operator from registry.
        let op_key = OperatorKey {
            equation_family: "elasticity".into(),
            backend: backend.into(),
            dimension: dim,
            space_signature: "H1Vector".into(),
            element_family: elem.into(),
            order: 1,
            block_structure: "single_field".into(),
        };
        let built_op = match registry.find_operator(&op_key) {
            ComponentLookup::Found { component, .. } => {
                component.build(&op_key, spec, nx, ny, nz)?
            }
            ComponentLookup::NotFound => {
                return Err(format!(
                    "no operator factory for equation='elasticity', backend='{backend}', dim={dim}"
                ));
            }
            ComponentLookup::Ambiguous(err) => return Err(err.to_string()),
        };

        // 2. Request preconditioner from registry.
        let pc_key = PreconditionerKey {
            precond_type: precond_hint.into(),
            backend: backend.into(),
            dimension: dim,
            operator_family: "elasticity".into(),
            space_signature: "H1Vector".into(),
            element_family: elem.into(),
            order: 1,
            block_structure: "single_field".into(),
        };
        let precond = match registry.find_preconditioner(&pc_key) {
            ComponentLookup::Found { component, .. } => {
                Some(component.build(&pc_key, built_op.operator.as_ref(), spec, nx, ny, nz)?)
            }
            ComponentLookup::NotFound => None,
            ComponentLookup::Ambiguous(err) => return Err(err.to_string()),
        };

        Ok(BuiltProblem {
            operator: built_op.operator,
            rhs: built_op.rhs,
            preconditioner: precond,
            field_names: built_op.field_names,
            field_sizes: built_op.field_sizes,
        })
    }
}
