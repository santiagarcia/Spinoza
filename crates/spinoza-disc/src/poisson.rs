use crate::matrix::SparseMatrixBuilder;
use crate::mesh::{StructuredHexMesh3D, StructuredQuadMesh2D};

pub fn assemble_poisson_system(mesh: &StructuredQuadMesh2D) -> (SparseMatrixBuilder, Vec<f64>) {
    assemble_poisson_system_with_source(mesh, |_, _| 0.0)
}

pub fn assemble_poisson_system_3d(mesh: &StructuredHexMesh3D) -> (SparseMatrixBuilder, Vec<f64>) {
    assemble_poisson_system_3d_with_source(mesh, |_, _, _| 0.0)
}

pub fn assemble_poisson_system_with_source<F>(
    mesh: &StructuredQuadMesh2D,
    source: F,
) -> (SparseMatrixBuilder, Vec<f64>)
where
    F: Fn(f64, f64) -> f64,
{
    let ndofs = mesh.num_nodes();
    let mut matrix = SparseMatrixBuilder::new(ndofs, ndofs);
    let mut rhs = vec![0.0; ndofs];

    for element in &mesh.elements {
        let coords = [
            mesh.nodes[element[0]],
            mesh.nodes[element[1]],
            mesh.nodes[element[2]],
            mesh.nodes[element[3]],
        ];

        let (ke, fe) = element_matrices_2d(coords, &source);

        for a in 0..4 {
            rhs[element[a]] += fe[a];
            for b in 0..4 {
                matrix.add_entry(element[a], element[b], ke[a][b]);
            }
        }
    }

    (matrix, rhs)
}

pub fn assemble_poisson_system_3d_with_source<F>(
    mesh: &StructuredHexMesh3D,
    source: F,
) -> (SparseMatrixBuilder, Vec<f64>)
where
    F: Fn(f64, f64, f64) -> f64,
{
    let ndofs = mesh.num_nodes();
    let mut matrix = SparseMatrixBuilder::new(ndofs, ndofs);
    let mut rhs = vec![0.0; ndofs];

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

        let (ke, fe) = element_matrices_3d(coords, &source);

        for a in 0..8 {
            rhs[element[a]] += fe[a];
            for b in 0..8 {
                matrix.add_entry(element[a], element[b], ke[a][b]);
            }
        }
    }

    (matrix, rhs)
}

pub fn apply_constant_dirichlet_on_boundary(
    mesh: &StructuredQuadMesh2D,
    matrix: &mut SparseMatrixBuilder,
    rhs: &mut [f64],
    value: f64,
) {
    let boundary_nodes = mesh.boundary_nodes();
    apply_constant_dirichlet(matrix, rhs, &boundary_nodes, value);
}

pub fn apply_constant_dirichlet_on_boundary_3d(
    mesh: &StructuredHexMesh3D,
    matrix: &mut SparseMatrixBuilder,
    rhs: &mut [f64],
    value: f64,
) {
    let boundary_nodes = mesh.boundary_nodes();
    apply_constant_dirichlet(matrix, rhs, &boundary_nodes, value);
}

pub fn apply_constant_dirichlet(
    matrix: &mut SparseMatrixBuilder,
    rhs: &mut [f64],
    nodes: &[usize],
    value: f64,
) {
    let values: Vec<(usize, f64)> = nodes.iter().map(|idx| (*idx, value)).collect();
    apply_dirichlet_values(matrix, rhs, &values);
}

pub fn apply_dirichlet_values(
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

pub(crate) fn element_matrices_2d<F>(coords: [[f64; 2]; 4], source: &F) -> ([[f64; 4]; 4], [f64; 4])
where
    F: Fn(f64, f64) -> f64,
{
    let gauss = [(-1.0 / 3.0_f64.sqrt(), 1.0), (1.0 / 3.0_f64.sqrt(), 1.0)];

    let mut ke = [[0.0; 4]; 4];
    let mut fe = [0.0; 4];

    for (xi, wx) in gauss {
        for (eta, wy) in gauss {
            let n = shape_q1(xi, eta);
            let dndxi = dshape_q1_dxi(xi, eta);
            let dndeta = dshape_q1_deta(xi, eta);

            let mut j11 = 0.0;
            let mut j12 = 0.0;
            let mut j21 = 0.0;
            let mut j22 = 0.0;
            let mut x = 0.0;
            let mut y = 0.0;

            for a in 0..4 {
                x += n[a] * coords[a][0];
                y += n[a] * coords[a][1];
                j11 += dndxi[a] * coords[a][0];
                j12 += dndeta[a] * coords[a][0];
                j21 += dndxi[a] * coords[a][1];
                j22 += dndeta[a] * coords[a][1];
            }

            let detj = j11 * j22 - j12 * j21;
            let invj = [[j22 / detj, -j12 / detj], [-j21 / detj, j11 / detj]];

            let mut grad = [[0.0; 2]; 4];
            for a in 0..4 {
                grad[a][0] = invj[0][0] * dndxi[a] + invj[0][1] * dndeta[a];
                grad[a][1] = invj[1][0] * dndxi[a] + invj[1][1] * dndeta[a];
            }

            let weight = wx * wy * detj;
            let fval = source(x, y);

            for a in 0..4 {
                fe[a] += n[a] * fval * weight;
                for b in 0..4 {
                    let dot = grad[a][0] * grad[b][0] + grad[a][1] * grad[b][1];
                    ke[a][b] += dot * weight;
                }
            }
        }
    }

    (ke, fe)
}

pub(crate) fn element_matrices_3d<F>(coords: [[f64; 3]; 8], source: &F) -> ([[f64; 8]; 8], [f64; 8])
where
    F: Fn(f64, f64, f64) -> f64,
{
    let g = 1.0 / 3.0_f64.sqrt();
    let gauss = [(-g, 1.0), (g, 1.0)];

    let mut ke = [[0.0; 8]; 8];
    let mut fe = [0.0; 8];

    for (xi, wx) in gauss {
        for (eta, wy) in gauss {
            for (zeta, wz) in gauss {
                let n = shape_q1_hex(xi, eta, zeta);
                let dndxi = dshape_q1_hex_dxi(xi, eta, zeta);
                let dndeta = dshape_q1_hex_deta(xi, eta, zeta);
                let dndzeta = dshape_q1_hex_dzeta(xi, eta, zeta);

                let mut j = [[0.0; 3]; 3];
                let mut x = 0.0;
                let mut y = 0.0;
                let mut z = 0.0;

                for a in 0..8 {
                    x += n[a] * coords[a][0];
                    y += n[a] * coords[a][1];
                    z += n[a] * coords[a][2];

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

                let mut grad = [[0.0; 3]; 8];
                for a in 0..8 {
                    grad[a][0] =
                        invj[0][0] * dndxi[a] + invj[0][1] * dndeta[a] + invj[0][2] * dndzeta[a];
                    grad[a][1] =
                        invj[1][0] * dndxi[a] + invj[1][1] * dndeta[a] + invj[1][2] * dndzeta[a];
                    grad[a][2] =
                        invj[2][0] * dndxi[a] + invj[2][1] * dndeta[a] + invj[2][2] * dndzeta[a];
                }

                let weight = wx * wy * wz * detj;
                let fval = source(x, y, z);

                for a in 0..8 {
                    fe[a] += n[a] * fval * weight;
                    for b in 0..8 {
                        let dot = grad[a][0] * grad[b][0]
                            + grad[a][1] * grad[b][1]
                            + grad[a][2] * grad[b][2];
                        ke[a][b] += dot * weight;
                    }
                }
            }
        }
    }

    (ke, fe)
}

fn shape_q1(xi: f64, eta: f64) -> [f64; 4] {
    [
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta),
    ]
}

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

fn shape_q1_hex(xi: f64, eta: f64, zeta: f64) -> [f64; 8] {
    let s = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ];
    let mut out = [0.0; 8];
    for a in 0..8 {
        out[a] = 0.125 * (1.0 + s[a].0 * xi) * (1.0 + s[a].1 * eta) * (1.0 + s[a].2 * zeta);
    }
    out
}

fn dshape_q1_hex_dxi(_xi: f64, eta: f64, zeta: f64) -> [f64; 8] {
    let s = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ];
    let mut out = [0.0; 8];
    for a in 0..8 {
        out[a] = 0.125 * s[a].0 * (1.0 + s[a].1 * eta) * (1.0 + s[a].2 * zeta);
    }
    out
}

fn dshape_q1_hex_deta(xi: f64, _eta: f64, zeta: f64) -> [f64; 8] {
    let s = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ];
    let mut out = [0.0; 8];
    for a in 0..8 {
        out[a] = 0.125 * (1.0 + s[a].0 * xi) * s[a].1 * (1.0 + s[a].2 * zeta);
    }
    out
}

fn dshape_q1_hex_dzeta(xi: f64, eta: f64, _zeta: f64) -> [f64; 8] {
    let s = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ];
    let mut out = [0.0; 8];
    for a in 0..8 {
        out[a] = 0.125 * (1.0 + s[a].0 * xi) * (1.0 + s[a].1 * eta) * s[a].2;
    }
    out
}

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
