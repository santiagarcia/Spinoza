use std::cell::RefCell;

use spinoza_core::{LinearOperator, Preconditioner};
use spinoza_disc::{
    DirichletConstraints, PoissonLaplacianMF2D, PoissonLaplacianMF3D, StructuredHexMesh3D,
    StructuredQuadMesh2D,
};

#[derive(Debug, Clone, Copy)]
pub struct MultigridConfig {
    pub min_cells_per_axis: usize,
    pub pre_smooth_iters: usize,
    pub post_smooth_iters: usize,
    pub coarse_iters: usize,
    pub jacobi_weight: f64,
}

impl Default for MultigridConfig {
    fn default() -> Self {
        Self {
            min_cells_per_axis: 2,
            pre_smooth_iters: 2,
            post_smooth_iters: 2,
            coarse_iters: 24,
            jacobi_weight: 2.0 / 3.0,
        }
    }
}

#[derive(Debug, Clone)]
struct Level2D {
    nx: usize,
    ny: usize,
    op: PoissonLaplacianMF2D,
    inv_diag: Vec<f64>,
}

#[derive(Debug)]
struct Workspace2D {
    x: Vec<Vec<f64>>,
    rhs: Vec<Vec<f64>>,
    res: Vec<Vec<f64>>,
    tmp: Vec<Vec<f64>>,
}

impl Workspace2D {
    fn new(sizes: &[usize]) -> Self {
        let mut x = Vec::with_capacity(sizes.len());
        let mut rhs = Vec::with_capacity(sizes.len());
        let mut res = Vec::with_capacity(sizes.len());
        let mut tmp = Vec::with_capacity(sizes.len());
        for &n in sizes {
            x.push(vec![0.0; n]);
            rhs.push(vec![0.0; n]);
            res.push(vec![0.0; n]);
            tmp.push(vec![0.0; n]);
        }
        Self { x, rhs, res, tmp }
    }
}

pub struct MultigridPreconditioner2D {
    levels: Vec<Level2D>,
    config: MultigridConfig,
    workspace: RefCell<Workspace2D>,
}

impl MultigridPreconditioner2D {
    pub fn new(
        nx: usize,
        ny: usize,
        bc_value: f64,
        config: MultigridConfig,
    ) -> Result<Self, String> {
        if nx < 1 || ny < 1 {
            return Err("multigrid requires nx and ny >= 1".to_string());
        }

        let mut levels = Vec::new();
        let mut cx = nx;
        let mut cy = ny;

        loop {
            let mesh = StructuredQuadMesh2D::unit_square(cx, cy);
            let op = PoissonLaplacianMF2D::with_constant_dirichlet_boundary(mesh, bc_value);
            let mut diag = vec![0.0; op.size()];
            op.diagonal(&mut diag)
                .map_err(|e| format!("mg diagonal extraction failed: {e}"))?;
            if diag.iter().any(|d| d.abs() < 1e-14) {
                return Err("mg encountered zero diagonal at a level".to_string());
            }
            let inv_diag = diag.into_iter().map(|d| 1.0 / d).collect();
            levels.push(Level2D {
                nx: cx,
                ny: cy,
                op,
                inv_diag,
            });

            if cx <= config.min_cells_per_axis || cy <= config.min_cells_per_axis {
                break;
            }
            if cx % 2 != 0 || cy % 2 != 0 {
                break;
            }
            cx /= 2;
            cy /= 2;
        }

        let sizes: Vec<usize> = levels.iter().map(|lvl| lvl.op.size()).collect();
        let workspace = RefCell::new(Workspace2D::new(&sizes));
        Ok(Self {
            levels,
            config,
            workspace,
        })
    }

    fn smooth_level(
        level: &Level2D,
        rhs: &[f64],
        x: &mut [f64],
        tmp: &mut [f64],
        omega: f64,
        iters: usize,
    ) {
        for _ in 0..iters {
            level.op.apply(x, tmp);
            for i in 0..x.len() {
                x[i] += omega * level.inv_diag[i] * (rhs[i] - tmp[i]);
            }
        }
    }

    fn v_cycle_level(&self, l: usize, ws: &mut Workspace2D) {
        let last = self.levels.len() - 1;
        let level = &self.levels[l];

        if l == last {
            Self::smooth_level(
                level,
                &ws.rhs[l],
                &mut ws.x[l],
                &mut ws.tmp[l],
                self.config.jacobi_weight,
                self.config.coarse_iters,
            );
            return;
        }

        Self::smooth_level(
            level,
            &ws.rhs[l],
            &mut ws.x[l],
            &mut ws.tmp[l],
            self.config.jacobi_weight,
            self.config.pre_smooth_iters,
        );

        level.op.apply(&ws.x[l], &mut ws.tmp[l]);
        for i in 0..ws.res[l].len() {
            ws.res[l][i] = ws.rhs[l][i] - ws.tmp[l][i];
        }

        let coarse = &self.levels[l + 1];
        restrict_full_weighting_2d(
            &ws.res[l],
            &mut ws.rhs[l + 1],
            level.nx,
            level.ny,
            coarse.nx,
            coarse.ny,
        );
        ws.x[l + 1].fill(0.0);

        self.v_cycle_level(l + 1, ws);

        {
            let (fine_levels, coarse_levels) = ws.x.split_at_mut(l + 1);
            let fine = &mut fine_levels[l];
            let coarse_corr = &coarse_levels[0];
            prolongate_add_2d(coarse_corr, fine, level.nx, level.ny, coarse.nx, coarse.ny);
        }

        Self::smooth_level(
            level,
            &ws.rhs[l],
            &mut ws.x[l],
            &mut ws.tmp[l],
            self.config.jacobi_weight,
            self.config.post_smooth_iters,
        );
    }
}

impl Preconditioner for MultigridPreconditioner2D {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        assert_eq!(r.len(), self.levels[0].op.size());
        assert_eq!(z.len(), self.levels[0].op.size());

        let mut ws = self.workspace.borrow_mut();
        ws.rhs[0].copy_from_slice(r);
        ws.x[0].fill(0.0);

        self.v_cycle_level(0, &mut ws);
        z.copy_from_slice(&ws.x[0]);
    }
}

#[derive(Debug, Clone)]
struct Level3D {
    nx: usize,
    ny: usize,
    nz: usize,
    op: PoissonLaplacianMF3D,
    inv_diag: Vec<f64>,
}

#[derive(Debug)]
struct Workspace3D {
    x: Vec<Vec<f64>>,
    rhs: Vec<Vec<f64>>,
    res: Vec<Vec<f64>>,
    tmp: Vec<Vec<f64>>,
}

impl Workspace3D {
    fn new(sizes: &[usize]) -> Self {
        let mut x = Vec::with_capacity(sizes.len());
        let mut rhs = Vec::with_capacity(sizes.len());
        let mut res = Vec::with_capacity(sizes.len());
        let mut tmp = Vec::with_capacity(sizes.len());
        for &n in sizes {
            x.push(vec![0.0; n]);
            rhs.push(vec![0.0; n]);
            res.push(vec![0.0; n]);
            tmp.push(vec![0.0; n]);
        }
        Self { x, rhs, res, tmp }
    }
}

pub struct MultigridPreconditioner3D {
    levels: Vec<Level3D>,
    config: MultigridConfig,
    workspace: RefCell<Workspace3D>,
}

impl MultigridPreconditioner3D {
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        bc_value: f64,
        config: MultigridConfig,
    ) -> Result<Self, String> {
        if nx < 1 || ny < 1 || nz < 1 {
            return Err("multigrid requires nx, ny, nz >= 1".to_string());
        }

        let mut levels = Vec::new();
        let mut cx = nx;
        let mut cy = ny;
        let mut cz = nz;

        loop {
            let mesh = StructuredHexMesh3D::unit_cube(cx, cy, cz);
            let op = PoissonLaplacianMF3D::with_constant_dirichlet_boundary(mesh, bc_value);
            let mut diag = vec![0.0; op.size()];
            op.diagonal(&mut diag)
                .map_err(|e| format!("mg diagonal extraction failed: {e}"))?;
            if diag.iter().any(|d| d.abs() < 1e-14) {
                return Err("mg encountered zero diagonal at a level".to_string());
            }
            let inv_diag = diag.into_iter().map(|d| 1.0 / d).collect();
            levels.push(Level3D {
                nx: cx,
                ny: cy,
                nz: cz,
                op,
                inv_diag,
            });

            if cx <= config.min_cells_per_axis
                || cy <= config.min_cells_per_axis
                || cz <= config.min_cells_per_axis
            {
                break;
            }
            if cx % 2 != 0 || cy % 2 != 0 || cz % 2 != 0 {
                break;
            }
            cx /= 2;
            cy /= 2;
            cz /= 2;
        }

        let sizes: Vec<usize> = levels.iter().map(|lvl| lvl.op.size()).collect();
        let workspace = RefCell::new(Workspace3D::new(&sizes));
        Ok(Self {
            levels,
            config,
            workspace,
        })
    }

    fn smooth_level(
        level: &Level3D,
        rhs: &[f64],
        x: &mut [f64],
        tmp: &mut [f64],
        omega: f64,
        iters: usize,
    ) {
        for _ in 0..iters {
            level.op.apply(x, tmp);
            for i in 0..x.len() {
                x[i] += omega * level.inv_diag[i] * (rhs[i] - tmp[i]);
            }
        }
    }

    fn v_cycle_level(&self, l: usize, ws: &mut Workspace3D) {
        let last = self.levels.len() - 1;
        let level = &self.levels[l];

        if l == last {
            Self::smooth_level(
                level,
                &ws.rhs[l],
                &mut ws.x[l],
                &mut ws.tmp[l],
                self.config.jacobi_weight,
                self.config.coarse_iters,
            );
            return;
        }

        Self::smooth_level(
            level,
            &ws.rhs[l],
            &mut ws.x[l],
            &mut ws.tmp[l],
            self.config.jacobi_weight,
            self.config.pre_smooth_iters,
        );

        level.op.apply(&ws.x[l], &mut ws.tmp[l]);
        for i in 0..ws.res[l].len() {
            ws.res[l][i] = ws.rhs[l][i] - ws.tmp[l][i];
        }

        let coarse = &self.levels[l + 1];
        restrict_full_weighting_3d(
            &ws.res[l],
            &mut ws.rhs[l + 1],
            Grid3 {
                nx: level.nx,
                ny: level.ny,
                nz: level.nz,
            },
            Grid3 {
                nx: coarse.nx,
                ny: coarse.ny,
                nz: coarse.nz,
            },
        );
        ws.x[l + 1].fill(0.0);

        self.v_cycle_level(l + 1, ws);

        {
            let (fine_levels, coarse_levels) = ws.x.split_at_mut(l + 1);
            let fine = &mut fine_levels[l];
            let coarse_corr = &coarse_levels[0];
            prolongate_add_3d(
                coarse_corr,
                fine,
                Grid3 {
                    nx: level.nx,
                    ny: level.ny,
                    nz: level.nz,
                },
                Grid3 {
                    nx: coarse.nx,
                    ny: coarse.ny,
                    nz: coarse.nz,
                },
            );
        }

        Self::smooth_level(
            level,
            &ws.rhs[l],
            &mut ws.x[l],
            &mut ws.tmp[l],
            self.config.jacobi_weight,
            self.config.post_smooth_iters,
        );
    }
}

impl Preconditioner for MultigridPreconditioner3D {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        assert_eq!(r.len(), self.levels[0].op.size());
        assert_eq!(z.len(), self.levels[0].op.size());

        let mut ws = self.workspace.borrow_mut();
        ws.rhs[0].copy_from_slice(r);
        ws.x[0].fill(0.0);

        self.v_cycle_level(0, &mut ws);
        z.copy_from_slice(&ws.x[0]);
    }
}

fn idx2(nx: usize, i: usize, j: usize) -> usize {
    j * (nx + 1) + i
}

fn restrict_full_weighting_2d(
    fine: &[f64],
    coarse: &mut [f64],
    nxf: usize,
    nyf: usize,
    nxc: usize,
    nyc: usize,
) {
    debug_assert_eq!(nxf, 2 * nxc);
    debug_assert_eq!(nyf, 2 * nyc);

    for j in 0..=nyc {
        for i in 0..=nxc {
            let ci = idx2(nxc, i, j);
            let fi = 2 * i;
            let fj = 2 * j;

            if i == 0 || j == 0 || i == nxc || j == nyc {
                coarse[ci] = fine[idx2(nxf, fi, fj)];
                continue;
            }

            let mut acc = 0.0;
            for dj in 0..3 {
                for di in 0..3 {
                    let w_i = if di == 1 { 2.0 } else { 1.0 };
                    let w_j = if dj == 1 { 2.0 } else { 1.0 };
                    let ff = idx2(nxf, fi + di - 1, fj + dj - 1);
                    acc += w_i * w_j * fine[ff];
                }
            }
            coarse[ci] = acc / 16.0;
        }
    }
}

fn prolongate_add_2d(
    coarse: &[f64],
    fine: &mut [f64],
    nxf: usize,
    nyf: usize,
    nxc: usize,
    nyc: usize,
) {
    debug_assert_eq!(nxf, 2 * nxc);
    debug_assert_eq!(nyf, 2 * nyc);

    for j in 0..=nyf {
        for i in 0..=nxf {
            let fi = idx2(nxf, i, j);
            let ic = i / 2;
            let jc = j / 2;

            let val = match (i % 2, j % 2) {
                (0, 0) => coarse[idx2(nxc, ic, jc)],
                (1, 0) => 0.5 * (coarse[idx2(nxc, ic, jc)] + coarse[idx2(nxc, ic + 1, jc)]),
                (0, 1) => 0.5 * (coarse[idx2(nxc, ic, jc)] + coarse[idx2(nxc, ic, jc + 1)]),
                (1, 1) => {
                    0.25 * (coarse[idx2(nxc, ic, jc)]
                        + coarse[idx2(nxc, ic + 1, jc)]
                        + coarse[idx2(nxc, ic, jc + 1)]
                        + coarse[idx2(nxc, ic + 1, jc + 1)])
                }
                _ => 0.0,
            };

            fine[fi] += val;
        }
    }
}

fn idx3(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> usize {
    k * (ny + 1) * (nx + 1) + j * (nx + 1) + i
}

#[derive(Clone, Copy)]
struct Grid3 {
    nx: usize,
    ny: usize,
    nz: usize,
}

fn restrict_full_weighting_3d(
    fine: &[f64],
    coarse: &mut [f64],
    fine_grid: Grid3,
    coarse_grid: Grid3,
) {
    debug_assert_eq!(fine_grid.nx, 2 * coarse_grid.nx);
    debug_assert_eq!(fine_grid.ny, 2 * coarse_grid.ny);
    debug_assert_eq!(fine_grid.nz, 2 * coarse_grid.nz);

    for k in 0..=coarse_grid.nz {
        for j in 0..=coarse_grid.ny {
            for i in 0..=coarse_grid.nx {
                let ci = idx3(coarse_grid.nx, coarse_grid.ny, i, j, k);
                let fi = 2 * i;
                let fj = 2 * j;
                let fk = 2 * k;

                if i == 0
                    || j == 0
                    || k == 0
                    || i == coarse_grid.nx
                    || j == coarse_grid.ny
                    || k == coarse_grid.nz
                {
                    coarse[ci] = fine[idx3(fine_grid.nx, fine_grid.ny, fi, fj, fk)];
                    continue;
                }

                let mut acc = 0.0;
                for dk in 0..3 {
                    for dj in 0..3 {
                        for di in 0..3 {
                            let w_i = if di == 1 { 2.0 } else { 1.0 };
                            let w_j = if dj == 1 { 2.0 } else { 1.0 };
                            let w_k = if dk == 1 { 2.0 } else { 1.0 };
                            let ff = idx3(
                                fine_grid.nx,
                                fine_grid.ny,
                                fi + di - 1,
                                fj + dj - 1,
                                fk + dk - 1,
                            );
                            acc += w_i * w_j * w_k * fine[ff];
                        }
                    }
                }
                coarse[ci] = acc / 64.0;
            }
        }
    }
}

fn prolongate_add_3d(coarse: &[f64], fine: &mut [f64], fine_grid: Grid3, coarse_grid: Grid3) {
    debug_assert_eq!(fine_grid.nx, 2 * coarse_grid.nx);
    debug_assert_eq!(fine_grid.ny, 2 * coarse_grid.ny);
    debug_assert_eq!(fine_grid.nz, 2 * coarse_grid.nz);

    for k in 0..=fine_grid.nz {
        for j in 0..=fine_grid.ny {
            for i in 0..=fine_grid.nx {
                let fi = idx3(fine_grid.nx, fine_grid.ny, i, j, k);
                let ic = i / 2;
                let jc = j / 2;
                let kc = k / 2;

                let wx = if i % 2 == 0 {
                    [(ic, 1.0), (ic, 0.0)]
                } else {
                    [(ic, 0.5), (ic + 1, 0.5)]
                };
                let wy = if j % 2 == 0 {
                    [(jc, 1.0), (jc, 0.0)]
                } else {
                    [(jc, 0.5), (jc + 1, 0.5)]
                };
                let wz = if k % 2 == 0 {
                    [(kc, 1.0), (kc, 0.0)]
                } else {
                    [(kc, 0.5), (kc + 1, 0.5)]
                };

                let mut val = 0.0;
                for &(ix, wxv) in &wx {
                    if wxv == 0.0 {
                        continue;
                    }
                    for &(iy, wyv) in &wy {
                        if wyv == 0.0 {
                            continue;
                        }
                        for &(iz, wzv) in &wz {
                            if wzv == 0.0 {
                                continue;
                            }
                            val += wxv
                                * wyv
                                * wzv
                                * coarse[idx3(coarse_grid.nx, coarse_grid.ny, ix, iy, iz)];
                        }
                    }
                }

                fine[fi] += val;
            }
        }
    }
}

pub fn make_constant_boundary_constraints_2d(
    mesh: &StructuredQuadMesh2D,
    value: f64,
) -> DirichletConstraints {
    let node_values: Vec<(usize, f64)> = mesh
        .boundary_nodes()
        .into_iter()
        .map(|idx| (idx, value))
        .collect();
    DirichletConstraints::new(mesh.num_nodes(), &node_values)
}

pub fn make_constant_boundary_constraints_3d(
    mesh: &StructuredHexMesh3D,
    value: f64,
) -> DirichletConstraints {
    let node_values: Vec<(usize, f64)> = mesh
        .boundary_nodes()
        .into_iter()
        .map(|idx| (idx, value))
        .collect();
    DirichletConstraints::new(mesh.num_nodes(), &node_values)
}
