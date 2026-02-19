#[derive(Debug, Clone)]
pub struct StructuredQuadMesh2D {
    pub nx: usize,
    pub ny: usize,
    pub nodes: Vec<[f64; 2]>,
    pub elements: Vec<[usize; 4]>,
}

#[derive(Debug, Clone)]
pub struct StructuredHexMesh3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nodes: Vec<[f64; 3]>,
    pub elements: Vec<[usize; 8]>,
}

impl StructuredQuadMesh2D {
    pub fn unit_square(nx: usize, ny: usize) -> Self {
        assert!(nx >= 1, "nx must be >= 1");
        assert!(ny >= 1, "ny must be >= 1");

        let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1));
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push([i as f64 / nx as f64, j as f64 / ny as f64]);
            }
        }

        let mut elements = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                let n0 = node_index(nx, i, j);
                let n1 = node_index(nx, i + 1, j);
                let n2 = node_index(nx, i + 1, j + 1);
                let n3 = node_index(nx, i, j + 1);
                elements.push([n0, n1, n2, n3]);
            }
        }

        Self {
            nx,
            ny,
            nodes,
            elements,
        }
    }

    pub fn boundary_nodes(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for j in 0..=self.ny {
            for i in 0..=self.nx {
                if i == 0 || i == self.nx || j == 0 || j == self.ny {
                    out.push(node_index(self.nx, i, j));
                }
            }
        }
        out
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }
}

impl StructuredHexMesh3D {
    pub fn unit_cube(nx: usize, ny: usize, nz: usize) -> Self {
        assert!(nx >= 1, "nx must be >= 1");
        assert!(ny >= 1, "ny must be >= 1");
        assert!(nz >= 1, "nz must be >= 1");

        let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    nodes.push([
                        i as f64 / nx as f64,
                        j as f64 / ny as f64,
                        k as f64 / nz as f64,
                    ]);
                }
            }
        }

        let mut elements = Vec::with_capacity(nx * ny * nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let n000 = node_index_3d(nx, ny, i, j, k);
                    let n100 = node_index_3d(nx, ny, i + 1, j, k);
                    let n110 = node_index_3d(nx, ny, i + 1, j + 1, k);
                    let n010 = node_index_3d(nx, ny, i, j + 1, k);
                    let n001 = node_index_3d(nx, ny, i, j, k + 1);
                    let n101 = node_index_3d(nx, ny, i + 1, j, k + 1);
                    let n111 = node_index_3d(nx, ny, i + 1, j + 1, k + 1);
                    let n011 = node_index_3d(nx, ny, i, j + 1, k + 1);
                    elements.push([n000, n100, n110, n010, n001, n101, n111, n011]);
                }
            }
        }

        Self {
            nx,
            ny,
            nz,
            nodes,
            elements,
        }
    }

    pub fn boundary_nodes(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for k in 0..=self.nz {
            for j in 0..=self.ny {
                for i in 0..=self.nx {
                    if i == 0 || i == self.nx || j == 0 || j == self.ny || k == 0 || k == self.nz {
                        out.push(node_index_3d(self.nx, self.ny, i, j, k));
                    }
                }
            }
        }
        out
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }
}

fn node_index(nx: usize, i: usize, j: usize) -> usize {
    j * (nx + 1) + i
}

fn node_index_3d(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> usize {
    k * (ny + 1) * (nx + 1) + j * (nx + 1) + i
}
