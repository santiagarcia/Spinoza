use spinoza_core::{LinearOperator, SpinozaError};

use crate::mesh::{StructuredHexMesh3D, StructuredQuadMesh2D};
use crate::poisson::{element_matrices_2d, element_matrices_3d};

#[derive(Debug, Clone)]
pub struct DirichletConstraints {
    values: Vec<Option<f64>>,
    constrained: Vec<usize>,
}

impl DirichletConstraints {
    pub fn new(size: usize, node_values: &[(usize, f64)]) -> Self {
        let mut values = vec![None; size];
        let mut constrained = Vec::with_capacity(node_values.len());
        for &(idx, val) in node_values {
            if values[idx].is_none() {
                constrained.push(idx);
            }
            values[idx] = Some(val);
        }
        constrained.sort_unstable();
        constrained.dedup();
        Self {
            values,
            constrained,
        }
    }

    pub fn unconstrained(&self, idx: usize) -> bool {
        self.values[idx].is_none()
    }

    pub fn constrained_indices(&self) -> &[usize] {
        &self.constrained
    }

    pub fn value(&self, idx: usize) -> Option<f64> {
        self.values[idx]
    }
}

#[derive(Debug, Clone)]
pub struct PoissonLaplacianMF2D {
    mesh: StructuredQuadMesh2D,
    constraints: DirichletConstraints,
}

impl PoissonLaplacianMF2D {
    pub fn new(mesh: StructuredQuadMesh2D, constraints: DirichletConstraints) -> Self {
        Self { mesh, constraints }
    }

    pub fn with_constant_dirichlet_boundary(mesh: StructuredQuadMesh2D, value: f64) -> Self {
        let node_values: Vec<(usize, f64)> = mesh
            .boundary_nodes()
            .into_iter()
            .map(|idx| (idx, value))
            .collect();
        let constraints = DirichletConstraints::new(mesh.num_nodes(), &node_values);
        Self::new(mesh, constraints)
    }

    pub fn constraints(&self) -> &DirichletConstraints {
        &self.constraints
    }
}

impl LinearOperator for PoissonLaplacianMF2D {
    fn size(&self) -> usize {
        self.mesh.num_nodes()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.size());
        assert_eq!(y.len(), self.size());

        y.fill(0.0);

        for element in &self.mesh.elements {
            let coords = [
                self.mesh.nodes[element[0]],
                self.mesh.nodes[element[1]],
                self.mesh.nodes[element[2]],
                self.mesh.nodes[element[3]],
            ];
            let (ke, _) = element_matrices_2d(coords, &|_, _| 0.0);

            for a in 0..4 {
                let i = element[a];
                if !self.constraints.unconstrained(i) {
                    continue;
                }
                let mut sum = 0.0;
                for (b, &j) in element.iter().enumerate().take(4) {
                    if self.constraints.unconstrained(j) {
                        sum += ke[a][b] * x[j];
                    }
                }
                y[i] += sum;
            }
        }

        for &i in self.constraints.constrained_indices() {
            y[i] = x[i];
        }
    }

    fn diagonal(&self, diag: &mut [f64]) -> Result<(), SpinozaError> {
        if diag.len() != self.size() {
            return Err(SpinozaError::OperatorError(
                "diagonal slice length mismatch".to_string(),
            ));
        }
        diag.fill(0.0);

        for element in &self.mesh.elements {
            let coords = [
                self.mesh.nodes[element[0]],
                self.mesh.nodes[element[1]],
                self.mesh.nodes[element[2]],
                self.mesh.nodes[element[3]],
            ];
            let (ke, _) = element_matrices_2d(coords, &|_, _| 0.0);

            for a in 0..4 {
                let i = element[a];
                if self.constraints.unconstrained(i) {
                    diag[i] += ke[a][a];
                }
            }
        }

        for &i in self.constraints.constrained_indices() {
            diag[i] = 1.0;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PoissonLaplacianMF3D {
    mesh: StructuredHexMesh3D,
    constraints: DirichletConstraints,
}

impl PoissonLaplacianMF3D {
    pub fn new(mesh: StructuredHexMesh3D, constraints: DirichletConstraints) -> Self {
        Self { mesh, constraints }
    }

    pub fn with_constant_dirichlet_boundary(mesh: StructuredHexMesh3D, value: f64) -> Self {
        let node_values: Vec<(usize, f64)> = mesh
            .boundary_nodes()
            .into_iter()
            .map(|idx| (idx, value))
            .collect();
        let constraints = DirichletConstraints::new(mesh.num_nodes(), &node_values);
        Self::new(mesh, constraints)
    }

    pub fn constraints(&self) -> &DirichletConstraints {
        &self.constraints
    }
}

impl LinearOperator for PoissonLaplacianMF3D {
    fn size(&self) -> usize {
        self.mesh.num_nodes()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.size());
        assert_eq!(y.len(), self.size());

        y.fill(0.0);

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
            let (ke, _) = element_matrices_3d(coords, &|_, _, _| 0.0);

            for a in 0..8 {
                let i = element[a];
                if !self.constraints.unconstrained(i) {
                    continue;
                }
                let mut sum = 0.0;
                for (b, &j) in element.iter().enumerate().take(8) {
                    if self.constraints.unconstrained(j) {
                        sum += ke[a][b] * x[j];
                    }
                }
                y[i] += sum;
            }
        }

        for &i in self.constraints.constrained_indices() {
            y[i] = x[i];
        }
    }

    fn diagonal(&self, diag: &mut [f64]) -> Result<(), SpinozaError> {
        if diag.len() != self.size() {
            return Err(SpinozaError::OperatorError(
                "diagonal slice length mismatch".to_string(),
            ));
        }
        diag.fill(0.0);

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
            let (ke, _) = element_matrices_3d(coords, &|_, _, _| 0.0);

            for a in 0..8 {
                let i = element[a];
                if self.constraints.unconstrained(i) {
                    diag[i] += ke[a][a];
                }
            }
        }

        for &i in self.constraints.constrained_indices() {
            diag[i] = 1.0;
        }
        Ok(())
    }
}

pub fn apply_dirichlet_rhs_correction_2d(
    mesh: &StructuredQuadMesh2D,
    rhs: &mut [f64],
    constraints: &DirichletConstraints,
) {
    for element in &mesh.elements {
        let coords = [
            mesh.nodes[element[0]],
            mesh.nodes[element[1]],
            mesh.nodes[element[2]],
            mesh.nodes[element[3]],
        ];
        let (ke, _) = element_matrices_2d(coords, &|_, _| 0.0);

        for a in 0..4 {
            let i = element[a];
            if !constraints.unconstrained(i) {
                continue;
            }
            for (b, &j) in element.iter().enumerate().take(4) {
                if let Some(vj) = constraints.value(j) {
                    rhs[i] -= ke[a][b] * vj;
                }
            }
        }
    }

    for &i in constraints.constrained_indices() {
        rhs[i] = constraints.value(i).unwrap_or(0.0);
    }
}

pub fn apply_dirichlet_rhs_correction_3d(
    mesh: &StructuredHexMesh3D,
    rhs: &mut [f64],
    constraints: &DirichletConstraints,
) {
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
        let (ke, _) = element_matrices_3d(coords, &|_, _, _| 0.0);

        for a in 0..8 {
            let i = element[a];
            if !constraints.unconstrained(i) {
                continue;
            }
            for (b, &j) in element.iter().enumerate().take(8) {
                if let Some(vj) = constraints.value(j) {
                    rhs[i] -= ke[a][b] * vj;
                }
            }
        }
    }

    for &i in constraints.constrained_indices() {
        rhs[i] = constraints.value(i).unwrap_or(0.0);
    }
}
