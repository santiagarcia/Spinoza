//! Block Jacobi preconditioner for 2×2 block systems.
//!
//! Applies the diagonal-block inverse approximation:
//!
//! ```text
//! M^{-1} = [ P0   0  ]
//!          [ 0   P1  ]
//! ```

use spinoza_core::Preconditioner;

/// Block Jacobi preconditioner wrapping two independent sub-preconditioners,
/// one per diagonal block.
pub struct BlockJacobiPreconditioner {
    p0: Box<dyn Preconditioner>,
    p1: Box<dyn Preconditioner>,
    n0: usize,
    n1: usize,
}

impl BlockJacobiPreconditioner {
    pub fn new(
        p0: Box<dyn Preconditioner>,
        p1: Box<dyn Preconditioner>,
        n0: usize,
        n1: usize,
    ) -> Self {
        Self { p0, p1, n0, n1 }
    }

    pub fn n0(&self) -> usize {
        self.n0
    }

    pub fn n1(&self) -> usize {
        self.n1
    }
}

impl Preconditioner for BlockJacobiPreconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        assert_eq!(r.len(), self.n0 + self.n1);
        assert_eq!(z.len(), self.n0 + self.n1);

        // Apply P0 to block-0 of residual
        self.p0.apply(&r[..self.n0], &mut z[..self.n0]);

        // Apply P1 to block-1 of residual
        self.p1.apply(&r[self.n0..], &mut z[self.n0..]);
    }
}
