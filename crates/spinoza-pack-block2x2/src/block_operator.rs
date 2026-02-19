//! 2×2 block linear operator: wraps four sub-operators and applies block
//! action on a single contiguous vector without allocations (reuses
//! pre-allocated workspace).

use std::cell::RefCell;

use spinoza_core::{BlockLayout, LinearOperator, SpinozaError};

/// A 2×2 block operator:
///
/// ```text
/// [ A00  A01 ] [ x0 ]   [ y0 ]
/// [ A10  A11 ] [ x1 ] = [ y1 ]
/// ```
///
/// Operates on contiguous vectors of size `n0 + n1`.
pub struct BlockOperator2x2 {
    a00: Box<dyn LinearOperator>,
    a01: Box<dyn LinearOperator>,
    a10: Box<dyn LinearOperator>,
    a11: Box<dyn LinearOperator>,
    n0: usize,
    n1: usize,
    /// Pre-allocated workspace to avoid allocation in apply.
    tmp: RefCell<Vec<f64>>,
}

impl BlockOperator2x2 {
    pub fn new(
        a00: Box<dyn LinearOperator>,
        a01: Box<dyn LinearOperator>,
        a10: Box<dyn LinearOperator>,
        a11: Box<dyn LinearOperator>,
    ) -> Self {
        let n0 = a00.size();
        let n1 = a11.size();
        assert_eq!(a01.size(), n0, "A01 must have same row dimension as A00");
        assert_eq!(a10.size(), n1, "A10 must have same row dimension as A11");
        let tmp = RefCell::new(vec![0.0; n0.max(n1)]);
        Self {
            a00,
            a01,
            a10,
            a11,
            n0,
            n1,
            tmp,
        }
    }

    pub fn layout(&self) -> BlockLayout {
        BlockLayout {
            block_sizes: vec![self.n0, self.n1],
        }
    }

    pub fn n0(&self) -> usize {
        self.n0
    }

    pub fn n1(&self) -> usize {
        self.n1
    }
}

impl LinearOperator for BlockOperator2x2 {
    fn size(&self) -> usize {
        self.n0 + self.n1
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n0 + self.n1);
        assert_eq!(y.len(), self.n0 + self.n1);

        let x0 = &x[..self.n0];
        let x1 = &x[self.n0..];
        let (y0, y1) = y.split_at_mut(self.n0);

        // y0 = A00 * x0
        self.a00.apply(x0, y0);
        // y0 += A01 * x1
        {
            let mut tmp = self.tmp.borrow_mut();
            if tmp.len() < self.n0 {
                tmp.resize(self.n0, 0.0);
            }
            let buf = &mut tmp[..self.n0];
            self.a01.apply(x1, buf);
            for (yi, ti) in y0.iter_mut().zip(buf.iter()) {
                *yi += *ti;
            }
        }

        // y1 = A10 * x0
        self.a10.apply(x0, y1);
        // y1 += A11 * x1
        {
            let mut tmp = self.tmp.borrow_mut();
            if tmp.len() < self.n1 {
                tmp.resize(self.n1, 0.0);
            }
            let buf = &mut tmp[..self.n1];
            self.a11.apply(x1, buf);
            for (yi, ti) in y1.iter_mut().zip(buf.iter()) {
                *yi += *ti;
            }
        }
    }

    fn diagonal(&self, diag: &mut [f64]) -> Result<(), SpinozaError> {
        if diag.len() != self.n0 + self.n1 {
            return Err(SpinozaError::OperatorError(
                "block diagonal slice length mismatch".to_string(),
            ));
        }
        self.a00.diagonal(&mut diag[..self.n0])?;
        self.a11.diagonal(&mut diag[self.n0..])?;
        Ok(())
    }
}
