use std::collections::BTreeMap;

use spinoza_core::{LinearOperator, SpinozaError};

#[derive(Debug, Clone)]
pub struct SparseMatrixBuilder {
    nrows: usize,
    ncols: usize,
    rows: Vec<BTreeMap<usize, f64>>,
}

impl SparseMatrixBuilder {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            rows: vec![BTreeMap::new(); nrows],
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn add_entry(&mut self, row: usize, col: usize, value: f64) {
        if row >= self.nrows || col >= self.ncols {
            panic!("sparse entry index out of bounds");
        }
        let entry = self.rows[row].entry(col).or_insert(0.0);
        *entry += value;
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.rows[row].get(&col).copied().unwrap_or(0.0)
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        if row >= self.nrows || col >= self.ncols {
            panic!("sparse set index out of bounds");
        }
        if value == 0.0 {
            self.rows[row].remove(&col);
        } else {
            self.rows[row].insert(col, value);
        }
    }

    pub fn rows_mut(&mut self) -> &mut [BTreeMap<usize, f64>] {
        &mut self.rows
    }

    pub fn to_csr(&self) -> CsrMatrix {
        let mut row_ptr = Vec::with_capacity(self.nrows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        row_ptr.push(0);
        for row in &self.rows {
            for (col, value) in row {
                col_idx.push(*col);
                values.push(*value);
            }
            row_ptr.push(col_idx.len());
        }

        CsrMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            row_ptr,
            col_idx,
            values,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl CsrMatrix {
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.ncols);
        let mut out = vec![0.0; self.nrows];

        for (row, out_row) in out.iter_mut().enumerate().take(self.nrows) {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            let mut sum = 0.0;
            for idx in start..end {
                sum += self.values[idx] * x[self.col_idx[idx]];
            }
            *out_row = sum;
        }

        out
    }

    pub fn diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0.0; self.nrows];
        for (row, diag_row) in diag.iter_mut().enumerate().take(self.nrows) {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for idx in start..end {
                if self.col_idx[idx] == row {
                    *diag_row = self.values[idx];
                    break;
                }
            }
        }
        diag
    }
}

impl LinearOperator for CsrMatrix {
    fn size(&self) -> usize {
        self.nrows
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.ncols);
        assert_eq!(y.len(), self.nrows);

        for yi in y.iter_mut() {
            *yi = 0.0;
        }

        for (row, yi) in y.iter_mut().enumerate().take(self.nrows) {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            let mut sum = 0.0;
            for idx in start..end {
                sum += self.values[idx] * x[self.col_idx[idx]];
            }
            *yi = sum;
        }
    }

    fn diagonal(&self, diag: &mut [f64]) -> Result<(), SpinozaError> {
        if diag.len() != self.nrows {
            return Err(SpinozaError::OperatorError(
                "diagonal slice length mismatch".to_string(),
            ));
        }
        for (dst, val) in diag.iter_mut().zip(self.diagonal()) {
            *dst = val;
        }
        Ok(())
    }
}
