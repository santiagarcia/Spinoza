//! Linear operator interfaces for matrix based and matrix free solvers.

use crate::SpinozaError;

/// A linear operator that can apply y = A x without allocating.
pub trait LinearOperator {
    /// Number of rows and columns for this square operator.
    fn size(&self) -> usize;

    /// Apply the operator as `y = A x`.
    ///
    /// Implementations must not allocate in this function.
    fn apply(&self, x: &[f64], y: &mut [f64]);

    /// Fill the Jacobi diagonal in `diag`.
    ///
    /// The default implementation returns an error when diagonal extraction
    /// is not implemented by the operator.
    fn diagonal(&self, _diag: &mut [f64]) -> Result<(), SpinozaError> {
        Err(SpinozaError::OperatorError(
            "operator diagonal extraction is not supported".to_string(),
        ))
    }
}
