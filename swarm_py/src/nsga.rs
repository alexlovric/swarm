use pyo3::prelude::*;

use swarm::nsga::{PolyMutationParams, SbxParams};

#[pyclass(name = "SbxParams")]
pub struct PySbxParams(pub(crate) SbxParams);

#[pymethods]
impl PySbxParams {
    #[new]
    fn new(prob: f64, eta: f64) -> Self {
        Self(SbxParams { prob, eta })
    }
}

#[pyclass(name = "PolyMutationParams")]
pub struct PyPolyMutationParams(pub(crate) PolyMutationParams);

#[pymethods]
impl PyPolyMutationParams {
    #[new]
    fn new(prob: f64, eta: f64) -> Self {
        Self(PolyMutationParams { prob, eta })
    }
}
