use pyo3::prelude::*;

use swarm::pso::PsoParams;

#[pyclass(name = "PsoParams")]
pub struct PyPsoParams(pub(crate) PsoParams);

#[pymethods]
impl PyPsoParams {
    #[new]
    fn new(inertia: f64, cognitive_coeff: f64, social_coeff: f64) -> Self {
        Self(PsoParams { inertia, cognitive_coeff, social_coeff })
    }
}