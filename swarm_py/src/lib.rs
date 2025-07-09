use pyo3::{exceptions::PyValueError, prelude::*};

use swarm::{
    error::Result,
    nsga::{PmParams, SbxParams},
    pso::PsoParams,
    ConstraintHandler, Optimiser, OptimiserResult, Variable,
};

#[pyclass(name = "Optimiser")]
pub struct PyOptimiser(Optimiser);

#[pymethods]
impl PyOptimiser {
    /// Creates a new NSGA-II optimiser instance.
    #[staticmethod]
    #[pyo3(signature = (pop_size, crossover = None, mutation = None, seed = None))]
    fn nsga(
        pop_size: usize,
        crossover: Option<&PySbxParams>,
        mutation: Option<&PyPmParams>,
        seed: Option<u64>,
    ) -> Self {
        let optimiser = Optimiser::Nsga {
            pop_size,
            crossover: crossover
                .unwrap_or(&PySbxParams(SbxParams::default()))
                .0,
            mutation: mutation
                .unwrap_or(&PyPmParams(PmParams::default()))
                .0,
            seed,
        };
        Self(optimiser)
    }

    /// Creates a new Particle Swarm optimiser instance.
    #[staticmethod]
    #[pyo3(signature = (n_particles, params = None, penalty_multiplier = None, seed = None))]
    fn pso(
        n_particles: usize,
        params: Option<&PyPsoParams>,
        penalty_multiplier: Option<f64>,
        seed: Option<u64>,
    ) -> Self {
        let optimiser = Optimiser::Pso {
            n_particles,
            params: params.unwrap_or(&PyPsoParams(PsoParams::default())).0,
            constraint_handler: match penalty_multiplier {
                Some(multiplier) => Some(ConstraintHandler::Penalty { multiplier }),
                None => None,
            },
            seed,
        };
        Self(optimiser)
    }

    /// Solves an optimisation problem using a single thread.
    fn solve(
        &self,
        py: Python,
        func: PyObject,
        vars: Vec<PyVariable>,
        max_iter: usize,
    ) -> PyResult<PyOptimiserResult> {
        let rust_vars: Vec<Variable> = vars.into_iter().map(|v| v.0).collect();

        // The closure for the sequential solver. It captures the GIL context `py`.
        let mut blackbox = |x: &[f64]| -> (Vec<f64>, Option<Vec<f64>>) {
            Self::_call_python_blackbox(py, &func, x)
        };

        // Call the underlying sequential solver from the Rust core.
        let result = self.0.solve(&mut blackbox, &rust_vars, max_iter);

        self._process_result(result)
    }

    /// Solves an optimisation problem in parallel using multiple threads.
    fn solve_par(
        &self,
        py: Python,
        func: PyObject,
        vars: Vec<PyVariable>,
        max_iter: usize,
    ) -> PyResult<PyOptimiserResult> {
        let rust_vars: Vec<Variable> = vars.into_iter().map(|v| v.0).collect();

        // The closure for the parallel solver. It must be `Sync`.
        let blackbox = |x: &[f64]| -> (Vec<f64>, Option<Vec<f64>>) {
            Python::with_gil(|py| Self::_call_python_blackbox(py, &func, x))
        };

        // Release the GIL to allow the underlying parallel Rust code to run.
        let result = py.allow_threads(|| self.0.solve_par(&blackbox, &rust_vars, max_iter));

        self._process_result(result)
    }

    /// Returns a string representation of the configured optimiser.
    fn __repr__(&self) -> String {
        match &self.0 {
            Optimiser::Nsga {
                pop_size,
                crossover,
                mutation,
                seed,
            } => {
                format!(
                    "Optimiser(NSGA-II: pop_size={}, crossover={:?}, mutation={:?}, seed={})",
                    pop_size,
                    crossover,
                    mutation,
                    seed.map_or_else(|| "None".to_string(), |s| s.to_string())
                )
            }
            Optimiser::Pso {
                n_particles,
                params,
                constraint_handler: _,
                seed,
            } => {
                format!(
                    "Optimiser(PSO: n_particles={}, params={:?}, seed={})",
                    n_particles,
                    params,
                    seed.map_or_else(|| "None".to_string(), |s| s.to_string())
                )
            }
        }
    }
}

/// Private helper methods for PyOptimiser.
impl PyOptimiser {
    /// A private helper to call the Python blackbox function and handle errors.
    fn _call_python_blackbox(
        py: Python,
        func: &PyObject,
        x: &[f64],
    ) -> (Vec<f64>, Option<Vec<f64>>) {
        let args = (x,);
        let result = func.call1(py, args);
        match result {
            Ok(res) => res.extract(py).expect(
                "Python function did not return a valid tuple of (list[float], list[float] | None)",
            ),
            Err(e) => {
                e.print(py);
                // A panic here will stop the optimisation immediately.
                panic!("Python blackbox function failed during optimisation.");
            }
        }
    }

    /// Processes the result from the core optimiser, converting it into a PyResult.
    fn _process_result(&self, result: Result<OptimiserResult>) -> PyResult<PyOptimiserResult> {
        let swarm_result = match result {
            Ok(res) => res,
            Err(e) => return Err(PyValueError::new_err(e.to_string())),
        };

        let py_solutions = swarm_result
            .solutions
            .into_iter()
            .map(|sol| PySolution {
                x: sol.x,
                f: sol.f,
                g: sol.g,
            })
            .collect();

        Ok(PyOptimiserResult {
            solutions: py_solutions,
            n_iterations: swarm_result.n_iterations,
            execution_time: swarm_result.execution_time,
        })
    }
}

#[pyclass(name = "Variable")]
#[derive(Clone)]
struct PyVariable(Variable);

#[pymethods]
impl PyVariable {
    #[new]
    fn new(min: f64, max: f64) -> Self {
        if min > max {
            panic!("'min' cannot be greater than 'max'");
        }
        Self(Variable(min, max))
    }

    fn __repr__(&self) -> String {
        format!("Variable(min={}, max={})", self.0 .0, self.0 .1)
    }
}

#[pyclass(name = "Solution")]
#[derive(Clone)]
struct PySolution {
    #[pyo3(get)]
    x: Vec<f64>,
    #[pyo3(get)]
    f: Vec<f64>,
    #[pyo3(get)]
    g: Option<Vec<f64>>,
}

#[pymethods]
impl PySolution {
    fn __repr__(&self) -> String {
        let x_str = format!(
            "[{}]",
            self.x
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let f_str = format!(
            "[{}]",
            self.f
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(", ")
        );

        match &self.g {
            Some(g_vec) => {
                let g_str = format!(
                    "[{}]",
                    g_vec
                        .iter()
                        .map(|v| format!("{:.4}", v))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                format!("Solution(x={}, f={}, g={})", x_str, f_str, g_str)
            }
            None => format!("Solution(x={}, f={})", x_str, f_str),
        }
    }
}

#[pyclass(name = "OptimiserResult")]
#[derive(Clone)]
struct PyOptimiserResult {
    #[pyo3(get)]
    solutions: Vec<PySolution>,
    #[pyo3(get)]
    n_iterations: usize,
    #[pyo3(get)]
    execution_time: f64,
}

#[pymethods]
impl PyOptimiserResult {
    fn __repr__(&self) -> String {
        format!(
            "<OptimiserResult(solutions={}, n_iterations={}, execution_time={:.4}s)>",
            self.solutions.len(),
            self.n_iterations,
            self.execution_time
        )
    }
}

#[pyclass(name = "PsoParams")]
pub struct PyPsoParams(pub(crate) PsoParams);

#[pymethods]
impl PyPsoParams {
    #[new]
    fn new(inertia: f64, cognitive_coeff: f64, social_coeff: f64) -> Self {
        Self(PsoParams {
            inertia,
            cognitive_coeff,
            social_coeff,
        })
    }
}

#[pyclass(name = "SbxParams")]
pub struct PySbxParams(pub(crate) SbxParams);

#[pymethods]
impl PySbxParams {
    #[new]
    fn new(prob: f64, eta: f64) -> Self {
        Self(SbxParams { prob, eta })
    }
}

#[pyclass(name = "PmParams")]
pub struct PyPmParams(pub(crate) PmParams);

#[pymethods]
impl PyPmParams {
    #[new]
    fn new(prob: f64, eta: f64) -> Self {
        Self(PmParams { prob, eta })
    }
}

/// A Python module implementing bindings for the Swarm optimisation library.
///
/// This function defines the module content and can be used to rename the module
/// in Python using the `name` attribute.
#[pymodule]
#[pyo3(name = "_swarm")] // This renames the module to `swarm` in Python
fn swarm_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOptimiser>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<PySolution>()?;
    m.add_class::<PyOptimiserResult>()?;
    m.add_class::<PyPsoParams>()?;
    m.add_class::<PySbxParams>()?;
    m.add_class::<PyPmParams>()?;
    Ok(())
}
