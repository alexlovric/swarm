use pyo3::{exceptions::PyValueError, prelude::*};

use swarm::{ConstraintHandler, Optimiser, Variable};

use crate::{
    nsga::{PyPolyMutationParams, PySbxParams},
    pso::PyPsoParams,
};

pub mod nsga;
pub mod pso;

#[pyclass(name = "Variable")]
#[derive(Clone)]
struct PyVariable(Variable);

#[pymethods]
impl PyVariable {
    #[new]
    fn new(min: f64, max: f64) -> Self {
        Self(Variable(min, max))
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

#[pyclass(name = "Optimiser")]
pub struct PyOptimiser(Optimiser);

#[pymethods]
impl PyOptimiser {
    /// Creates a new NSGA-II optimiser instance.
    #[staticmethod]
    fn nsga2(
        pop_size: usize,
        crossover_params: &PySbxParams,
        mutation_params: &PyPolyMutationParams,
        seed: Option<u64>,
    ) -> Self {
        let optimiser = Optimiser::Nsga {
            pop_size,
            crossover_params: crossover_params.0,
            mutation_params: mutation_params.0,
            seed,
        };
        Self(optimiser)
    }

    /// Creates a new Particle Swarm optimiser instance.
    #[staticmethod]
    fn particle_swarm(
        n_particles: usize,
        params: &PyPsoParams,
        penalty_multiplier: Option<f64>,
        seed: Option<u64>,
    ) -> Self {
        let optimiser = Optimiser::ParticleSwarm {
            n_particles,
            params: params.0,
            constraint_handler: match penalty_multiplier {
                Some(multiplier) => Some(ConstraintHandler::Penalty { multiplier }),
                None => None,
            },
            seed,
        };
        Self(optimiser)
    }

    /// Solves an optimisation problem using the configured algorithm.
    ///
    /// Args:
    ///     func (callable): The objective function to optimise. It must accept a list of floats
    ///         and return a tuple containing two elements: a list of objective values (floats),
    ///         and either a list of constraint values (floats) or None.
    ///     vars (list[Variable]): A list defining the bounds for each variable.
    ///     max_iter (int): The maximum number of iterations to run.
    ///
    /// Returns:
    ///     OptimiserResult: An object containing the results of the optimisation.
    fn solve(
        &self,
        py: Python,
        func: PyObject,
        vars: Vec<PyVariable>,
        max_iter: usize,
    ) -> PyResult<PyOptimiserResult> {
        // --- Type Conversion: Python -> Rust ---
        let rust_vars: Vec<Variable> = vars.into_iter().map(|v| v.0).collect();

        // --- Create a Rust closure that calls the Python function ---
        // This closure captures the `py` token, allowing it to call Python code.
        let mut objective_closure = |x: &[f64]| -> (Vec<f64>, Option<Vec<f64>>) {
            let args = (x,);
            let result = func.call1(py, args);

            match result {
                Ok(res) => res.extract(py).expect("Python function did not return a valid tuple of (list[float], list[float] | None)"),
                Err(e) => {
                    // An error in the user's Python code is a fatal error for the optimisation.
                    // We print the Python traceback and panic the Rust thread.
                    e.print(py);
                    panic!("Python objective function failed during optimisation.");
                }
            }
        };

        // --- Call the core Rust library's solve method ---
        // We do NOT use `py.allow_threads` because the `objective_closure` needs to call
        // back into Python, which requires holding the GIL. The serial version of the
        // optimiser runs on a single thread, so this is safe.
        let result = self.0.solve(&mut objective_closure, &rust_vars, max_iter);

        // --- Handle potential errors from the Rust optimiser ---
        let swarm_result = match result {
            Ok(res) => res,
            Err(e) => return Err(PyValueError::new_err(e.to_string())),
        };

        // --- Type Conversion: Rust -> Python ---
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

/// A Python module implementing bindings for the Swarm optimisation library.
///
/// This function defines the module content and can be used to rename the module
/// in Python using the `name` attribute.
#[pymodule]
#[pyo3(name = "swarm")] // This renames the module to `swarm` in Python
fn swarm_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOptimiser>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<PySolution>()?;
    m.add_class::<PyOptimiserResult>()?;
    m.add_class::<PyPsoParams>()?;
    m.add_class::<PySbxParams>()?;
    m.add_class::<PyPolyMutationParams>()?;
    Ok(())
}
