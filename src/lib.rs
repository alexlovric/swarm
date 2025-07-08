use std::time::Instant;

pub mod error;
pub mod initialisation;
pub mod nsga;
pub mod particle_swarm;

use crate::{
    error::Result,
    initialisation::Initialisation,
    nsga::{nsga, PolyMutationParams, SbxParams},
    particle_swarm::{particle_swarm, PsoParams},
};

/// Represents a variable with floating point bounds.
///
/// # Fields
/// * `0`: The value of the lower bound.
/// * `1`: The value of th upper bound.
#[derive(Debug, Clone)]
pub struct Variable(pub f64, pub f64);

/// Represents the top-level configuration for an optimisation algorithm.
///
/// This enum acts as a factory for creating and running different optimisation strategies,
/// such as NSGA-II or Particle Swarm Optimisation.
#[derive(Debug)]
pub enum Optimiser {
    /// Configures the NSGA-II (Non-dominated Sorting Genetic Algorithm II) optimiser.
    Nsga {
        /// The number of individuals in the population for each generation.
        /// This should be an even number, and typically >= 4.
        pop_size: usize,
        /// Parameters for the Simulated Binary Crossover (SBX) operator.
        crossover_params: SbxParams,
        /// Parameters for the Polynomial Mutation operator.
        mutation_params: PolyMutationParams,
        /// An optional seed for the random number generator to ensure reproducible results.
        seed: Option<u64>,
    },
    /// Configures the Particle Swarm Optimisation (PSO) optimiser.
    ParticleSwarm {
        /// The number of particles in the swarm.
        n_particles: usize,
        /// Parameters controlling the behaviour of the particles (e.g., inertia, social/cognitive factors).
        params: PsoParams,
        /// An optional constraint handler to apply to the solutions.
        constraint_handler: Option<ConstraintHandler>,
        /// An optional seed for the random number generator to ensure reproducible results.
        seed: Option<u64>,
    },
}

impl Optimiser {
    /// Solves an optimisation problem using the configured algorithm (SERIAL VERSION).
    ///
    /// This method is compiled by default. It accepts a mutable closure (`FnMut`),
    /// allowing the objective function to maintain its own internal state between calls.
    ///
    /// # Arguments
    /// * `func` - A mutable closure that takes a slice of `f64` (the variables) and returns
    ///   the objective values and optional constraint values.
    /// * `vars` - A slice defining the bounds for each variable in the search space.
    /// * `max_iter` - The maximum number of iterations or generations to run the optimiser for.
    ///
    /// # Returns
    /// A `Result` containing the `OptimiserResult` on success, or an error string on failure.
    #[cfg(not(feature = "parallel"))]
    pub fn solve<F>(
        &self,
        func: &mut F,
        vars: &[Variable],
        max_iter: usize,
    ) -> Result<OptimiserResult>
    where
        F: FnMut(&[f64]) -> (Vec<f64>, Option<Vec<f64>>),
    {
        let now = Instant::now();
        let mut result = match &self {
            Optimiser::Nsga {
                pop_size,
                crossover_params,
                mutation_params,
                seed,
            } => nsga(
                func,
                vars,
                max_iter,
                *pop_size,
                crossover_params,
                mutation_params,
                Initialisation::LatinHypercube { centred: true },
                None,
                *seed,
            )?,
            Optimiser::ParticleSwarm {
                n_particles,
                params,
                constraint_handler,
                seed,
            } => particle_swarm(
                func,
                vars,
                max_iter,
                *n_particles,
                params,
                Initialisation::LatinHypercube { centred: true },
                *constraint_handler,
                *seed,
            )?,
        };

        result.execution_time = now.elapsed().as_secs_f64();
        Ok(result)
    }

    /// Solves an optimisation problem using the configured algorithm (PARALLEL VERSION).
    ///
    /// This method is compiled only when the `parallel` feature is enabled.
    /// It requires a thread-safe closure (`Fn + Sync + Send`).
    /// See `solve` [Serial version] for more details
    #[cfg(feature = "parallel")]
    pub fn solve<F>(&self, func: &F, vars: &[Variable], max_iter: usize) -> Result<OptimiserResult>
    where
        F: Fn(&[f64]) -> (Vec<f64>, Option<Vec<f64>>) + Sync + Send,
    {
        let now = Instant::now();
        let mut result = match self {
            Optimiser::Nsga {
                pop_size,
                crossover_params,
                mutation_params,
                seed,
            } => nsga(
                func,
                vars,
                max_iter,
                *pop_size,
                crossover_params,
                mutation_params,
                Initialisation::LatinHypercube { centred: true },
                None,
                *seed,
            )?,
            Optimiser::ParticleSwarm {
                n_particles,
                params,
                constraint_handler,
                seed,
            } => particle_swarm(
                func,
                vars,
                max_iter,
                *n_particles,
                params,
                Initialisation::LatinHypercube { centred: true },
                *constraint_handler,
                *seed,
            )?,
        };

        result.execution_time = now.elapsed().as_secs_f64();
        Ok(result)
    }
}

/// Contains the results of a successful optimisation run.
#[derive(Debug, Clone)]
pub struct OptimiserResult {
    /// For multi-objective problems, this contains the set of non-dominated solutions (the Pareto front).
    /// For single-objective problems, it will typically contain a single best solution.
    pub solutions: Vec<Solution>,
    /// The number of iterations the optimiser ran for.
    pub n_iterations: usize,
    /// The total time taken for the `solve` method to execute, in seconds.
    pub execution_time: f64,
}

impl OptimiserResult {
    /// Creates a new `OptimiserResult`.
    pub fn new(solutions: Vec<Solution>, n_iterations: usize) -> Self {
        OptimiserResult {
            solutions,
            n_iterations,
            execution_time: 0.0,
        }
    }

    /// Finds the best solution in the Pareto front based on a single objective.
    pub fn best_solution(&self) -> Option<&Solution> {
        if self.solutions.is_empty() {
            return None;
        }
        if self.solutions.len() == 1 {
            return Some(&self.solutions[0]);
        }
        self.solutions.iter().min_by(|a, b| {
            let f_a = a.f.first().unwrap_or(&f64::INFINITY);
            let f_b = b.f.first().unwrap_or(&f64::INFINITY);
            f_a.partial_cmp(f_b).unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// Represents a single solution found by an optimiser.
#[derive(Debug, Clone)]
pub struct Solution {
    /// The vector of variable values for this solution.
    pub x: Vec<f64>,
    /// The vector of objective function values for this solution.
    pub f: Vec<f64>,
    /// An optional vector of constraint values for this solution.
    pub g: Option<Vec<f64>>,
}

impl Solution {
    /// Creates a new `Solution`.
    ///
    /// # Arguments
    /// * `x` - The vector of variable values for this solution.
    /// * `f` - The vector of objective function values for this solution.
    /// * `g` - An optional vector of constraint values for this solution.
    ///
    /// # Returns
    /// A new `Solution` instance.
    pub fn new(x: Vec<f64>, f: Vec<f64>, g: Option<Vec<f64>>) -> Self {
        Solution { x, f, g }
    }
}

/// Defines the strategy for handling constraints.
#[derive(Debug, Clone, Copy)]
pub enum ConstraintHandler {
    /// Applies a quadratic penalty for violated constraints.
    Penalty { multiplier: f64 },
}

impl ConstraintHandler {
    /// Calculates the total penalty based on the constraint values.
    ///
    /// This implementation assumes that a constraint `c` is violated if `c > 0`.
    /// The penalty is the sum of the squares of the violated constraints, scaled by a multiplier.
    ///
    /// # Arguments
    /// * `constraints` - An optional vector of constraint values for this solution.
    ///
    /// # Returns
    /// The total penalty based on the constraint values.
    fn calculate_penalty(&self, constraints: &Option<Vec<f64>>) -> f64 {
        match self {
            ConstraintHandler::Penalty { multiplier } => {
                constraints
                    .as_deref() // Get a slice `&[f64]` from `&Option<Vec<f64>>`
                    .unwrap_or(&[]) // Default to an empty slice if None
                    .iter()
                    .filter(|&&c| c > 0.0) // Only penalize violations
                    .map(|&c| c * c) // Use quadratic penalty
                    .sum::<f64>()
                    * multiplier
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    // A simple quadratic test function: f(x) = (x - 5)^2
    fn quadratic_problem(x: &[f64]) -> (Vec<f64>, Option<Vec<f64>>) {
        (vec![(x[0] - 5.0).powi(2)], None)
    }

    #[test]
    fn test_optimiser_solve_nsga_dispatch() {
        let optimizer = Optimiser::Nsga {
            pop_size: 10,
            crossover_params: SbxParams::default(),
            mutation_params: PolyMutationParams::default(),
            seed: Some(1),
        };
        let vars = vec![Variable(0.0, 10.0)];
        let mut func = |x: &[f64]| quadratic_problem(x);

        let result = optimizer.solve(&mut func, &vars, 10);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.n_iterations, 10);
        assert_abs_diff_eq!(result.solutions[0].x[0], 4.999, epsilon = 1e-3);
    }

    #[test]
    fn test_optimiser_solve_pso_dispatch() {
        let optimizer = Optimiser::ParticleSwarm {
            n_particles: 10,
            params: PsoParams::default(),
            constraint_handler: None,
            seed: Some(1),
        };
        let vars = vec![Variable(0.0, 10.0)];
        let mut func = |x: &[f64]| quadratic_problem(x);

        let result = optimizer.solve(&mut func, &vars, 10);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.n_iterations, 10);

        #[cfg(not(feature = "parallel"))]
        assert_abs_diff_eq!(result.solutions[0].x[0], 4.991, epsilon = 1e-3);

        #[cfg(feature = "parallel")]
        assert_abs_diff_eq!(result.solutions[0].x[0], 4.982, epsilon = 1e-3);
    }
}
