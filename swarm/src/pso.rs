use rand::prelude::*;
use rand::rngs::StdRng;

use crate::{
    error::Result, initialisation::Initialisation, ConstraintHandler, OptimiserResult, Solution,
    Variable,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A helper struct to hold the parameters for the PSO algorithm.
#[derive(Debug, Copy, Clone)]
pub struct PsoParams {
    pub inertia: f64,
    pub cognitive_coeff: f64,
    pub social_coeff: f64,
}

impl Default for PsoParams {
    /// Default values for the PSO parameters.
    fn default() -> Self {
        PsoParams {
            inertia: 0.5,
            cognitive_coeff: 1.5,
            social_coeff: 1.5,
        }
    }
}

#[derive(Clone, Debug)]
/// A struct representing a particle in the swarm.
struct Particle {
    position: Vec<f64>,
    velocity: Vec<f64>,
    best_position: Vec<f64>,
    best_fitness: f64,
}

/// The main Particle Swarm Optimisation function (SERIAL VERSION).
/// This version is compiled when the "parallel" feature is NOT enabled.
///
/// # Arguments
/// * `func` - A mutable closure that takes a slice of `f64` (the variables) and returns
///   the objective values and optional constraint values.
/// * `vars` - A slice defining the bounds for each variable in the search space.
/// * `max_iter` - The maximum number of iterations or generations to run the optimiser for.
/// * `n_particles` - The number of particles in the swarm.
/// * `params` - Parameters controlling the behaviour of the particles (e.g., inertia, social/cognitive factors).
/// * `initialisation` - The initialisation strategy to use for generating the swarm.
/// * `constraint_handler` - The constraint handler to use for handling constraint violations.
/// * `seed` - An optional seed for the random number generator to ensure reproducible results.
///
/// # Returns
/// A `Result` containing the `OptimiserResult` on success, or an error string on failure.
#[cfg(not(feature = "parallel"))]
pub fn particle_swarm<F>(
    func: &mut F,
    vars: &[Variable],
    max_iter: usize,
    n_particles: usize,
    params: &PsoParams,
    initialisation: Initialisation,
    constraint_handler: Option<ConstraintHandler>,
    seed: Option<u64>,
) -> Result<OptimiserResult>
where
    F: FnMut(&[f64]) -> (Vec<f64>, Option<Vec<f64>>),
{
    let num_dimensions = vars.len();
    let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);

    // Generate all starting positions at once using the chosen strategy.
    let initial_positions = initialisation.generate_samples(n_particles, vars, &mut rng);

    let mut swarm: Vec<Particle> = initial_positions
        .into_iter()
        .map(|position| {
            let velocity = vec![0.0; num_dimensions];
            let (objectives, constraints) = func(&position);
            let fitness = match constraint_handler {
                Some(constraint_handler) => {
                    objectives.iter().sum::<f64>()
                        + constraint_handler.calculate_penalty(&constraints)
                }
                None => objectives.iter().sum::<f64>(),
            };

            Particle {
                position: position.clone(),
                velocity,
                best_position: position,
                best_fitness: fitness,
            }
        })
        .collect();

    // Initialise global best from the initial swarm
    let (global_best_position, global_best_fitness) = swarm
        .iter()
        .min_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap())
        .map(|p| (p.best_position.clone(), p.best_fitness))
        .unwrap_or((vec![], f64::INFINITY));

    let mut best_known_position = global_best_position;
    let mut best_known_fitness = global_best_fitness;

    // Optimization Loop
    for _ in 0..max_iter {
        // Evaluate Fitness and Update Bests
        for particle in &mut swarm {
            let (objectives, constraints) = func(&particle.position);
            let fitness = match constraint_handler {
                Some(constraint_handler) => {
                    objectives.iter().sum::<f64>()
                        + constraint_handler.calculate_penalty(&constraints)
                }
                None => objectives.iter().sum::<f64>(),
            };

            // Update particle's personal best
            if fitness < particle.best_fitness {
                particle.best_fitness = fitness;
                particle.best_position = particle.position.clone();
            }

            // Update global best
            if fitness < best_known_fitness {
                best_known_fitness = fitness;
                best_known_position = particle.position.clone();
            }
        }

        // Update Particle Velocities and Positions
        for particle in &mut swarm {
            for i in 0..num_dimensions {
                let r1: f64 = rng.gen();
                let r2: f64 = rng.gen();

                // Velocity update
                let cognitive_component = params.cognitive_coeff
                    * r1
                    * (particle.best_position[i] - particle.position[i]);
                let social_component =
                    params.social_coeff * r2 * (best_known_position[i] - particle.position[i]);
                particle.velocity[i] = (params.inertia * particle.velocity[i])
                    + cognitive_component
                    + social_component;

                // Position update
                particle.position[i] += particle.velocity[i];

                // Boundary handling
                if particle.position[i] < vars[i].0 {
                    particle.position[i] = vars[i].0;
                    particle.velocity[i] = 0.0; // Reset velocity to prevent sticking
                } else if particle.position[i] > vars[i].1 {
                    particle.position[i] = vars[i].1;
                    particle.velocity[i] = 0.0; // Reset velocity
                }
            }
        }
    }

    // Final Result
    // Re-evaluate the best found position to get the final objectives and constraints
    let (best_objectives, best_constraints) = func(&best_known_position);
    Ok(OptimiserResult::new(
        vec![Solution::new(
            best_known_position,
            best_objectives,
            best_constraints,
        )],
        max_iter, // Returning max_iter as n_iterations
    ))
}

/// The main Particle Swarm Optimisation function (PARALLEL VERSION).
/// This version is compiled ONLY when the "parallel" feature is enabled.
#[cfg(feature = "parallel")]
pub fn particle_swarm<F>(
    func: &F, // Note: F is now Fn, not FnMut
    vars: &[Variable],
    n_particles: usize,
    max_iter: usize,
    params: &PsoParams,
    initialisation: Initialisation,
    constraint_handler: Option<ConstraintHandler>,
    seed: Option<u64>,
) -> Result<OptimiserResult>
where
    F: Fn(&[f64]) -> (Vec<f64>, Option<Vec<f64>>) + Sync + Send,
{
    let num_dimensions = vars.len();
    let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);

    let initial_positions = initialisation.generate_samples(n_particles, vars, &mut rng);

    let mut swarm: Vec<Particle> = initial_positions
        .into_par_iter() // Parallel initialization
        .map(|position| {
            let velocity = vec![0.0; num_dimensions];
            let (objectives, constraints) = func(&position);
            let fitness = match constraint_handler {
                Some(constraint_handler) => {
                    objectives.iter().sum::<f64>()
                        + constraint_handler.calculate_penalty(&constraints)
                }
                None => objectives.iter().sum::<f64>(),
            };
            Particle {
                position: position.clone(),
                velocity,
                best_position: position,
                best_fitness: fitness,
            }
        })
        .collect();

    let (mut best_known_position, mut best_known_fitness) = swarm
        .iter()
        .min_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap())
        .map(|p| (p.best_position.clone(), p.best_fitness))
        .unwrap_or((vec![], f64::INFINITY));

    for _ in 0..max_iter {
        // Step A: Evaluate fitness for all particles in parallel.
        let fitness_values: Vec<f64> = swarm
            .par_iter()
            .map(|p| {
                let (objectives, constraints) = func(&p.position);
                match constraint_handler {
                    Some(constraint_handler) => {
                        objectives.iter().sum::<f64>()
                            + constraint_handler.calculate_penalty(&constraints)
                    }
                    None => objectives.iter().sum::<f64>(),
                }
            })
            .collect();

        // Step B: Update personal and global bests sequentially. This is very fast.
        for (i, particle) in swarm.iter_mut().enumerate() {
            let fitness = fitness_values[i];
            if fitness < particle.best_fitness {
                particle.best_fitness = fitness;
                particle.best_position = particle.position.clone();
            }
            if fitness < best_known_fitness {
                best_known_fitness = fitness;
                best_known_position = particle.position.clone();
            }
        }

        // Step C: Update velocities and positions in parallel.
        // To maintain determinism, we generate random numbers sequentially first.
        let r1_values: Vec<f64> = (0..n_particles * num_dimensions)
            .map(|_| rng.gen())
            .collect();
        let r2_values: Vec<f64> = (0..n_particles * num_dimensions)
            .map(|_| rng.gen())
            .collect();

        swarm
            .par_iter_mut()
            .enumerate()
            .for_each(|(p_idx, particle)| {
                for i in 0..num_dimensions {
                    let r1 = r1_values[p_idx * num_dimensions + i];
                    let r2 = r2_values[p_idx * num_dimensions + i];
                    let cognitive = params.cognitive_coeff
                        * r1
                        * (particle.best_position[i] - particle.position[i]);
                    let social =
                        params.social_coeff * r2 * (best_known_position[i] - particle.position[i]);
                    particle.velocity[i] =
                        (params.inertia * particle.velocity[i]) + cognitive + social;
                    particle.position[i] += particle.velocity[i];

                    if particle.position[i] < vars[i].0 {
                        particle.position[i] = vars[i].0;
                        particle.velocity[i] = 0.0;
                    } else if particle.position[i] > vars[i].1 {
                        particle.position[i] = vars[i].1;
                        particle.velocity[i] = 0.0;
                    }
                }
            });
    }

    let (best_objectives, best_constraints) = func(&best_known_position);
    Ok(OptimiserResult::new(
        vec![Solution::new(
            best_known_position,
            best_objectives,
            best_constraints,
        )],
        max_iter, // Returning max_iter as n_iterations
    ))
}
