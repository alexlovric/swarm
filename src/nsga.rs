use std::cmp::Ordering;

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    error::{Result, SwarmError},
    initialisation::Initialisation,
    ConstraintHandler, OptimiserResult, Solution, Variable,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Debug)]
pub struct SbxParams {
    // Simulated Binary Crossover
    pub prob: f64,
    pub eta: f64,
}
impl Default for SbxParams {
    fn default() -> Self {
        Self {
            prob: 0.9199,
            eta: 0.1057,
        }
    }
}

#[derive(Debug)]
pub struct PolyMutationParams {
    // Polynomial Mutation
    pub prob: f64,
    pub eta: f64,
}
impl Default for PolyMutationParams {
    fn default() -> Self {
        Self {
            prob: 1.0,
            eta: 20.0,
        }
    }
}

#[derive(Clone, Debug)]
struct Individual {
    x: Vec<f64>,
    f: Vec<f64>,
    g: Option<Vec<f64>>,
    // The total constraint violation, calculated once per evaluation.
    // A value of 0.0 means the individual is feasible.
    constraint_violation: f64,
    // Transient data for survival selection
    rank: usize,
    crowding_distance: f64,
}

impl Individual {
    fn new(x: Vec<f64>) -> Self {
        Self {
            x,
            f: vec![],
            g: None,
            constraint_violation: 0.0,
            rank: 0,
            crowding_distance: 0.0,
        }
    }

    /// Evaluates the individual using the objective function and calculates its fitness.
    /// Evaluates the individual (SERIAL VERSION).
    #[cfg(not(feature = "parallel"))]
    fn evaluate<F: FnMut(&[f64]) -> (Vec<f64>, Option<Vec<f64>>)>(
        &mut self,
        func: &mut F,
        _constraint_handler: &Option<ConstraintHandler>,
    ) {
        let (f, g) = func(&self.x);
        self.f = f;
        self.g = g;
        self.constraint_violation = match &self.g {
            Some(constraints) => constraints.iter().filter(|&&c| c > 0.0).sum(),
            None => 0.0,
        };
    }

    /// Evaluates the individual (PARALLEL VERSION).
    #[cfg(feature = "parallel")]
    fn evaluate<F: Fn(&[f64]) -> (Vec<f64>, Option<Vec<f64>>) + Sync + Send>(
        &mut self,
        func: &F,
        _constraint_handler: &Option<ConstraintHandler>,
    ) {
        let (f, g) = func(&self.x);
        self.f = f;
        self.g = g;
        self.constraint_violation = match &self.g {
            Some(constraints) => constraints.iter().filter(|&&c| c > 0.0).sum(),
            None => 0.0,
        };
    }
}

/// The main NSGA-II function (SERIAL VERSION).
/// This version is compiled when the "parallel" feature is NOT enabled.
#[cfg(not(feature = "parallel"))]
pub fn nsga<F>(
    func: &mut F,
    vars: &[Variable],
    max_iter: usize,
    pop_size: usize,
    crossover_params: &SbxParams,
    mutation_params: &PolyMutationParams,
    initialisation: Initialisation,
    constraint_handler: Option<ConstraintHandler>,
    seed: Option<u64>,
) -> Result<OptimiserResult>
where
    F: FnMut(&[f64]) -> (Vec<f64>, Option<Vec<f64>>),
{
    // Checks
    if pop_size < 4 || pop_size % 2 != 0 {
        return Err(SwarmError::FailedToMeetCondition(
            "Population size must be an even number >= 4.".to_string(),
        ));
    }

    let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);

    // Initialisation
    let mut population: Vec<Individual> = initialisation
        .generate_samples(pop_size, vars, &mut rng)
        .into_iter()
        .map(Individual::new)
        .collect();

    // Evaluate the initial population
    for ind in &mut population {
        ind.evaluate(func, &constraint_handler);
    }

    // Assign rank and crowding to the initial population for the first selection
    survival_selection(&mut population, pop_size);

    // Optimisation Loop
    for _ in 0..max_iter {
        // Create Offspring
        let n_offsprings = pop_size / 2;
        let parents = selection(&population, n_offsprings, &mut rng);
        let mut offspring = crossover(&parents, vars, crossover_params, &mut rng);
        mutate(&mut offspring, vars, mutation_params, &mut rng);

        // Evaluate Offspring
        for ind in &mut offspring {
            ind.evaluate(func, &constraint_handler);
        }

        // Survival of the Fittest
        // Combine parent and offspring populations
        population.append(&mut offspring);

        // Select the next generation
        survival_selection(&mut population, pop_size);
    }

    // Final Result
    // The final population contains the best solutions. The first front is the Pareto front.
    let solutions = population
        .into_iter()
        .filter(|ind| ind.rank == 0) // Filter for the first non-dominated front
        .map(|ind| Solution {
            x: ind.x,
            f: ind.f,
            g: ind.g,
        })
        .collect();

    Ok(OptimiserResult::new(solutions, max_iter))
}

/// The main NSGA-II function (PARALLEL VERSION).
/// This version is compiled only when the "parallel" feature is enabled.
#[cfg(feature = "parallel")]
pub fn nsga<F>(
    func: &F, // Takes an immutable, thread-safe closure
    vars: &[Variable],
    max_iter: usize,
    pop_size: usize,
    crossover_params: &SbxParams,
    mutation_params: &PolyMutationParams,
    initialisation: Initialisation,
    constraint_handler: Option<ConstraintHandler>,
    seed: Option<u64>,
) -> Result<OptimiserResult>
where
    F: Fn(&[f64]) -> (Vec<f64>, Option<Vec<f64>>) + Sync + Send,
{
    if pop_size < 4 || pop_size % 2 != 0 {
        return Err(SwarmError::FailedToMeetCondition(
            "Population size must be an even number >= 4.".to_string(),
        ));
    }
    let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);

    // Initialization (Parallel)
    let mut population: Vec<Individual> = initialisation
        .generate_samples(pop_size, vars, &mut rng)
        .into_par_iter() // Parallel iterator
        .map(|x| {
            let mut ind = Individual::new(x);
            ind.evaluate(func, &constraint_handler);
            ind
        })
        .collect();

    survival_selection(&mut population, pop_size);

    // Optimization Loop
    for _ in 0..max_iter {
        let n_parent_pairs = pop_size / 2;
        let parents = selection(&population, n_parent_pairs, &mut rng);
        let mut offspring = crossover(&parents, vars, crossover_params, &mut rng);
        mutate(&mut offspring, vars, mutation_params, &mut rng);

        // Evaluate Offspring (Parallel)
        offspring.par_iter_mut().for_each(|ind| {
            ind.evaluate(func, &constraint_handler);
        });

        population.append(&mut offspring);
        survival_selection(&mut population, pop_size);
    }

    let solutions = population
        .into_iter()
        .filter(|ind| ind.rank == 0)
        .map(|ind| Solution {
            x: ind.x,
            f: ind.f,
            g: ind.g,
        })
        .collect();
    Ok(OptimiserResult::new(solutions, max_iter))
}

/// Performs survival selection on a combined population to select the next generation.
///
/// # Arguments
/// * `population` - The population to select from.
/// * `n_survive` - The number of individuals to select.
fn survival_selection(population: &mut Vec<Individual>, n_survive: usize) {
    // 1. Perform non-dominated sorting to get the fronts
    let fronts = non_dominated_sort(population);

    // 2. Assign rank and calculate crowding distance for each front
    for (rank_idx, front_indices) in fronts.iter().enumerate() {
        for &pop_idx in front_indices {
            population[pop_idx].rank = rank_idx;
        }
        // This is the corrected call, passing the whole population and the relevant indices
        crowding_distance_assignment(population, front_indices);
    }

    // 3. Sort the entire population based on rank (primary) and crowding distance (secondary)
    population.sort_by(|a, b| {
        a.rank.cmp(&b.rank).then_with(|| {
            b.crowding_distance
                .partial_cmp(&a.crowding_distance)
                .unwrap_or(Ordering::Equal)
        })
    });

    // 4. Truncate the population to the desired size
    population.truncate(n_survive);
}

/// Sorts the population into non-dominated fronts.
///
/// # Arguments
/// * `population` - The population to sort.
///
/// # Returns
/// A vector of vectors, where each inner vector contains the indices of the individuals in a front.
fn non_dominated_sort(population: &[Individual]) -> Vec<Vec<usize>> {
    let n = population.len();
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut dominated_by_count = vec![0; n];
    let mut dominates_list: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&population[i], &population[j]) {
                dominates_list[i].push(j);
                dominated_by_count[j] += 1;
            } else if dominates(&population[j], &population[i]) {
                dominates_list[j].push(i);
                dominated_by_count[i] += 1;
            }
        }
        if dominated_by_count[i] == 0 {
            if fronts.is_empty() {
                fronts.push(Vec::new());
            }
            fronts[0].push(i);
        }
    }

    let mut current_front_idx = 0;
    while current_front_idx < fronts.len() {
        let mut next_front = Vec::new();
        for &i in &fronts[current_front_idx] {
            for &j in &dominates_list[i] {
                dominated_by_count[j] -= 1;
                if dominated_by_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        if !next_front.is_empty() {
            fronts.push(next_front);
        }
        current_front_idx += 1;
    }

    fronts
}

/// Calculates and assigns crowding distance to all individuals within a single front.
///
/// # Arguments
/// * `population` - The population to calculate crowding distance for.
/// * `front_indices` - The indices of the individuals in the front.
fn crowding_distance_assignment(population: &mut [Individual], front_indices: &[usize]) {
    let n = front_indices.len();
    if n == 0 {
        return;
    }

    // Set initial crowding distance to 0 for all individuals in the front
    for &idx in front_indices {
        population[idx].crowding_distance = 0.0;
    }

    let n_obj = population[front_indices[0]].f.len();

    for m in 0..n_obj {
        // Create a mutable copy of the indices to sort them
        let mut sorted_indices = front_indices.to_vec();

        // Sort the indices based on the m-th objective value of the individuals they point to
        sorted_indices.sort_by(|&a, &b| {
            population[a].f[m]
                .partial_cmp(&population[b].f[m])
                .unwrap_or(Ordering::Equal)
        });

        // Assign infinite distance to the boundary points
        let first_idx = sorted_indices[0];
        let last_idx = sorted_indices[n - 1];
        population[first_idx].crowding_distance = f64::INFINITY;
        population[last_idx].crowding_distance = f64::INFINITY;

        let f_min = population[first_idx].f[m];
        let f_max = population[last_idx].f[m];
        let range = f_max - f_min;

        if range.abs() < 1e-9 {
            continue;
        } // Avoid division by zero

        // Add distance for intermediate points
        for i in 1..(n - 1) {
            let prev_idx = sorted_indices[i - 1];
            let next_idx = sorted_indices[i + 1];
            let current_idx = sorted_indices[i];

            let numerator = population[next_idx].f[m] - population[prev_idx].f[m];
            population[current_idx].crowding_distance += numerator / range;
        }
    }
}

/// Selects parents from the population using a binary tournament.
///
/// # Arguments
/// * `population` - The population to select from.
/// * `n_select` - The number of parents to select.
/// * `rng` - A mutable random number generator.
///
/// # Returns
/// A vector of pairs of selected parents.
fn selection(
    population: &[Individual],
    n_select: usize,
    rng: &mut StdRng,
) -> Vec<(Individual, Individual)> {
    let mut parents = Vec::with_capacity(n_select);
    for _ in 0..n_select {
        let p1 = tournament(population, rng);
        let p2 = tournament(population, rng);
        parents.push((p1.clone(), p2.clone()));
    }
    parents
}

/// Performs a single binary tournament selection.
///
/// # Arguments
/// * `population` - The population to select from.
/// * `rng` - A mutable random number generator.
///
/// # Returns
/// A reference to the selected individual.
fn tournament<'a>(population: &'a [Individual], rng: &mut StdRng) -> &'a Individual {
    let i1 = rng.gen_range(0..population.len());
    let i2 = rng.gen_range(0..population.len());
    let ind1 = &population[i1];
    let ind2 = &population[i2];

    if dominates(ind1, ind2) {
        ind1
    } else if dominates(ind2, ind1) {
        ind2
    } else if ind1.crowding_distance > ind2.crowding_distance {
        ind1
    } else {
        ind2
    }
}

/// The core dominance check, implementing the Constrained Dominance principle.
///
/// # Arguments
/// * `a` - The first individual to compare.
/// * `b` - The second individual to compare.
///
/// # Returns
/// A boolean indicating whether `a` dominates `b`.
fn dominates(a: &Individual, b: &Individual) -> bool {
    // Principle 1: If one is feasible and the other is not, the feasible one dominates.
    if a.constraint_violation == 0.0 && b.constraint_violation > 0.0 {
        return true;
    }
    if a.constraint_violation > 0.0 && b.constraint_violation == 0.0 {
        return false;
    }

    // Principle 2: If both are infeasible, the one with the smaller violation dominates.
    if a.constraint_violation > 0.0 && b.constraint_violation > 0.0 {
        return a.constraint_violation < b.constraint_violation;
    }

    // Principle 3: If both are feasible, use standard Pareto dominance.
    let a_is_better = a.f.iter().zip(&b.f).any(|(v_a, v_b)| v_a < v_b);
    let b_is_better = a.f.iter().zip(&b.f).any(|(v_a, v_b)| v_b < v_a);

    a_is_better && !b_is_better
}

/// Creates a new offspring population using Simulated Binary Crossover (SBX).
///
/// # Arguments
/// * `parents` - A slice of tuples, each containing two parent individuals.
/// * `vars` - A slice defining the bounds for each variable in the search space.
/// * `params` - Parameters controlling the crossover process.
/// * `rng` - A mutable random number generator.
///
/// # Returns
/// A vector of new offspring individuals.
fn crossover(
    parents: &[(Individual, Individual)],
    vars: &[Variable],
    params: &SbxParams,
    rng: &mut StdRng,
) -> Vec<Individual> {
    let mut offspring = Vec::with_capacity(parents.len() * 2);

    for (p1, p2) in parents {
        let mut c1 = Individual::new(p1.x.clone());
        let mut c2 = Individual::new(p2.x.clone());

        if rng.gen::<f64>() > params.prob {
            offspring.push(c1);
            offspring.push(c2);
            continue;
        }

        for (i, var) in vars.iter().enumerate() {
            let (x1, x2) = (p1.x[i], p2.x[i]);
            let (min, max) = (var.0, var.1);

            if (x1 - x2).abs() < 1e-9 {
                continue;
            }

            let (y1, y2) = if x1 < x2 { (x1, x2) } else { (x2, x1) };

            let rand = rng.gen::<f64>();
            let beta = if rand <= 0.5 {
                (2.0 * rand).powf(1.0 / (params.eta + 1.0))
            } else {
                (1.0 / (2.0 * (1.0 - rand))).powf(1.0 / (params.eta + 1.0))
            };

            let mut off1 = 0.5 * ((y1 + y2) - beta * (y2 - y1));
            let mut off2 = 0.5 * ((y1 + y2) + beta * (y2 - y1));

            off1 = off1.clamp(min, max);
            off2 = off2.clamp(min, max);

            if rng.gen::<f64>() < 0.5 {
                c1.x[i] = off1;
                c2.x[i] = off2;
            } else {
                c1.x[i] = off2;
                c2.x[i] = off1;
            }
        }
        offspring.push(c1);
        offspring.push(c2);
    }
    offspring
}

/// Modifies an offspring population using Polynomial Mutation.
///
/// # Arguments
/// * `offspring` - A mutable slice of `Individual` representing the offspring population.
/// * `vars` - A slice defining the bounds for each variable in the search space.
/// * `params` - Parameters controlling the mutation process.
/// * `rng` - A mutable random number generator.
fn mutate(
    offspring: &mut [Individual],
    vars: &[Variable],
    params: &PolyMutationParams,
    rng: &mut StdRng,
) {
    let prob_per_var = params.prob / vars.len() as f64;

    for ind in offspring {
        for (i, var) in vars.iter().enumerate() {
            if rng.gen::<f64>() > prob_per_var {
                continue;
            }

            let (x, min, max) = (ind.x[i], var.0, var.1);
            let delta1 = (x - min) / (max - min);
            let delta2 = (max - x) / (max - min);

            let rand = rng.gen::<f64>();
            let mut_pow = 1.0 / (params.eta + 1.0);

            let deltaq = if rand < 0.5 {
                (2.0 * rand + (1.0 - 2.0 * rand) * (1.0 - delta1).powf(params.eta + 1.0))
                    .powf(mut_pow)
                    - 1.0
            } else {
                1.0 - (2.0 * (1.0 - rand)
                    + 2.0 * (rand - 0.5) * (1.0 - delta2).powf(params.eta + 1.0))
                .powf(mut_pow)
            };

            ind.x[i] = (x + deltaq * (max - min)).clamp(min, max);
        }
    }
}
