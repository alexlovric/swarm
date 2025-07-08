use swarm::{
    nsga::{PolyMutationParams, SbxParams},
    particle_swarm::PsoParams,
    ConstraintHandler, Optimiser, Variable,
};

/// A simple constrained quadratic problem.
///
/// The objective is to minimize `f(x) = (x - 2)^2`.
/// The unconstrained minimum is at `x = 2`.
///
/// However, there is a constraint `x >= 5`.
/// This is expressed as `g(x) = 5 - x <= 0`.
/// A positive `g(x)` value indicates a violation.
///
/// The true constrained minimum is at the boundary `x = 5`, where `f(x) = 9`.
fn constrained_quadratic_problem(x: &[f64]) -> (Vec<f64>, Option<Vec<f64>>) {
    if x.len() != 1 {
        return (vec![f64::MAX], None);
    }
    let objective = (x[0] - 2.0).powi(2);
    let constraint = 5.0 - x[0]; // Constraint: g(x) = 5 - x <= 0

    (vec![objective], Some(vec![constraint]))
}

/// Contains the core logic for testing PSO on the constrained problem.
fn run_constrained_quadratic_test(optimiser: &Optimiser) {
    // The search space is wide, but the true optimum is outside the feasible region.
    let vars = vec![Variable(0.0, 10.0)];
    let mut func = |x: &[f64]| constrained_quadratic_problem(x);
    let max_iter = 100;

    let result = optimiser.solve(&mut func, &vars, max_iter);
    assert!(result.is_ok(), "Optimizer returned an error");
    let optimiser_result = &result.unwrap().solutions[0];

    let actual_x = optimiser_result.x[0];
    let actual_f = optimiser_result.f[0];
    let actual_g = optimiser_result.g.as_ref().unwrap()[0];

    let optimal_x = 5.0;
    let optimal_f = 9.0;
    let tolerance = 1e-2;

    // Assert that the found solution is very close to the known constrained optimum.
    assert!(
        (actual_x - optimal_x).abs() < tolerance,
        "Solution x = {} is not close enough to the known optimum of {}",
        actual_x,
        optimal_x
    );

    // Assert that the objective value is correct at the optimum.
    assert!(
        (actual_f - optimal_f).abs() < tolerance,
        "Solution f(x) = {} is not close enough to the optimal value of {}",
        actual_f,
        optimal_f
    );

    // Assert that the final solution is feasible (constraint value is <= 0).
    assert!(
        actual_g.abs() < tolerance,
        "Solution is not feasible. Constraint value g(x) was {}",
        actual_g
    );
}

#[test]
fn test_particleswarm_on_constrained_quadratic_problem() {
    let optimiser = Optimiser::ParticleSwarm {
        n_particles: 50,
        params: PsoParams::default(),
        constraint_handler: Some(ConstraintHandler::Penalty {
            multiplier: 10000.0,
        }),
        seed: Some(1),
    };
    run_constrained_quadratic_test(&optimiser);
}

#[test]
fn test_nsga_on_constrained_quadratic_problem() {
    let optimiser = Optimiser::Nsga {
        pop_size: 50,
        crossover_params: SbxParams::default(),
        mutation_params: PolyMutationParams::default(),
        seed: Some(1),
    };
    run_constrained_quadratic_test(&optimiser);
}
