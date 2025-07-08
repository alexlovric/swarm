use swarm::{
    nsga::{PolyMutationParams, SbxParams},
    particle_swarm::PsoParams,
    Optimiser, Variable,
};

/// Himmelblau's function, a standard 2D benchmark for optimisation.
///
/// f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
/// It has four global minima, all with a value of 0.
fn himmelblau_problem(x: &[f64]) -> (Vec<f64>, Option<Vec<f64>>) {
    if x.len() != 2 {
        return (vec![f64::MAX], None);
    }
    let x1 = x[0];
    let x2 = x[1];

    let term1 = (x1.powi(2) + x2 - 11.0).powi(2);
    let term2 = (x1 + x2.powi(2) - 7.0).powi(2);
    let objective = term1 + term2;

    (vec![objective], None)
}

fn run_himmelblau_test(optimiser: &Optimiser) {
    let vars = vec![Variable(-5.0, 5.0), Variable(-5.0, 5.0)];
    let mut func = |x: &[f64]| himmelblau_problem(x);
    let max_iter = 100;

    let result = optimiser.solve(&mut func, &vars, max_iter);
    assert!(result.is_ok(), "Optimizer returned an error");
    let optimiser_result = &result.unwrap().solutions[0];

    let (actual_x, actual_y) = (optimiser_result.x[0], optimiser_result.x[1]);
    let actual_f = optimiser_result.f[0];

    let minima = [
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ];
    let tolerance = 1e-3;
    let is_close = minima
        .iter()
        .any(|(mx, my)| ((actual_x - mx).powi(2) + (actual_y - my).powi(2)).sqrt() < tolerance);

    assert!(
        is_close,
        "Solution ({}, {}) is not close to any known minimum.",
        actual_x, actual_y
    );
    assert!(
        actual_f.abs() < tolerance,
        "f(x, y) = {} is not close to 0.",
        actual_f
    );
}

#[test]
fn test_nsga2_on_himmelblau() {
    let optimizer = Optimiser::Nsga {
        pop_size: 50,
        crossover_params: SbxParams::default(),
        mutation_params: PolyMutationParams::default(),
        seed: Some(1),
    };
    run_himmelblau_test(&optimizer);
}

#[test]
fn test_particleswarm_on_himmelblau() {
    let optimiser = Optimiser::ParticleSwarm {
        n_particles: 50,
        params: PsoParams::default(),
        constraint_handler: None,
        seed: Some(1),
    };
    run_himmelblau_test(&optimiser);
}
