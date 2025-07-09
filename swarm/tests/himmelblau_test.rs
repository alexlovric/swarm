use swarm::{
    error::Result,
    nsga::{PmParams, SbxParams},
    pso::PsoParams,
    Optimiser, OptimiserResult, Variable,
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

fn check_himmelblau_solution(result: Result<OptimiserResult>) {
    assert!(result.is_ok(), "Optimiser returned an error");
    let result = result.unwrap();

    let minima = [
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ];
    let tolerance = 1e-3;

    let actual_x = result.solutions[0].x[0];
    let actual_y = result.solutions[0].x[1];
    let actual_f = result.solutions[0].f[0];

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
fn test_nsga_on_himmelblau() {
    let optimiser = Optimiser::Nsga {
        pop_size: 50,
        crossover: SbxParams::default(),
        mutation: PmParams::default(),
        seed: Some(1),
    };

    let vars = vec![Variable(-5.0, 5.0), Variable(-5.0, 5.0)];
    let max_iter = 100;

    let result = optimiser.solve(&mut himmelblau_problem, &vars, max_iter);
    check_himmelblau_solution(result);

    #[cfg(feature = "parallel")]
    {
        let result = optimiser.solve_par(&mut himmelblau_problem, &vars, max_iter);
        check_himmelblau_solution(result);
    }
}

#[test]
fn test_pso_on_himmelblau() {
    let optimiser = Optimiser::Pso {
        n_particles: 50,
        params: PsoParams::default(),
        constraint_handler: None,
        seed: Some(1),
    };

    let vars = vec![Variable(-5.0, 5.0), Variable(-5.0, 5.0)];
    let max_iter = 100;

    let result = optimiser.solve(&mut himmelblau_problem, &vars, max_iter);
    check_himmelblau_solution(result);

    #[cfg(feature = "parallel")]
    {
        let result = optimiser.solve_par(&mut himmelblau_problem, &vars, max_iter);
        check_himmelblau_solution(result);
    }
}
