use swarm::{
    nsga::{PolyMutationParams, SbxParams},
    Optimiser, Variable,
};

/// Defines the ZDT1 benchmark problem, a standard for multi-objective optimisation.
///
/// The goal is to minimize two objective functions, f1(x) and f2(x).
///
/// - f1(x) = x1
/// - f2(x) = g(x) * (1 - sqrt(x1 / g(x)))
///   where g(x) = 1 + 9 * (sum(xi for i=2 to n)) / (n - 1)
///
/// The variables x are in the range [0.0, 1.0].
///
/// The optimal solutions form a "Pareto Front" where g(x) = 1. This occurs when
/// x2, x3, ..., xn are all 0. On this front, the objectives satisfy the relationship:
/// f2 = 1 - sqrt(f1).
fn zdt1_problem(x: &[f64]) -> (Vec<f64>, Option<Vec<f64>>) {
    let n_vars = x.len();
    if n_vars == 0 {
        return (vec![], None);
    }

    let f1 = x[0];

    let sum_of_others: f64 = x.iter().skip(1).sum();
    let g = 1.0 + 9.0 * sum_of_others / (n_vars as f64 - 1.0);

    let h = 1.0 - (x[0] / g).sqrt();
    let f2 = g * h;

    (vec![f1, f2], None)
}

fn run_zdt1_test(optimiser: &Optimiser) {
    const N_VARS: usize = 30;
    let vars = vec![Variable(0.0, 1.0); N_VARS];
    let mut func = |x: &[f64]| zdt1_problem(x);
    let max_iter = 250; // Use enough iterations for good convergence

    let result = optimiser.solve(&mut func, &vars, max_iter);
    assert!(result.is_ok(), "Optimizer returned an error");
    let solutions = &result.unwrap().solutions;

    assert!(
        !solutions.is_empty(),
        "The Pareto front should not be empty."
    );

    for sol in solutions {
        assert_eq!(
            sol.x.len(),
            N_VARS,
            "Solution has incorrect number of variables."
        );
        assert_eq!(
            sol.f.len(),
            2,
            "Solution has incorrect number of objectives."
        );

        let sum_of_others: f64 = sol.x.iter().skip(1).sum();
        let g_val = 1.0 + 9.0 * sum_of_others / (N_VARS as f64 - 1.0);

        assert!(
            (g_val - 1.0).abs() < 0.1,
            "A solution is not on the Pareto front: g(x) was {}. Solution: {:?}",
            g_val,
            sol
        );
    }

    let f1_values: Vec<f64> = solutions.iter().map(|s| s.f[0]).collect();
    let min_f1 = f1_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_f1 = f1_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let spread = max_f1 - min_f1;
    assert!(
        spread > 0.5,
        "Poor diversity. The spread of f1 values is only {:.3}, expected > 0.5. Min f1: {}, Max f1: {}",
        spread,
        min_f1,
        max_f1
    );
}

#[test]
fn test_nsga_on_zdt1() {
    let optimiser = Optimiser::Nsga {
        pop_size: 100, // Must be even
        crossover_params: SbxParams::default(),
        mutation_params: PolyMutationParams::default(),
        seed: Some(1),
    };
    run_zdt1_test(&optimiser);
}
