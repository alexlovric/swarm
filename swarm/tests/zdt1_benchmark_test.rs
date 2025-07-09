use swarm::{
    error::Result,
    nsga::{PmParams, SbxParams},
    Optimiser, OptimiserResult, Variable,
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

/// Checks the solutions returned by the optimiser for the ZDT1 problem.
fn check_zdt1_solution(result: Result<OptimiserResult>) {
    assert!(result.is_ok(), "Optimiser returned an error");
    let solutions = &result.unwrap().solutions;

    assert!(
        !solutions.is_empty(),
        "The Pareto front should not be empty."
    );

    for sol in solutions {
        assert_eq!(
            sol.x.len(),
            30,
            "Solution has incorrect number of variables."
        );
        assert_eq!(
            sol.f.len(),
            2,
            "Solution has incorrect number of objectives."
        );

        let sum_of_others: f64 = sol.x.iter().skip(1).sum();
        let g_val = 1.0 + 9.0 * sum_of_others / (30 as f64 - 1.0);

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
    let vars = vec![Variable(0.0, 1.0); 30];
    let max_iter = 250;

    let optimiser = Optimiser::Nsga {
        pop_size: 100,
        crossover: SbxParams::new(0.9, 20.0),
        mutation: PmParams::new(1.0 / vars.len() as f64, 20.0),
        seed: Some(1),
    };

    check_zdt1_solution(optimiser.solve(&mut zdt1_problem, &vars, max_iter));
}
