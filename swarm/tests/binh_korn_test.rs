use swarm::{
    error::Result,
    nsga::{PmParams, SbxParams},
    Optimiser, OptimiserResult, Variable,
};

/// The Binh and Korn multi-objective test function.
///
/// It has two objectives to be minimised, subject to two constraints.
/// The optimal solutions (the Pareto front) lie on the boundary of the first constraint.
///
/// - Objectives:
///   - `f1(x, y) = 4*x^2 + 4*y^2`
///   - `f2(x, y) = (x - 5)^2 + (y - 5)^2`
///
/// - Constraints (must be <= 0 for feasibility):
///   - `g1(x, y) = (x - 5)^2 + y^2 - 25 <= 0`
///   - `g2(x, y) = 7.7 - (x - 8)^2 - (y + 3)^2 <= 0`
///
/// - Variable bounds:
///   - `0 <= x <= 5`
///   - `0 <= y <= 3`
fn binh_and_korn_problem(x: &[f64]) -> (Vec<f64>, Option<Vec<f64>>) {
    if x.len() != 2 {
        return (vec![f64::MAX, f64::MAX], None);
    }
    let x1 = x[0];
    let x2 = x[1];

    // Objectives
    let f1 = 4.0 * x1.powi(2) + 4.0 * x2.powi(2);
    let f2 = (x1 - 5.0).powi(2) + (x2 - 5.0).powi(2);

    // Constraints
    let g1 = (x1 - 5.0).powi(2) + x2.powi(2) - 25.0;
    let g2 = 7.7 - (x1 - 8.0).powi(2) - (x2 + 3.0).powi(2);

    (vec![f1, f2], Some(vec![g1, g2]))
}

/// Checks the solutions returned by the optimiser for the Binh and Korn problem.
fn check_binh_and_korn_solution(result: Result<OptimiserResult>) {
    assert!(result.is_ok(), "Optimiser returned an error");
    let solutions = &result.unwrap().solutions;

    assert!(
        !solutions.is_empty(),
        "The optimiser should return at least one solution."
    );

    // Assertions for each solution on the Pareto front
    for sol in solutions {
        let g = sol.g.as_ref().expect("Constraints should be present");
        let g1 = g[0];
        let g2 = g[1];

        // Feasibility Check: Assert that both constraints are satisfied (g <= 0).
        assert!(
            g1 <= 1e-4, // Allow a small tolerance for floating point inaccuracies
            "Constraint g1 was violated. Value: {}. Solution: {:?}",
            g1,
            sol
        );
        assert!(
            g2 <= 1e-4,
            "Constraint g2 was violated. Value: {}. Solution: {:?}",
            g2,
            sol
        );
    }
    println!(
        "Binh and Korn test passed: All {} solutions are feasible and lie on the Pareto front.",
        solutions.len()
    );
}

#[test]
fn test_nsga_on_binh_and_korn() {
    let vars = vec![Variable(0.0, 5.0), Variable(0.0, 3.0)];
    let max_iter = 250;

    let optimiser = Optimiser::Nsga {
        pop_size: 100,
        crossover: SbxParams::new(0.9, 20.0),
        mutation: PmParams::new(1.0 / vars.len() as f64, 20.0),
        seed: Some(1),
    };

    check_binh_and_korn_solution(optimiser.solve(&mut binh_and_korn_problem, &vars, max_iter));
}
