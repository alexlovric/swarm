use std::time::Instant;

use swarm::{error::Result, pso::PsoParams, Optimiser, Variable};

/// Himmelblau's function, a standard 2D benchmark for optimisation.
///
/// f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
/// It has four global minima, all with a value of 0.
fn himmelblau_problem(x: &[f64]) -> (Vec<f64>, Option<Vec<f64>>) {
    let f = (x[0].powi(2) + x[1] - 11.0).powi(2) + (x[0] + x[1].powi(2) - 7.0).powi(2);

    // Slow down for parallel testing
    // std::thread::sleep(std::time::Duration::from_millis(1));

    (vec![f], None)
}

fn check_solution(actual_x: f64, actual_y: f64, actual_f: f64) {
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

fn main() -> Result<()> {
    let now = Instant::now();

    let vars = vec![Variable(-5.0, 5.0), Variable(-5.0, 5.0)];
    let max_iter = 200;

    // let optimiser = Optimiser::Nsga {
    //     pop_size: 50,
    //     crossover: SbxParams::default(),
    //     mutation: PmParams::default(),
    //     seed: Some(1),
    // };

    let optimiser = Optimiser::Pso {
        n_particles: 50,
        params: PsoParams::default(),
        constraint_handler: None,
        seed: None,
    };

    // Serial solving
    let result = optimiser.solve(&mut himmelblau_problem, &vars, max_iter)?;

    let (actual_x, actual_y) = (result.solutions[0].x[0], result.solutions[0].x[1]);
    let actual_f = result.solutions[0].f[0];

    check_solution(actual_x, actual_y, actual_f);

    println!("Serial:");
    println!("Solution: ({:.6}, {:.6})", actual_x, actual_y);
    println!("Objective: {:.6e}", actual_f);

    // Parallel solving (only makes sense if blackbox function is computationally expensive)
    #[cfg(feature = "parallel")]
    {
        let result = optimiser.solve_par(&himmelblau_problem, &vars, max_iter)?;

        let (actual_x, actual_y) = (result.solutions[0].x[0], result.solutions[0].x[1]);
        let actual_f = result.solutions[0].f[0];

        check_solution(actual_x, actual_y, actual_f);

        println!("\nParallel:");
        println!("Solution: ({:.6}, {:.6})", actual_x, actual_y);
        println!("Objective: {:.6e}", actual_f);
    }

    let elapsed = now.elapsed();
    println!("Execution time: {} ms", elapsed.as_millis());

    Ok(())
}
