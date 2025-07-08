use std::time::Instant;

use swarm::{
    error::{Result, SwarmError},
    pso::PsoParams,
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

    // Slow down for parallel testing
    // std::thread::sleep(std::time::Duration::from_millis(1));

    (vec![objective], None)
}

fn main() -> Result<()> {
    let now = Instant::now();

    let vars = vec![Variable(-5.0, 5.0), Variable(-5.0, 5.0)];
    let max_iter = 200;

    // let optimizer = Optimiser::Nsga {
    //     pop_size: 50,
    //     crossover_params: SbxParams::default(),
    //     mutation_params: PolyMutationParams::default(),
    //     seed: Some(1),
    // };

    let optimizer = Optimiser::ParticleSwarm {
        n_particles: 50,
        params: PsoParams::default(),
        constraint_handler: None,
        seed: Some(1),
    };

    let result = optimizer.solve(&mut himmelblau_problem, &vars, max_iter)?;
    let best_solution = result
        .best_solution()
        .ok_or(SwarmError::FailedToMeetCondition(
            "No solutions found".to_string(),
        ))?;

    let (actual_x, actual_y) = (best_solution.x[0], best_solution.x[1]);
    let actual_f = best_solution.f[0];

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

    println!("Solution: ({:.6}, {:.6})", actual_x, actual_y);
    println!("Objective: {:.6e}", actual_f);

    println!("Iterations: {}", result.n_iterations);

    let elapsed = now.elapsed();
    println!("Execution time: {} ms", elapsed.as_millis());

    Ok(())
}
