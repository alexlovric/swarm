use std::fs::File;
use std::io::{Result as IoResult, Write};
use swarm::{
    error::Result,
    nsga::{PolyMutationParams, SbxParams},
    Optimiser, Solution, Variable,
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

/// Writes the solutions from a Pareto front to a CSV file.
fn write_pareto_to_csv(solutions: &[Solution], filename: &str) -> IoResult<()> {
    let mut file = File::create(filename)?;

    // Write the header
    writeln!(file, "f1,f2,x1,x2")?;

    // Write the data for each solution
    for sol in solutions {
        writeln!(file, "{},{},{},{}", sol.f[0], sol.f[1], sol.x[0], sol.x[1])?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let vars = vec![Variable(0.0, 5.0), Variable(0.0, 3.0)];
    let max_iter = 250;

    let optimizer = Optimiser::Nsga {
        pop_size: 50,
        crossover_params: SbxParams::default(),
        mutation_params: PolyMutationParams::default(),
        seed: Some(1),
    };

    let result = optimizer.solve(&mut binh_and_korn_problem, &vars, max_iter)?;

    // Write results to CSV for plotting
    match write_pareto_to_csv(&result.solutions, "target/binh_and_korn_pareto.csv") {
        Ok(_) => println!("Successfully wrote Pareto front to binh_and_korn_pareto.csv"),
        Err(e) => eprintln!("Failed to write CSV file: {}", e),
    }
    Ok(())
}
