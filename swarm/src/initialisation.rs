use crate::Variable;
use rand::{rngs::StdRng, seq::SliceRandom, Rng};

/// Defines the strategy for generating the initial population of solutions.
#[derive(Debug, Clone)]
pub enum Initialisation {
    /// Generates points using Latin Hypercube Sampling for a more even spread.
    ///
    /// - `centred`: If true, samples are placed in the middle of each interval.
    ///   If false, they are placed randomly within each interval.
    LatinHypercube { centred: bool },
    /// Generates points purely at random within the variable bounds.
    Random,
}

impl Initialisation {
    /// Generates a set of starting positions for the particles.
    ///
    /// # Arguments
    ///
    /// * `n_points` - The number of sample points to generate (e.g., the population size).
    /// * `variables` - A slice defining the bounds for each dimension of the search space.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Returns
    ///
    /// A `Vec<Vec<f64>>` where each inner vector is a single sample point.
    pub fn generate_samples(
        &self,
        n_points: usize,
        variables: &[Variable],
        rng: &mut StdRng,
    ) -> Vec<Vec<f64>> {
        match self {
            Initialisation::LatinHypercube { centred } => {
                latin_hypercube_sample(n_points, variables, *centred, rng)
            }
            Initialisation::Random => (0..n_points)
                .map(|_| {
                    variables
                        .iter()
                        .map(|var| rng.gen_range(var.0..var.1))
                        .collect()
                })
                .collect(),
        }
    }
}

/// Generates a sample of points using the Latin Hypercube Sampling (LHS) method.
///
/// LHS is a stratified sampling technique that ensures that the set of sample points is
/// well-distributed across the parameter space by dividing each dimension into `n_points`
/// equally sized intervals and placing exactly one sample in each interval.
///
/// # Arguments
///
/// * `n_points` - The number of sample points to generate.
/// * `variables` - A slice defining the bounds for each dimension.
/// * `centred` - If true, samples are placed in the middle of each interval. If false,
///   they are placed randomly within each interval.
/// * `rng` - A mutable reference to a random number generator.
///
/// # Returns
///
/// A `Vec<Vec<f64>>` where each inner vector is a single sample point.
pub fn latin_hypercube_sample(
    n_points: usize,
    variables: &[Variable],
    centred: bool,
    rng: &mut StdRng,
) -> Vec<Vec<f64>> {
    if n_points == 0 {
        return Vec::new();
    }
    let interval_size = 1.0 / (n_points as f64);
    let mut sample: Vec<Vec<f64>> = Vec::new();

    for var in variables {
        let (lower, upper) = (var.0, var.1);
        let scale = upper - lower;
        let mut vec: Vec<f64> = (0..n_points)
            .map(|i| i as f64 * interval_size)
            .map(|xx| {
                let shift = if centred {
                    0.5 * interval_size
                } else {
                    rng.gen::<f64>() * interval_size
                };
                xx + shift
            })
            .map(|xx| xx * scale + lower)
            .collect();
        vec.shuffle(rng);
        sample.push(vec);
    }

    // Transpose the matrix to get the final samples
    (0..n_points)
        .map(|i| (0..variables.len()).map(|j| sample[j][i]).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_random_initialisation() {
        let vars = vec![Variable(-5.0, 5.0), Variable(0.0, 10.0)];
        let n_points = 10;
        let mut rng = StdRng::seed_from_u64(42);
        let init = Initialisation::Random;

        let samples = init.generate_samples(n_points, &vars, &mut rng);

        // Check dimensions
        assert_eq!(samples.len(), n_points);
        assert_eq!(samples[0].len(), vars.len());

        // Check bounds
        for sample in samples {
            assert!(sample[0] >= -5.0 && sample[0] <= 5.0);
            assert!(sample[1] >= 0.0 && sample[1] <= 10.0);
        }
    }

    #[test]
    fn test_lhs_properties() {
        let vars = vec![Variable(0.0, 1.0), Variable(100.0, 200.0)];
        let n_points = 10;
        let mut rng = StdRng::seed_from_u64(42);
        let init = Initialisation::LatinHypercube { centred: false };

        let samples = init.generate_samples(n_points, &vars, &mut rng);

        // Check dimensions and bounds
        assert_eq!(samples.len(), n_points);
        for sample in &samples {
            assert_eq!(sample.len(), vars.len());
            assert!(sample[0] >= 0.0 && sample[0] <= 1.0);
            assert!(sample[1] >= 100.0 && sample[1] <= 200.0);
        }

        // Check the core LHS property: one sample per interval per dimension
        for dim in 0..vars.len() {
            let mut interval_counts = vec![0; n_points];
            let (min, max) = (vars[dim].0, vars[dim].1);
            let interval_size = (max - min) / n_points as f64;

            for sample in &samples {
                let value = sample[dim];
                let interval_index = ((value - min) / interval_size).floor() as usize;
                // Handle edge case where value is exactly the max bound
                let clamped_index = interval_index.min(n_points - 1);
                interval_counts[clamped_index] += 1;
            }
            assert!(
                interval_counts.iter().all(|&count| count == 1),
                "LHS property failed for dimension {}",
                dim
            );
        }
    }

    #[test]
    fn test_lhs_centred() {
        let vars = vec![Variable(0.0, 10.0)];
        let n_points = 10;
        let mut rng = StdRng::seed_from_u64(42);
        let init = Initialisation::LatinHypercube { centred: true };

        let samples = init.generate_samples(n_points, &vars, &mut rng);

        // With a fixed seed, the shuffle is deterministic. We can find the point
        // that should have been in the first interval [0, 1] and check its value.
        // The centre of the first interval is 0.5.
        let point_in_first_interval = samples.iter().find(|s| s[0] < 1.0).unwrap();
        assert!((point_in_first_interval[0] - 0.5).abs() < 1e-9);

        // The centre of the last interval [9, 10] is 9.5.
        let point_in_last_interval = samples.iter().find(|s| s[0] > 9.0).unwrap();
        assert!((point_in_last_interval[0] - 9.5).abs() < 1e-9);
    }
}
