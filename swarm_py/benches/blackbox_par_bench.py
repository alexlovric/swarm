import pytest
import swarm_py as swarm
import numpy as np
import time


# --- Computationally Expensive Problem Definition ---
def expensive_blackbox(x):
    time.sleep(0.0005)  # Artificial delay
    n_vars = len(x)
    f1 = x[0]
    g = 1.0 + 9.0 * sum(x[1:]) / (n_vars - 1)
    h = 1.0 - np.sqrt(f1 / (g + 1e-8))
    f2 = g * h
    return ([f1, f2], None)


# --- Common Parameters ---
N_VARS = 30
POP_SIZE = 100
MAX_ITER = 100
SEED = 1


# --- Benchmark Fixture for Swarm Optimiser ---
@pytest.fixture
def swarm_optimiser():
    """Provides a configured swarm optimiser instance."""
    return swarm.Optimiser.nsga(
        pop_size=POP_SIZE,
        crossover=swarm.SbxParams(prob=0.9, eta=20.0),
        mutation=swarm.PmParams(prob=1.0 / N_VARS, eta=20.0),
        seed=SEED,
    )


# --- Benchmark Tests ---
def test_swarm_serial(benchmark, swarm_optimiser):
    """Benchmarks the swarm serial solver."""
    variables = [swarm.Variable(0, 1) for _ in range(N_VARS)]
    benchmark(lambda: swarm_optimiser.solve(expensive_blackbox, variables, MAX_ITER))


def test_swarm_parallel(benchmark, swarm_optimiser):
    """Benchmarks the swarm parallel solver."""
    variables = [swarm.Variable(0, 1) for _ in range(N_VARS)]
    benchmark(
        lambda: swarm_optimiser.solve_par(expensive_blackbox, variables, MAX_ITER)
    )
