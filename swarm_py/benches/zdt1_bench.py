import swarm
import numpy as np

# --- Pymoo imports ---
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem


# --- Common Problem Definition (for Swarm) ---
def zdt1_problem(x):
    """
    ZDT1 benchmark problem for the `swarm` optimiser.
    """
    n_vars = len(x)
    f1 = x[0]
    g = 1.0 + 9.0 * sum(x[1:]) / (n_vars - 1)
    # Add a small epsilon to prevent division by zero if g is somehow zero
    h = 1.0 - np.sqrt(f1 / (g + 1e-8))
    f2 = g * h
    return ([f1, f2], None)


# --- Common Problem Definition (for Pymoo) ---
class ZDT1(Problem):
    """
    ZDT1 benchmark problem for the `pymoo` optimiser.
    """

    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        # Vectorized implementation for Pymoo
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
        f2 = g * (1.0 - np.sqrt(f1 / (g + 1e-8)))
        out["F"] = np.column_stack([f1, f2])


# --- Common Parameters for a fair test ---
# Note: Iterations are kept lower for faster benchmark runs.
N_VARS = 30
POP_SIZE = 100
MAX_ITER = 100
SEED = 1
CROSSOVER_PROB = 0.9
CROSSOVER_ETA = 20.0
MUTATION_PROB = 1.0 / N_VARS
MUTATION_ETA = 20.0


# --- Benchmark for Swarm ---
def test_swarm_zdt1(benchmark):
    """
    Benchmarks the swarm library on the ZDT1 problem.
    """
    variables = [swarm.Variable(0, 1) for _ in range(N_VARS)]

    optimiser = swarm.Optimiser.nsga(
        pop_size=POP_SIZE,
        crossover=swarm.SbxParams(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
        mutation=swarm.PmParams(prob=MUTATION_PROB, eta=MUTATION_ETA),
        seed=SEED,
    )

    # The function to benchmark, wrapped in a lambda to pass to the benchmark fixture.
    benchmark(lambda: optimiser.solve(zdt1_problem, variables, MAX_ITER))


# --- Benchmark for Pymoo ---
def test_pymoo_zdt1(benchmark):
    """
    Benchmarks the pymoo library on the ZDT1 problem.
    """
    problem = ZDT1(n_var=N_VARS)

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
        mutation=PM(prob=MUTATION_PROB, eta=MUTATION_ETA),
        eliminate_duplicates=True,
    )

    # The function to benchmark, wrapped in a lambda.
    benchmark(
        lambda: minimize(
            problem, algorithm, ("n_gen", MAX_ITER), seed=SEED, verbose=False
        )
    )
