import swarm_py as swarm
import time
import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem


def zdt1_problem(x):
    """
    ZDT1 (Zitzler-Deb-Thiele's function 1) benchmark problem.
    This function is used by the `swarm` optimiser.
    """
    n_vars = len(x)
    f1 = x[0]

    # Calculate g(x)
    g = 1.0 + 9.0 * sum(x[1:]) / (n_vars - 1)

    # Calculate f2(x)
    # Add a small epsilon to prevent division by zero if g is somehow zero
    h = 1.0 - np.sqrt(f1 / (g + 1e-8))
    f2 = g * h

    return ([f1, f2], None)


def plot_comparison(swarm_solutions, pymoo_results, true_pf):
    """
    Plots the Pareto fronts from swarm, pymoo, and the true front.
    """
    # Extract objective values from swarm solutions
    swarm_f1 = [sol.f[0] for sol in swarm_solutions]
    swarm_f2 = [sol.f[1] for sol in swarm_solutions]

    # Extract objective values from pymoo results
    pymoo_f1 = pymoo_results[:, 0]
    pymoo_f2 = pymoo_results[:, 1]

    # Extract true Pareto front
    true_f1 = true_pf[:, 0]
    true_f2 = true_pf[:, 1]

    plt.figure(figsize=(10, 8))

    # Plot true Pareto Front
    plt.plot(
        true_f1,
        true_f2,
        color="black",
        linestyle="--",
        label="True Pareto Front",
        zorder=1,
    )

    # Plot swarm results
    plt.scatter(
        swarm_f1,
        swarm_f2,
        facecolor="none",
        edgecolor="red",
        s=80,
        label="Swarm",
        zorder=3,
    )

    # Plot pymoo results
    plt.scatter(
        pymoo_f1,
        pymoo_f2,
        facecolor="none",
        edgecolor="blue",
        s=80,
        label="Pymoo",
        zorder=2,
    )

    plt.title("ZDT1 Pareto Front Comparison: Swarm vs. Pymoo")
    plt.xlabel("Objective 1 (f1)")
    plt.ylabel("Objective 2 (f2)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Custom Pymoo Problem class to ensure identical logic
class ZDT1(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        # Vectorized implementation for Pymoo
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
        f2 = g * (1.0 - np.sqrt(f1 / (g + 1e-8)))
        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, n_pareto_points=100):
        # The true Pareto front for ZDT1 is f2 = 1 - sqrt(f1)
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T


if __name__ == "__main__":
    # --- Common optimisation parameters ---
    n_vars = 30
    pop_size = 100
    max_iter = 400  # Corresponds to n_gen in pymoo

    # --- NSGA-II operator parameters ---
    # Crossover (SBX)
    crossover_prob = 0.9
    crossover_eta = 20.0

    # Mutation (Polynomial)
    mutation_prob = 1.0 / n_vars
    mutation_eta = 20.0

    # --- 1. Solve with your `swarm` optimiser ---
    print("--- Running Swarm Optimiser ---")
    swarm_start_time = time.time()

    variables = [swarm.Variable(0, 1) for _ in range(n_vars)]

    swarm_optimiser = swarm.Optimiser.nsga(
        pop_size=pop_size,
        crossover=swarm.SbxParams(crossover_prob, crossover_eta),
        mutation=swarm.PmParams(mutation_prob, mutation_eta),
        seed=None,
    )

    swarm_result = swarm_optimiser.solve(zdt1_problem, variables, max_iter)

    print(f"Swarm finished in {time.time() - swarm_start_time:.4f} seconds.")
    print(f"Found {len(swarm_result.solutions)} solutions.")

    # --- 2. Solve with `pymoo` for benchmarking ---
    print("\n--- Running Pymoo Optimiser ---")
    pymoo_start_time = time.time()

    # Use the custom ZDT1 problem class
    pymoo_problem = ZDT1(n_var=n_vars)

    # Configure the NSGA-II algorithm with identical parameters
    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=crossover_prob, eta=crossover_eta),
        mutation=PM(prob=mutation_prob, eta=mutation_eta),
        eliminate_duplicates=True,
    )

    pymoo_res = minimize(
        pymoo_problem, algorithm, ("n_gen", max_iter), seed=None, verbose=False
    )

    print(f"Pymoo finished in {time.time() - pymoo_start_time:.4f} seconds.")
    print(f"Found {len(pymoo_res.F)} solutions.")

    # --- 3. Plot the comparison ---
    print("\nPlotting results...")
    # Get the known true pareto front from our custom problem class
    true_pf = pymoo_problem.pareto_front()
    plot_comparison(swarm_result.solutions, pymoo_res.F, true_pf)
