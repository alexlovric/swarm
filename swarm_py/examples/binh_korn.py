import swarm_py as sp
import time
import matplotlib.pyplot as plt


def binh_and_korn_problem(x):
    """
    The Binh and Korn multi-objective test function.
    This function has two objectives to be minimised and is subject to two constraints.

    - Objectives:
      - f1(x, y) = 4*x^2 + 4*y^2
      - f2(x, y) = (x - 5)^2 + (y - 5)^2

    - Constraints (must be <= 0 for feasibility):
      - g1(x, y) = (x - 5)^2 + y^2 - 25
      - g2(x, y) = 7.7 - (x - 8)^2 - (y + 3)^2

    - Variable bounds:
      - 0 <= x <= 5
      - 0 <= y <= 3

    The function signature matches the required blackbox format:
    def blackbox(x: list[float]) -> tuple[list[float], list[float] | None]
    """
    x1, x2 = x[0], x[1]

    # Calculate the two objective values
    f1 = 4.0 * x1**2 + 4.0 * x2**2
    f2 = (x1 - 5.0) ** 2 + (x2 - 5.0) ** 2
    objectives = [f1, f2]

    # Calculate the two constraint violations
    g1 = (x1 - 5.0) ** 2 + x2**2 - 25.0
    g2 = -((x1 - 8.0) ** 2 + (x2 + 3.0) ** 2 - 7.7)
    constraints = [g1, g2]

    return (objectives, constraints)


def plot_pareto_front(solutions):
    """
    Plots the Pareto front from a list of solutions.
    """
    if not solutions:
        print("No solutions found to plot.")
        return

    # Extract the objective values (f1, f2) from each solution
    f1_values = [sol.f[0] for sol in solutions]
    f2_values = [sol.f[1] for sol in solutions]

    plt.figure(figsize=(8, 6))
    plt.scatter(f1_values, f2_values, c="blue", label="Pareto Front")
    plt.title("Pareto Front for Binh and Korn Problem")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    start_time = time.time()

    # Define variable bounds and optimisation settings
    variables = [sp.Variable(0, 5), sp.Variable(0, 3)]
    print(variables)

    max_iter = 250

    # Configure the NSGA-II optimiser
    optimiser = sp.Optimiser.nsga(pop_size=100)
    print(optimiser)

    # Run the optimisation
    result = optimiser.solve(binh_and_korn_problem, variables, max_iter)
    # result = optimiser.solve_par(binh_and_korn_problem, variables, max_iter)

    print(
        f"\nOptimisation finished in {(time.time() - start_time)*1000:.4f} milliseconds."
    )
    print(f"Found {len(result.solutions)} solutions in the Pareto front.")

    # Plot the results
    plot_pareto_front(result.solutions)
