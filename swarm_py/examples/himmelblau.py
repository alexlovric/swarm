import swarm
import time


def himmelblau_problem(x):
    """
    Himmelblau's function, a standard 2D benchmark for optimisation.

    f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    It has four global minima, all with a value of 0.

    Note: The function must be of the signature:

        def blackbox(x: list[float]) -> tuple[list[float], list[float] | None]

    Where:
        - x: A list of input variable values `x`.
        - Returns: A tuple containing the objective values `f` and optional constraint violations `g`.
    """

    f = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    return ([f], None)


if __name__ == "__main__":
    start_time = time.time()

    variables = [swarm.Variable(-5, 5), swarm.Variable(-5, 5)]
    print(variables)

    max_iter = 100

    optimiser = swarm.Optimiser.pso(50)
    # optimiser = swarm.Optimiser.nsga(50)
    print(optimiser)

    result = optimiser.solve(himmelblau_problem, variables, max_iter)
    # result = optimiser.solve_par(himmelblau_problem, variables, max_iter)
    print(result.solutions)

    print(f"Execution time: {(time.time() - start_time)*1000:.4f} milliseconds")
