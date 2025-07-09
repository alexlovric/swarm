from typing import Callable, TypeAlias

# --- Type Aliases -----------------------------------------------------------

BlackboxFunc: TypeAlias = Callable[
    [list[float]], tuple[list[float], list[float] | None]
]
"""
A callable blackbox function.

Parameters
----------
list[float]
    A list of input variable values `x`.

Returns
-------
tuple[list[float], list[float] | None]
    A tuple containing the objective values `f` and optional constraint violations `g`.
"""

# --- Parameter Classes ------------------------------------------------------

class SbxParams:
    """Parameters for the Simulated Binary Crossover (SBX) operator."""

    def __init__(self, prob: float, eta: float) -> None:
        """Instantiate SBX parameters.

        Parameters
        ----------
        prob : float
            The probability of applying crossover.
        eta : float
            The distribution index for crossover.
        """
        ...

class PmParams:
    """Parameters for the Polynomial Mutation operator."""

    def __init__(self, prob: float, eta: float) -> None:
        """Instantiate Polynomial Mutation parameters.

        Parameters
        ----------
        prob : float
            The probability of applying mutation.
        eta : float
            The distribution index for mutation.
        """
        ...

class PsoParams:
    """Parameters for the Particle Swarm Optimisation (PSO) algorithm."""

    def __init__(
        self, inertia: float, cognitive_coeff: float, social_coeff: float
    ) -> None:
        """Instantiate PSO parameters.

        Parameters
        ----------
        inertia : float
            The inertia weight controlling particle momentum.
        cognitive_coeff : float
            The cognitive coefficient (attraction to personal best).
        social_coeff : float
            The social coefficient (attraction to global best).
        """
        ...

# --- Core Classes -----------------------------------------------------------

class Variable:
    """Represents a single variable with its lower and upper bounds."""

    min: float
    max: float

    def __init__(self, min: float, max: float) -> None:
        """Instantiate a floating point variable.

        Parameters
        ----------
        min : float
            The lower bound of the variable.
        max : float
            The upper bound of the variable.
        """
        ...

class Solution:
    """
    Represents a single solution found by the optimiser.

    Attributes
    ----------
    x : list[float]
        The list of variable values for this solution.
    f : list[float]
        The list of objective values for this solution.
    g : list[float] | None
        The list of constraint violation values, if any.
    """

    x: list[float]
    f: list[float]
    g: list[float] | None

class OptimiserResult:
    """
    A container for the final results of an optimisation run.

    Attributes
    ----------
    solutions : list[Solution]
        A list of the solutions found in the final population or pareto front.
    n_iterations : int
        The total number of iterations performed.
    execution_time : float
        The total execution time of the solver in seconds.
    """

    solutions: list[Solution]
    n_iterations: int
    execution_time: float

class Optimiser:
    """The main optimiser class for running optimisation algorithms."""

    @staticmethod
    def nsga(
        pop_size: int,
        crossover: SbxParams,
        mutation: PmParams,
        seed: int | None = None,
    ) -> "Optimiser":
        """Configure an optimiser instance to use the NSGA-II algorithm.

        Parameters
        ----------
        pop_size : int
            The size of the population.
        crossover : SbxParams
            The parameters for the crossover operator.
        mutation : PmParams
            The parameters for the mutation operator.
        seed : int | None, optional
            An optional seed for the random number generator.

        Returns
        -------
        Optimiser
            A new optimiser instance configured for NSGA-II.
        """
        ...

    @staticmethod
    def pso(
        n_particles: int,
        params: PsoParams,
        penalty_multiplier: float | None = None,
        seed: int | None = None,
    ) -> "Optimiser":
        """Configure an optimiser instance to use the Particle Swarm (PSO) algorithm.

        Parameters
        ----------
        n_particles : int
            The number of particles in the swarm.
        params : PsoParams
            The core parameters for the PSO algorithm.
        penalty_multiplier : float | None, optional
            The multiplier for the penalty constraint handler. If None,
            constraints are not handled by a penalty function.
        seed : int | None, optional
            An optional seed for the random number generator.

        Returns
        -------
        Optimiser
            A new optimiser instance configured for PSO.
        """
        ...

    def solve(
        self,
        func: BlackboxFunc,
        vars: list[Variable],
        max_iter: int,
    ) -> OptimiserResult:
        """Solve an optimisation problem using a single thread.

        Parameters
        ----------
        func : Callable
            The objective function to optimise.
        vars : list[Variable]
            A list defining the bounds for each decision variable.
        max_iter : int
            The maximum number of iterations to run.

        Returns
        -------
        OptimiserResult
            An object containing the results of the optimisation.
        """
        ...

    def solve_par(
        self,
        func: BlackboxFunc,
        vars: list[Variable],
        max_iter: int,
    ) -> OptimiserResult:
        """Solve an optimisation problem in parallel using multiple threads.

        Parameters
        ----------
        func : Callable
            The objective function to optimise.
        vars : list[Variable]
            A list defining the bounds for each decision variable.
        max_iter : int
            The maximum number of iterations to run.

        Returns
        -------
        OptimiserResult
            An object containing the results of the optimisation.
        """
        ...
