"""
A high-performance, parallel optimisation library for Python, implemented in Rust.
"""

from ._swarm import (
    Optimiser,
    OptimiserResult,
    PsoParams,
    SbxParams,
    PmParams,
    Solution,
    Variable,
)

# The __all__ variable defines the public API of the package.
__all__ = [
    "Optimiser",
    "OptimiserResult",
    "PsoParams",
    "SbxParams",
    "PmParams",
    "Solution",
    "Variable",
]
