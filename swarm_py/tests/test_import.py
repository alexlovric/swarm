"""
This is a basic import test for swarm_py. 
Used to verify that the package is installed correctly on the various operating systems.
"""

import swarm_py as sp

# Create variable basic build test
var = sp.Variable(0, 1)
print(f"Successfully created a swarm.Variable object: {var}")
print("Basic import test passed!")
