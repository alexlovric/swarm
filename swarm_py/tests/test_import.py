import swarm_py as sp

print(f"Successfully imported swarm version: {sp.__version__}")
var = sp.Variable(0, 1)
print(f"Successfully created a swarm.Variable object: {var}")
print("Basic import test passed!")