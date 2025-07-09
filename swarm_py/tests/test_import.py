import swarm_py as swarm

print(f"Successfully imported swarm version: {swarm.__version__}")
var = swarm.Variable(0, 1)
print(f"Successfully created a swarm.Variable object: {var}")
print("Basic import test passed!")