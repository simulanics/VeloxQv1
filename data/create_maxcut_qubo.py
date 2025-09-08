import numpy as np
from veloxq_reconstructed import save_qubo_npz, maxcut_to_qubo, random_maxcut_weights

# small reproducible graph
n = 6
p = 0.5
seed = 123

W = random_maxcut_weights(n, p, seed=seed)
qubo = maxcut_to_qubo(W)

# Save to disk
save_qubo_npz("example_qubo.npz", qubo.Q, qubo.b)
print("Saved example_qubo.npz with", n, "variables.")
