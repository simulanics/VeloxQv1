import numpy as np

# These are the exact arrays for example_qubo.npz
Q = np.array([
 [ 0.     , 0.18727, 0.73239, 0.47088, 0.47445, 0.48639],
 [ 0.18727, 0.     , 0.42527, 0.81272, 0.20737, 0.34842],
 [ 0.73239, 0.42527, 0.     , 0.53986, 0.70310, 0.46365],
 [ 0.47088, 0.81272, 0.53986, 0.     , 0.91825, 0.52088],
 [ 0.47445, 0.20737, 0.70310, 0.91825, 0.     , 0.55276],
 [ 0.48639, 0.34842, 0.46365, 0.52088, 0.55276, 0.     ]
], dtype=float)

b = np.array([
 -(Q[0].sum()),
 -(Q[1].sum()),
 -(Q[2].sum()),
 -(Q[3].sum()),
 -(Q[4].sum()),
 -(Q[5].sum()),
], dtype=float)

np.savez_compressed("example_qubo.npz", Q=Q, b=b)
print("Saved example_qubo.npz")
