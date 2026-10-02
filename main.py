## Test
import numpy as np
from algoritm import RFF

# här kan vi leka runt med parametrar
N = 500
d = 5
D = 1000
gamma = 0.5

# ändrar man 10 så får man ett annat seed, så andra omega samples.
rng = np.random.default_rng(10)

X = rng.normal(0, 1/np.sqrt(d), size = (N, d))

rff = RFF(D = D, gamma = gamma, random_state=10)

Z = rff.fit_transform(X)

print("X shape: ", X.shape)
print("omega shape: ", rff.omega.shape)
print("Z shape: ", Z.shape)