import numpy as np

# data will come from a known distribution
def generate_batch(batch_size=32, seed=0):
    np.random.seed(seed)
    #x = 2 - 3 * np.random.normal(0, 1, batch_size)
    x = np.array([i for i in range(0, batch_size)])
    y = -4*x + 0.02 * (x ** 2) + -0.00001 * (x ** 3) + np.random.normal(-30, 30, batch_size)
    return x, y