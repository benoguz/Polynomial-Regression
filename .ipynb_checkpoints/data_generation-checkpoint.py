import numpy as np

# data will come from a known distribution
def generate_batch(batch_size=32):
    np.random.seed(12)
    x = np.random.random(batch_size)*30 - 1

    # sd is a function of x
    sd = 0.02 + 0.04 * (x + 9)

    # target = mean + noise * sd
    y = np.cos(x) - 0.2 * x + np.random.randn(batch_size) * sd

    return x, y