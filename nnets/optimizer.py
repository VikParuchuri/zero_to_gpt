import matplotlib.pyplot as plt
import numpy as np

class Optimizer():
    def __init__(self):
        self.w_vals = []

    def save_vector(self, layers):
        # Do SVD on the matrix to get singular values
        _, singular, _ = np.linalg.svd(layers[-1].weights)
        # Add the final layer singular value to the list
        self.w_vals.append(singular[0])

    def plot_path(self):
        indices = np.linspace(0, len(self.w_vals)-1, 500, dtype=int)
        y = [self.w_vals[i] for i in indices]
        x = [i for i in range(len(y))]

        # Plot how the final layer singular value changes over time
        plt.scatter([x[0]], [y[0]], color='red')
        plt.scatter([x[-1]], [y[-1]], color='green')
        plt.plot(x, y)
        plt.show()

class Scheduler():
    def __init__(self):
        pass

    def __call__(self):
        pass

    def plot_lr(self):
        x = np.linspace(0, self.total_steps, 500, dtype=int)
        y = [self(i) for i in x]
        plt.plot(x, y)
        plt.show()