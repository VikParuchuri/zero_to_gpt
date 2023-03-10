import matplotlib.pyplot as plt
import numpy as np

class Optimizer():
    def __init__(self):
        self.w_vals = []

    def save_vector(self, layers):
        self.w_vals.append((
            np.mean(layers[-1].weights[0, 0]),
            np.mean(layers[-1].weights[1, 0])
        ))

    def plot_path(self):
        indices = np.linspace(0, len(self.w_vals)-1, 500, dtype=int)
        positions = [self.w_vals[i] for i in indices]
        x = [p[0] for p in positions]
        y = [p[1] for p in positions]
        plt.scatter([x[0]], [y[0]], color='red')
        plt.scatter([x[-1]], [y[-1]], color='green')
        plt.plot(x, y)
        plt.show()