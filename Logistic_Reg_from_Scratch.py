from matplotlib import matplotlib_fname
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from pyrsistent import b
# linear function: f(x) = mx + b
# signmoid function: z(x) = 1 / (1 + e ** z)
# sigmoid applied to linear function: z(x) = 1 / (1 + e ** mx + b)

class Logistic_Regression:
    def __init__(self, X, l_rate, steps):
        self.l_rate = l_rate
        self.steps = steps
        self.n = len(X)

    def grad_desc(self, X, y):
        self.m = 0
        self.b = 0

        for i in range(self.steps):
            plt.clf()
            z = self.m * X + self.b
            y_pred = self.sigmoid(z)
            # get partial derivative for m from cost function
            m_der =  (1/self.n) * 2 * (sum(X * (y-y_pred)))
            # get partial derivative for b from cost function
            b_der =  (1/self.n) * 2 * (sum(y-y_pred))
            # m update
            self.m = self.m - self.l_rate * m_der
            # b update
            self.b = self.b - self.l_rate * b_der
            plt.scatter(X, y)
            plt.plot([min(X), max(y)], [min(y_pred), max(y_pred)], color='green')
            plt.pause(0.01)

        return plt.show()

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-1 * z))


if __name__ == '__main__':
    X = random.randint(20, size=(20))
    y = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    Logistic_Regression(X, 0.0001, 100).grad_desc(X,y)