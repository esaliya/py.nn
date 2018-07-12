import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    z = sigmoid(z)
    return z*(1-z)


class NeuralNetwork():
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.yHat = None
        self.delta3 = None
        self.delta2 = None
        self.dJdW2 = None
        self.dJdW1 = None

    def forward(self, X):
        self.z2 = X @ self.W1
        self.a2 = sigmoid(self.z2)
        self.z3 = self.a2 @ self.W2
        self.yHat = sigmoid(self.z3)
        return self.yHat

    def backward(self, X, y):
        self.delta3 = (-(y - self.yHat) * sigmoid_prime(self.z3))
        self.dJdW2 = np.transpose(self.a2) @ self.delta3
        self.delta2 = (self.delta3 @ np.transpose(self.W2)) * sigmoid_prime(self.z2)
        self.dJdW1 = np.transpose(X) @ self.delta2

    def update_weights(self, eta):
        self.W1 = self.W1 - eta * self.dJdW1
        self.W2 = self.W2 - eta * self.dJdW2

    def cost(self, y):
        return 0.5*sum((y-self.yHat)**2)
