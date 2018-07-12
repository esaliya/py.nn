import numpy as np
from nn_demyst_by_hand.neuralnetwork import NeuralNetwork


def gen_data():
    # X = (hours sleeping, hours studying), y = Score on test
    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)

    # Normalize
    X = X / np.amax(X, axis=0)
    y = y / 100  # Max test score is 100

    return X, y


def plot_data(X, y):
    pass


if __name__ == '__main__':
    n = NeuralNetwork()
    X, y = gen_data()

    print(y)
    eta = 1e-1
    for t in range(500):
        yHat = n.forward(X)
        # print(f"iter {t}\n{yHat}")
        # print(f"iter {t}\n{yHat}\n{n.W1}")
        print(n.cost(y))
        n.backward(X, y)
        n.update_weights(eta)
