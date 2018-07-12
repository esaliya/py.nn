from matplotlib.pyplot import plot, grid, xlabel, ylabel

from nn_demyst.partSix import *

if __name__ == '__main__':
    NN = Neural_Network()
    T = trainer(NN)
    T.train(X, y)
    plot(T.J)
    grid(1)
    xlabel('Iterations')
    ylabel('Cost')