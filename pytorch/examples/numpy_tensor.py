import numpy as np


def relu(x):
    return np.maximum(x, 0)

# N -- Batch size
# D_in -- Input dimension
# H -- Hidden dimension
# D_out -- Output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# set random seed (for reproducible results)
# np.random.seed(10)

# Generate input and output
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Generate weights
W1 = np.random.randn(D_in, H)
W2 = np.random.randn(H, D_out)

learning_rate = 1e-6

for t in range(500):
    # Forward pass: compute predicted yHat
    z2 = x @ W1
    a2 = relu(z2)
    z3 = a2 @ W2
    yHat = z3 # In this example the output layer function is f(x)=x

    J = np.square(y - yHat).sum()
    print(t, J)

    # Backpropagation
    # dJdW2 = (a2).T @ del3
    # del3 = -(y-yHat)*f'(z3) ('*' is element-wise multiplication)

    # Note. 2.0 because our loss is not 1/2*sum-of-squares
    #       f'(z3) is 1 because f for output layer is f(x)=x
    del3 = -2.0*(y - yHat)
    dJdW2 = a2.T @ del3

    # dJdW1 = (x).T @ del2
    # Note. Here x is like the activation from previous layer
    # del2 = del3@(W2).T * f'(z2) ('*" is element-wise multiplication)
    # Note. Derivative of Relu -- f'(z2) is z2 where all z2i < 0 are set to zero
    #       Therefore, in the element-wise multiplication we can do a trick
    #       by setting the elements of the first matrix to zero where the
    #       corresponding element in the second (f'(z2)) matrix is less than zero
    del2 = (del3 @ W2.T)
    del2[z2 < 0] = 0
    dJdW1 = x.T @ del2

    # Update weights
    W1 -= learning_rate * dJdW1
    W2 -= learning_rate * dJdW2


