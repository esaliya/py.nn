import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import importlib
from pathlib import Path
import numpy as np


def load_data(examples_file):
    stem = Path(examples_file).stem
    loader = importlib.machinery.SourceFileLoader(stem, examples_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is not None:
        examples = importlib.util.module_from_spec(spec)
        loader.exec_module(examples)
    else:
        print("Error! Module ", spec, " cannot be found!")
        return

    (X_train, Y_train), (X_test, Y_test) = (examples.X_train, examples.y_train), (examples.X_test, examples.y_test)

    #split data later, now combine them for processing
    return (X_train, Y_train), (X_test, Y_test)


def main(file):
    (X_train, Y_train), (X_test, Y_test) = load_data(file)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    x = X_train.copy()
    xt = X_test.copy()
    y = Y_train.copy()
    yt = Y_test.copy()

    print(x.shape, xt.shape)
    x = x[:, 0:25]
    xt = xt[:, 0:25]
    print(x.shape, xt.shape)

    x = x.reshape((x.shape[0], 1, 5, 5))
    xt = xt.reshape((xt.shape[0], 1, 5, 5))
    print(x.shape, xt.shape)

    print(x[0][0][0][0])


if __name__ == '__main__':
    file = "/Users/esaliya/sali/projects/lbl/ldrd_dnn/photon_dnn/ml_out.py"
    main(file)