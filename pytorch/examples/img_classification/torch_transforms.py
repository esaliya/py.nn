from utils.data_utils import load_data


def main(file):
    (X_train, Y_train), (X_test, Y_tst) = load_data(file)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


if __name__ == '__main__':
    file = "/Users/esaliya/sali/projects/lbl/ldrd_dnn/photon_dnn/ml_out.py"
    main(file)