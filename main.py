import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile  # for dealing with data packed in a zip - not compressed data is too large to put on git


def read_input_data():
    """
    reads data from files in MNIST_datasets.zip archive and returns them as np.array()
    :return: tuple(train_data, test_data), both are numpy arrays
    """
    zip_file_name = "MNIST_datasets.zip"
    train_csv_name = "mnist_train.csv"
    test_csv_name = "mnist_test.csv"

    with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
        with zip_ref.open(train_csv_name) as f:
            train_dataframe = pd.read_csv(f)
        with zip_ref.open(test_csv_name) as f:
            test_dataframe = pd.read_csv(f)
    print(train_dataframe.head(10))
    print(test_dataframe.head(10))
    train_array = np.array(train_dataframe, dtype=np.uint16)
    test_array = np.array(test_dataframe, dtype=np.uint16)

    return train_array, test_array


def init_params(in_layer, hid_layer, out_layer):
    """
    initializes neural network parameters, i.e. weights and biases
    :return: tuple(W1, B1, W2, B2), all with sizes corresponding to given in_layer, hid_layer, out_layer
    """
    # will try out Kaiming He initialization in the future
    W1 = np.random.rand(hid_layer, in_layer) - 0.5
    W2 = np.random.rand(out_layer, hid_layer)
    B1 = np.random.rand(hid_layer, 1)
    B2 = np.random.rand(out_layer, 1)
    return W1, B1, W2, B2


def main():
    train_data, test_data = read_input_data()
    X_train = train_data[:, 1:]
    Y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    Y_test = test_data[:, 0]
    INPUT_FEATURES_AMOUNT = X_train.shape[1]
    hidden_layer_count: int = 10
    OUTPUT_LAYER_COUNT = 10  # for this use-case (handwritten digits) it will always be 10
    print(f"{Y_test.shape=}")

    W1, B1, W2, B2 = init_params(INPUT_FEATURES_AMOUNT, hidden_layer_count, OUTPUT_LAYER_COUNT)



if __name__ == "__main__":
    main()
