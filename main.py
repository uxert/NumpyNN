import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile  # for dealing with data packed in a zip - not compressed data is too large to put on git
import warnings
warnings.simplefilter('error')

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
    train_array = np.array(train_dataframe, dtype=np.float32)
    test_array = np.array(test_dataframe, dtype=np.float32)
    # normalizes pixel values so that they are between 0 and 1
    train_array[:, 1:] = train_array[:, 1:] / 255
    test_array[:, 1:] = test_array[:, 1:] / 255

    return train_array, test_array


def init_params(in_layer, hid_layer, out_layer):
    """
    initializes neural network parameters, i.e. weights and biases
    :return: tuple(W1, B1, W2, B2), all with sizes corresponding to given in_layer, hid_layer, out_layer
    """
    # will try out Kaiming He initialization in the future
    W1 = np.random.rand(hid_layer, in_layer) - 0.5
    W2 = np.random.rand(out_layer, hid_layer) - 0.5
    B1 = np.random.rand(hid_layer, 1) - 0.5
    B2 = np.random.rand(out_layer, 1) - 0.5
    return W1, B1, W2, B2


def relu(x):
    # uses np.maximum to operate on the whole array at once
    return np.maximum(0, x)


def drelu(x):
    return np.choose(x <= 0, [1, 0])


def softmax(x):
    try:
        exponents = np.exp(x - np.max(x))
        return exponents / np.sum(exponents)
    except:
        print("come on")


def forward(X, W1, B1, W2, B2):
    L1 = W1 @ X + B1  # hid X in @ in X 1 = hid x 1
    A1 = relu(L1)
    L2 = W2 @ A1 + B2  # out X hid @ hid X 1 = out X 1
    A2 = softmax(L2)
    return L1, A1, L2, A2


def backward(X, Y, L1, A1, A2, W2):
    """
    :param X: input given to the network
    :param Y: one-hot vector with desired number
    :param A1: hidden layer activations
    :param A2: output layer activations
    :return: dW1, dB1, dW2, dB2
    """
    dL2 = (A2 - Y) / Y.size
    dW2 = dL2 @ A1.T
    dB2 = dL2
    dL1 = W2.T @ dL2 * drelu(L1)
    dW1 = dL1 @ X.T
    dB1 = dL1

    return dW1, dB1, dW2, dB2


def apply_learning_rate(lr, W1, dW1, B1, dB1, W2, dW2, B2, dB2):
    """
    this function applies given learning rate and applies it to all parameters
    :param lr: learning rate
    :return:
    """
    W1 -= (dW1 * lr)
    B1 -= (dB1 * lr)
    W2 -= (dW2 * lr)
    B2 -= (dB2 * lr)

    return W1, B1, W2, B2


def one_hot(y, count):
    """
    :return: y encoded as one_hot with size of (count, 1) - a column vector
    """
    arr = np.zeros(shape=(count, 1), dtype=np.uint8)
    arr[int(y)] = 1
    return arr


def learn(lr, X, Y, W1, B1, W2, B2, iterations, batch_size, output_count):
    """
    this function is responsible for learning of the network using gradient descent.
    Returns params afterwards
    :return: W1, B1, W2, B2
    """
    good_predictions = 0
    for i in range(iterations):
        indexes = np.random.randint(0, X.shape[0], size=batch_size)
        X_batch = X[indexes]
        Y_batch = Y[indexes]
        for x, y in zip(X_batch, Y_batch):
            x = x[:, np.newaxis]  # adds new axis, so that x is treated as a column vector
            y_vect = one_hot(y, output_count)
            L1, A1, L2, A2 = forward(x, W1, B1, W2, B2)
            prediction = np.argmax(A2)
            if prediction == y:
                good_predictions += 1
            dW1, dB1, dW2, dB2 = backward(x, y_vect, L1, A1, A2, W2)
            W1, B1, W2, B2 = apply_learning_rate(lr, W1, dW1, B1, dB1, W2, dW2, B2, dB2)
        if i > 0 and ((i < 500 and i % 50 == 0) or i % 200 == 0):
            print(f"learning iteration:{i},\tlearning dataset accuracy: {good_predictions/(i * batch_size) * 100}:.1f%")

    return W1, B1, W2, B2


def test_network(X, Y, W1, B1, W2, B2):
    good_predictions = 0
    for x, y in zip(X, Y):
        x = x[:, np.newaxis]  # adds new axis, so that x is treated as a column vector
        L1, A1, L2, A2 = forward(x, W1, B1, W2, B2)
        prediction = np.argmax(A2)
        if prediction == y:
            good_predictions += 1
    print(3 * "\n" + 30 * "-")
    print(f"Test data accuracy: {good_predictions/ X.shape[0]}")


def main():
    train_data, test_data = read_input_data()
    X_train = train_data[:, 1:]
    Y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    Y_test = test_data[:, 0]
    INPUT_FEATURES_AMOUNT = X_train.shape[1]
    hidden_layer_count: int = 10
    OUTPUT_LAYER_COUNT = 10  # for this use-case (handwritten digits) it will always be 10
    LEARNING_RATE = 0.1

    W1, B1, W2, B2 = init_params(INPUT_FEATURES_AMOUNT, hidden_layer_count, OUTPUT_LAYER_COUNT)
    W1, B1, W2, B2 = learn(LEARNING_RATE, X_train, Y_train, W1, B1, W2, B2, 6000, 32, OUTPUT_LAYER_COUNT)

    test_network(X_test, Y_test, W1, B1, W2, B2)


if __name__ == "__main__":
    main()

