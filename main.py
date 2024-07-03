import numpy as np
import pandas as pd
import zipfile  # for dealing with data packed in a zip - not compressed data is too large to put on git
import numpy.typing as npt


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
    # print(train_dataframe.head(10))
    # print(test_dataframe.head(10))
    train_array = np.array(train_dataframe, dtype=np.float32)
    test_array = np.array(test_dataframe, dtype=np.float32)

    # normalize pixel values so that they are between 0 and 1
    train_array[:, 1:] = train_array[:, 1:] / 255
    test_array[:, 1:] = test_array[:, 1:] / 255

    return train_array, test_array


def init_params(in_layer: int, hid_layer: int, out_layer: int, augmented: bool = True):
    """
    initializes neural network parameters, i.e. its weights and biases using simple uniform dist with values [-0.5,0.5)
    :return: W1, B1, W2, B2 - numpy 2D arrays with sizes corresponding to given in_layer, hid_layer, out_layer
    """
    # simple, uniform [-0.5; 0.5) dist
    if augmented is False:
        print("You chose uniform distribution init")
        W1 = np.random.rand(hid_layer, in_layer) - 0.5
        W2 = np.random.rand(out_layer, hid_layer) - 0.5
        B1 = np.random.rand(hid_layer, 1) - 0.5
        B2 = np.random.rand(out_layer, 1) - 0.5
        return W1, B1, W2, B2

    # kaiming He dist for the Relu layer
    rng = np.random.default_rng()  # providing no seed so that each time it provides different weights and biases
    mu = 0.0
    sigmaW1 = np.sqrt(2/in_layer)
    # sigmaW1 = 2/in_layer
    W1 = rng.normal(loc=mu, scale=sigmaW1, size=(hid_layer, in_layer))
    B1 = rng.normal(loc=mu, scale=sigmaW1, size=(hid_layer, 1))

    # Xavier Golort init for the softmax layer
    sigmaW2 = np.sqrt(2/(hid_layer+out_layer))
    # sigmaW2 = 2/(hid_layer+out_layer)
    W2 = rng.normal(loc=mu, scale=sigmaW2, size=(out_layer, hid_layer))
    B2 = rng.normal(loc=mu, scale=sigmaW1, size=(out_layer, 1))
    return W1, B1, W2, B2


def relu(x: npt.NDArray):
    """
    takes numpy array x and for each element computes ReLu.
    :param x: numpy ndarray
    :return: numpy ndarray where each element of given ndarray x is now ReLu of this element
    """
    # uses np.maximum to operate on the whole array at once
    return np.maximum(0, x)


def drelu(x: npt.NDArray):
    """
    takes numpy ndarray x and returns array after element-wise relu derivative, i.e. if element is less or
     equal 0 it is set to 0, else it is set to 1
    :param x:
    :return:
    """
    return np.choose(x <= 0, [1, 0])


def softmax(x: npt.NDArray):
    """
    this function calculates the softmax of a given vector (works for multi-dimensional arrays
    as well but vectors are the prime usecase of softmax), i.e it normalizes the vector to sum exactly to 1
    using formula  e^xi / sum (e^xi)
    :param x: numpy ndarray (it should be one-dimensional vector)
    :return: numpy ndarray of the same shape after performing softmax on it
    """
    # substracting max allows to be certain that float will not overflow
    exponents = np.exp(x - np.max(x))
    # exponents = np.exp(x)
    return exponents / np.sum(exponents, axis=None)


def forward(X, W1, B1, W2, B2):
    """
    this function performs forward propagation, i.e. it makes a prediction of input X using the network
    :return: L1, A1, L2, A2
    """
    L1 = W1 @ X + B1  # hid X in @ in X 1 = hid x 1
    A1 = relu(L1)
    L2 = W2 @ A1 + B2  # out X hid @ hid X 1 = out X 1
    A2 = softmax(L2)
    return L1, A1, L2, A2


def backward(X, Y, L1, A1, A2, W2):
    """
    this function calculates all of the derivatives needed to perform gradient descent
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
    :return: W1, B1, W2, B2
    """
    W1 -= dW1 * lr
    B1 -= dB1 * lr
    W2 -= dW2 * lr
    B2 -= dB2 * lr

    return W1, B1, W2, B2


def one_hot_column(y, count):
    """
    :return: y encoded as one_hot with size of (count, 1) - a column vector
    """
    arr = np.zeros(shape=(count, 1), dtype=np.uint8)
    arr[int(y)] = 1
    return arr


def train(lr, X, Y, W1, B1, W2, B2, iterations, batch_size, output_count):
    """
    this function is responsible for learning of the network using gradient descent.
    Returns network params afterwards
    :return: W1, B1, W2, B2
    """
    good_predictions = 0
    predictions_count = 0
    for i in range(iterations):
        # pick random training examples to form a mini-batch. Usually training many times on smaller batches
        # works better than training fewer times using bigger batches
        indexes = np.random.randint(0, X.shape[0], size=batch_size)
        X_batch = X[indexes]
        Y_batch = Y[indexes]

        # iterate through each training pair in the minibatch
        for x, y in zip(X_batch, Y_batch):
            x = x[:, np.newaxis]  # adds new axis, so that x is treated as a column vector
            y_vect = one_hot_column(y, output_count)
            L1, A1, L2, A2 = forward(x, W1, B1, W2, B2)
            prediction = np.argmax(A2)  # uses np.argmax to check which output is the most likely
            predictions_count += 1
            if prediction == y:
                good_predictions += 1
            # calculates all necessary gradients
            dW1, dB1, dW2, dB2 = backward(x, y_vect, L1, A1, A2, W2)
            # updates the parameters with gradients obtained from backward()
            W1, B1, W2, B2 = apply_learning_rate(lr, W1, dW1, B1, dB1, W2, dW2, B2, dB2)

        # prints current learning results from time to time
        if i > 0 and ((i < 500 and i % 50 == 0) or i % 200 == 0):
            print(f"learning iteration:{i},\tlearning dataset accuracy: {good_predictions/predictions_count * 100:.1f}%")
            good_predictions = 0
            predictions_count = 0

    return W1, B1, W2, B2


def test_network(X, Y, W1, B1, W2, B2):
    good_predictions = 0
    all_predictions = 0
    for x, y in zip(X, Y):
        x = x[:, np.newaxis]  # adds new axis, so that x is treated as a column vector
        L1, A1, L2, A2 = forward(x, W1, B1, W2, B2)
        prediction = np.argmax(A2)
        all_predictions += 1
        if prediction == y:
            good_predictions += 1
    print(3 * "\n" + 30 * "-")
    print(f"Test data accuracy: {good_predictions/ all_predictions}")


def main():
    # Read input data
    train_data, test_data = read_input_data()
    X_train = train_data[:, 1:]
    Y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    Y_test = test_data[:, 0]

    # declare hyperparameters
    INPUT_FEATURES_AMOUNT = X_train.shape[1]
    hidden_layer_count: int = 12
    OUTPUT_LAYER_COUNT = 10  # for this use-case (handwritten digits) it will always be 10
    LEARNING_RATE = 0.1

    print("Welcome to MNIST handwritten digits classifier!")
    augmented_choice = "A"
    choice = input(f"Which param init do you wish to use? print {augmented_choice} for augmented inits or print"
                   f" anything else for simple uniform distribution\n")
    iterations = int(input("How many iterations do you want to perform?\n"))
    # initialize network parameters
    W1, B1, W2, B2 = init_params(INPUT_FEATURES_AMOUNT, hidden_layer_count, OUTPUT_LAYER_COUNT, augmented_choice == choice)

    # perform gradient descent
    W1, B1, W2, B2 = train(LEARNING_RATE, X_train, Y_train, W1, B1, W2, B2, iterations, 64, OUTPUT_LAYER_COUNT)

    # test the network on another dataset
    test_network(X_test, Y_test, W1, B1, W2, B2)


if __name__ == "__main__":
    main()
