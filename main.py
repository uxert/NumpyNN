import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile  # for dealing with data packed in a zip - not compressed data is too large to put on git


def main():

    # data import
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


if __name__ == "__main__":
    main()