import math, random

def generate_xor_dataset(size):
    X = []
    y = []
    for i in range(size):
        X.append([random.uniform(0, 49) / 100, random.uniform(0, 49) / 100])
        y.append([0])
        X.append([random.uniform(0, 49) / 100, random.uniform(50, 100) / 100])
        y.append([1])
        X.append([random.uniform(50, 100) / 100, random.uniform(0, 49) / 100])
        y.append([1])
        X.append([random.uniform(50, 100) / 100, random.uniform(50, 100) / 100])
        y.append([0])

    return X, y

def train_test_slit(X, y, split):

    assert len(X) == len(y), "X and y are not same length, can't do train/test split"

    num_train = int(len(X) * split)

    X_train = X[:num_train]
    y_train = y[:num_train]

    X_test = X[num_train:]
    y_test = y[num_train:]

    return X_train, y_train, X_test, y_test