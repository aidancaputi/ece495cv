import math, random

def generate_dataset(dataset_name):

    X = []
    y = []

    if(dataset_name == 'xor'):
        X.append([0.0, 0.0])
        y.append([0])
        X.append([0.0, 1.0])
        y.append([1])
        X.append([1.0, 0.0])
        y.append([1])
        X.append([1.0, 1.0])
        y.append([0])

    else:

        X.append([0.0, 0.0, 0.0, 0.0, 0.0])        
        X.append([0.0, 0.0, 0.0, 0.0, 1.0])
        y.append([0, 0, 0])
        y.append([0, 1, 0])

        X.append([0.0, 0.0, 0.0, 1.0, 0.0])
        X.append([0.0, 0.0, 0.0, 1.0, 1.0])
        y.append([0, 1, 0])
        y.append([1, 0, 1])

        X.append([0.0, 0.0, 1.0, 0.0, 0.0])
        X.append([0.0, 0.0, 1.0, 0.0, 1.0])
        y.append([0, 1, 0])
        y.append([1, 0, 1])

        X.append([0.0, 0.0, 1.0, 1.0, 0.0])
        X.append([0.0, 0.0, 1.0, 1.0, 1.0])
        y.append([1, 0, 1])
        y.append([1, 1, 1])

        X.append([0.0, 1.0, 0.0, 0.0, 0.0])
        X.append([0.0, 1.0, 0.0, 0.0, 1.0])
        y.append([0, 1, 0])
        y.append([1, 0, 1])

        X.append([0.0, 1.0, 0.0, 1.0, 0.0])
        X.append([0.0, 1.0, 0.0, 1.0, 1.0])
        y.append([1, 0, 1])
        y.append([1, 1, 1])

        X.append([0.0, 1.0, 1.0, 0.0, 0.0])
        X.append([0.0, 1.0, 1.0, 0.0, 1.0])
        y.append([1, 0, 1])
        y.append([1, 1, 1])

        X.append([0.0, 1.0, 1.0, 1.0, 0.0])
        X.append([0.0, 1.0, 1.0, 1.0, 1.0])
        y.append([1, 1, 1])
        y.append([1, 0, 1])

        X.append([1.0, 0.0, 0.0, 0.0, 0.0])
        X.append([1.0, 0.0, 0.0, 0.0, 1.0])
        y.append([1, 0, 0])
        y.append([1, 1, 0])

        X.append([1.0, 0.0, 0.0, 1.0, 0.0])
        X.append([1.0, 0.0, 0.0, 1.0, 1.0])
        y.append([1, 1, 0])
        y.append([0, 0, 1])

        X.append([1.0, 0.0, 1.0, 0.0, 0.0])
        X.append([1.0, 0.0, 1.0, 0.0, 1.0])
        y.append([1, 1, 0])
        y.append([0, 0, 1])

        X.append([1.0, 0.0, 1.0, 1.0, 0.0])
        X.append([1.0, 0.0, 1.0, 1.0, 1.0])
        y.append([0, 0, 1])
        y.append([0, 1, 1])

        X.append([1.0, 1.0, 0.0, 0.0, 0.0])
        X.append([1.0, 1.0, 0.0, 0.0, 1.0])
        y.append([1, 1, 0])
        y.append([0, 0, 1])

        X.append([1.0, 1.0, 0.0, 1.0, 0.0])
        X.append([1.0, 1.0, 0.0, 1.0, 1.0])
        y.append([0, 0, 1])
        y.append([0, 1, 1])

        X.append([1.0, 1.0, 1.0, 0.0, 0.0])
        X.append([1.0, 1.0, 1.0, 0.0, 1.0])
        y.append([0, 1, 1])
        y.append([0, 0, 1])

        X.append([1.0, 1.0, 1.0, 1.0, 0.0])
        X.append([1.0, 1.0, 1.0, 1.0, 1.0])
        y.append([0, 0, 1])
        y.append([0, 1, 1])

    return X, y

def train_test_slit(X, y, split):

    assert len(X) == len(y), "X and y are not same length, can't do train/test split"

    num_train = int(len(X) * split)

    X_train = X[:num_train]
    y_train = y[:num_train]

    X_test = X[num_train:]
    y_test = y[num_train:]

    return X_train, y_train, X_test, y_test