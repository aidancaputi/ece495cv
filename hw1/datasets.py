import math, random

def generate_dataset(dataset_name):

    X = []
    y = []

    if(dataset_name == 'xor'):
        for i in range(100):
            X.append([random.uniform(0, 49) / 100, random.uniform(0, 49) / 100])
            y.append([0])
            X.append([random.uniform(0, 49) / 100, random.uniform(50, 100) / 100])
            y.append([1])
            X.append([random.uniform(50, 100) / 100, random.uniform(0, 49) / 100])
            y.append([1])
            X.append([random.uniform(50, 100) / 100, random.uniform(50, 100) / 100])
            y.append([0])

    else:
        for i in range(100):
            #00000 -> 000
            X.append([random.uniform(0, 49) / 100, random.uniform(0, 49) / 100, random.uniform(0, 49) / 100, random.uniform(0, 49) / 100, random.uniform(0, 49) / 100])
            y.append([0, 0, 0])

            #10000 -> 100
            X.append([random.uniform(50, 100) / 100, random.uniform(0, 49) / 100, random.uniform(0, 49) / 100, random.uniform(0, 49) / 100, random.uniform(0, 49) / 100])
            y.append([1, 0, 0])

            #01000 -> 010
            X.append([random.uniform(0, 49) / 100, random.uniform(50, 100) / 100, random.uniform(0, 49) / 100, random.uniform(0, 49) / 100, random.uniform(0, 49) / 100])
            y.append([0, 1, 0])

            #10100 -> 010
            X.append([random.uniform(50, 100) / 100, random.uniform(0, 49) / 100, random.uniform(50, 100) / 100, random.uniform(0, 49) / 100, random.uniform(0, 49) / 100])
            y.append([0, 1, 0])

            #01010 -> 001
            X.append([random.uniform(0, 49) / 100, random.uniform(50, 100) / 100, random.uniform(0, 49) / 100, random.uniform(50, 100) / 100, random.uniform(0, 49) / 100])
            y.append([0, 0, 1])

            #01001 -> 110
            X.append([random.uniform(0, 49) / 100, random.uniform(50, 100) / 100, random.uniform(0, 49) / 100, random.uniform(0, 49) / 100, random.uniform(50, 100) / 100])
            y.append([1, 1, 0])

            #10101 -> 110
            X.append([random.uniform(50, 100) / 100, random.uniform(0, 49) / 100, random.uniform(50, 100) / 100, random.uniform(0, 49) / 100, random.uniform(50, 100) / 100])
            y.append([1, 1, 0])

            #11111 -> 111
            X.append([random.uniform(50, 100) / 100, random.uniform(50, 100) / 100, random.uniform(50, 100) / 100, random.uniform(50, 100) / 100, random.uniform(50, 100) / 100])
            y.append([1, 1, 1])

    return X, y

def generate_dataset2(dataset_name):

    X = []
    y = []

    if(dataset_name == 'xor'):
        for i in range(100):
            X.append([0.0, 0.0])
            y.append([0])
            X.append([0.0, 1.0])
            y.append([1])
            X.append([1.0, 0.0])
            y.append([1])
            X.append([1.0, 1.0])
            y.append([0])

    else:
        for i in range(100):
            #00000 -> 000
            X.append([0.0, 0.0, 0.0, 0.0, 0.0])
            y.append([0, 0, 0])

            #10000 -> 100
            X.append([1.0, 0.0, 0.0, 0.0, 0.0])
            y.append([1, 0, 0])

            #01000 -> 010
            X.append([0.0, 1.0, 0.0, 0.0, 0.0])
            y.append([0, 1, 0])

            #10100 -> 010
            X.append([1.0, 0.0, 1.0, 0.0, 0.0])
            y.append([0, 1, 0])

            #01010 -> 001
            X.append([0.0, 1.0, 0.0, 1.0, 0.0])
            y.append([0, 0, 1])

            #01001 -> 110
            X.append([0.0, 1.0, 0.0, 0.0, 1.0])
            y.append([1, 1, 0])

            #10101 -> 110
            X.append([1.0, 0.0, 1.0, 0.0, 1.0])
            y.append([1, 1, 0])

            #11111 -> 111
            X.append([1.0, 1.0, 1.0, 1.0, 1.0])
            y.append([1, 1, 1])

    return X, y

def train_test_slit(X, y, split):

    assert len(X) == len(y), "X and y are not same length, can't do train/test split"

    num_train = int(len(X) * split)

    X_train = X[:num_train]
    y_train = y[:num_train]

    X_test = X[num_train:]
    y_test = y[num_train:]

    return X_train, y_train, X_test, y_test