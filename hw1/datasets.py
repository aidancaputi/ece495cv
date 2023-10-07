def generate_xor_dataset(size):
    X = []
    y = []
    for i in range(size):
        X.append([0, 0])
        y.append([0])
        X.append([0, 1])
        y.append([1])
        X.append([1, 0])
        y.append([1])
        X.append([1, 1])
        y.append([0])

    return X, y