from nn import *

def main():

    #create a dict where keys are 'x', 'y', and 'out'
    X_xor = [
        [[0, 0]],
        [[0, 1]],
        [[1, 0]],
        [[1, 1]],
    ]

    y_xor = [
        [[0]],
        [[1]],
        [[1]],
        [[0]],
    ]

    X_adder = [
        [[0, 0, 0, 0, 0]],
        [[1, 0, 0, 0, 0]],
        [[0, 1, 0, 0, 0]],
        [[1, 0, 1, 0, 0]],
        [[0, 1, 0, 1, 0]],
        [[0, 1, 0, 0, 1]],
        [[1, 0, 1, 0, 1]],
        [[1, 1, 1, 1, 1]],
    ]

    y_adder = [
        [[0, 0, 0]],
        [[1, 0, 0]],
        [[0, 1, 0]],
        [[0, 1, 0]],
        [[0, 0, 1]],
        [[1, 1, 0]],
        [[1, 1, 0]],
        [[1, 1, 1]],
    ]

    X = X_adder
    y = y_adder

    
    n_layers = 2 #number of layers
    input_dim = len(X[0][0]) #input dimensions
    output_dim = len(y[0][0]) #output dimensions
    hidden_units = 5 #hidden units in each layer
    if((hidden_units > input_dim) or (hidden_units < output_dim)):
        print("ERROR: the number of hidden units needs to be between the size of the input and the size of the output.")
        exit(0)
    learning_rate = 0.01 #learning rate
    train_test_split = 0.75 #percent of data to be used for training, remaining is testing
    num_epochs = 10 #number of iterations through training data

    #should be able to try changing combinations of the following:
        #number of layers
        #number of units in each hidden layer
        #learning rate
        #train test split
        #epochs

    train(X, y, n_layers, input_dim, output_dim, hidden_units, learning_rate, train_test_split, num_epochs)


if __name__ == "__main__":
    main()