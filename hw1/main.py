from nn import *
from datasets import *

def main():

    X, y = generate_xor_dataset(100)
    for input, output in zip (X, y):
        print(input, output)
    exit(0)

    n_layers = 2 #number of layers
    input_dim = len(X[0]) #input dimensions
    output_dim = len(y[0]) #output dimensions
    hidden_units = 5 #hidden units in each layer
    learning_rate = 0.03 #learning rate
    train_test_split = 0.75 #percent of data to be used for training, remaining is testing
    num_epochs = 2000 #number of iterations through training data

    #should be able to try changing combinations of the following:
        #number of layers
        #number of units in each hidden layer
        #learning rate
        #train test split
        #epochs

    train(X, y, n_layers, input_dim, output_dim, hidden_units, learning_rate, train_test_split, num_epochs)


if __name__ == "__main__":
    main()