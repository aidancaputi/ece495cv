from nn import *
from datasets import *

def main():

    #create dataset with 90/10 train/test split
    X, y = generate_xor_dataset(100)
    train_test_split = 0.9
    X_train, y_train, X_test, y_test = train_test_slit(X, y, train_test_split)

    n_layers = 3 #number of layers
    input_dim = len(X[0]) #input dimensions
    output_dim = len(y[0]) #output dimensions
    hidden_units = 5 #hidden units in each layer
    learning_rate = 0.03 #learning rate
    num_epochs = 2000 #number of iterations through training data

    #should be able to try changing combinations of the following:
        #number of layers
        #number of units in each hidden layer
        #learning rate
        #train test split
        #epochs

    train(X_train, y_train, X_test, y_test, n_layers, input_dim, output_dim, hidden_units, learning_rate, num_epochs, train_test_split)


if __name__ == "__main__":
    main()