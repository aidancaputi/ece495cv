from nn import *
from datasets import *

def main():

    #should be able to try changing combinations of the following:
        #number of layers
        #number of units in each hidden layer
        #learning rate
        #train test split
        #epochs
    datasets_to_try = ['xor', 'adder']
    splits_to_try = [0.5, 0.6, 0.7, 0.8, 0.9]
    hyperparameters_to_try = [(1, 5), (1,10), (2, 5), (2, 10)] #these are in the form (number of hidden layers, number of units per layer)
    learning_rates_to_try = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    epochs_to_try = [50, 100, 500, 1000, 2000, 5000]

    #train(X_train, y_train, X_test, y_test, n_layers, input_dim, output_dim, hidden_units, learning_rate, num_epochs, train_test_split)
    train2('xor', (3,5), 0.03, 2000, 0.9)


if __name__ == "__main__":
    main()