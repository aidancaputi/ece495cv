from nn import *

def main():

    #create a dict where keys are 'x', 'y', and 'out'
    X_xor = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]

    y_xor = [
        [0],
        [1],
        [1],
        [0],
    ]

    
    n_layers = 2 #number of layers
    input_dim = 2 #input dimensions
    output_dim = 1 #output dimensions
    hidden_units = 3 #hidden units in each layer
    learning_rate = 0.1 #learning rate

    train(X_xor, y_xor, n_layers, input_dim, output_dim, hidden_units, learning_rate, 0.75)


if __name__ == "__main__":
    main()